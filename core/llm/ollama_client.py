import os
from typing import Dict, List, Optional
import httpx


class OllamaClient:
    """
    Client for interacting with local Ollama instance using the native Chat API
    with full RTX GPU offload, thread acceleration, and automatic fallback on 500 errors.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2:3b",
        num_threads: Optional[int] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.num_threads = num_threads or os.cpu_count() or 12

    def list_local_models(self) -> List[str]:
        """Returns a list of models currently installed in local Ollama."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
        return []

    def chat_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> str:
        """
        Sends chat messages to Ollama `/api/chat` endpoint with automatic fallback if a model crashes.
        """
        selected_model = model or self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_thread": self.num_threads,
                "num_ctx": 4096,
                "num_gpu": 99
            }
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                res = client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code == 200:
                    return res.json().get("message", {}).get("content", "").strip()

                err_text = res.text
                print(f"[!] Ollama model {selected_model} returned {res.status_code}. Attempting fallback...")

                # Auto-fallback to reliable text models
                for fallback_model in ["llama3.2:3b", "llama3.2:1b", "llama3.2"]:
                    if fallback_model != selected_model:
                        try:
                            payload["model"] = fallback_model
                            res_fb = client.post(f"{self.base_url}/api/chat", json=payload)
                            if res_fb.status_code == 200:
                                print(f"[OK] Fallback to {fallback_model} succeeded.")
                                return res_fb.json().get("message", {}).get("content", "").strip()
                        except Exception:
                            pass

                raise RuntimeError(f"Ollama error: {err_text}")
        except Exception as e:
            if "Ollama error:" in str(e):
                raise e
            raise RuntimeError(f"Ollama connection error: {str(e)}")
