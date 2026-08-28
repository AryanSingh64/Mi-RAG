import os
from typing import Dict, List, Optional
import httpx


class OllamaClient:
    """
    Client for interacting with local Ollama instance using the native Chat API
    with full RTX GPU offload and 12-thread acceleration.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2:1b",
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
        Sends chat messages to Ollama `/api/chat` endpoint with full GPU acceleration.
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

        with httpx.Client(timeout=120.0) as client:
            res = client.post(f"{self.base_url}/api/chat", json=payload)
            res.raise_for_status()
            return res.json().get("message", {}).get("content", "").strip()
