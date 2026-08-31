"""
Multi-Provider LLM & Cloud API Client for Mi:RAG
Supports local Ollama (default) alongside OpenAI, Google Gemini, OpenRouter, Groq, and Anthropic Claude.
"""

import json
from typing import Any, Dict, List, Optional
import httpx


class MultiProviderLLM:
    """Unified dispatcher for offline Ollama and optional cloud LLM providers."""

    DEFAULT_MODELS = {
        "ollama": "llama3.2:3b",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "openrouter": "deepseek/deepseek-r1",
        "groq": "llama-3.3-70b-versatile",
        "anthropic": "claude-3-5-haiku-20241022"
    }

    PROVIDER_PRESETS = [
        {
            "id": "ollama",
            "name": "Local Ollama (Offline / Zero-Cost)",
            "models": ["llama3.2:3b", "llama3.2:1b", "qwen2.5:3b", "qwen2.5:7b", "mistral:latest", "gemma2:2b"],
            "requires_key": False
        },
        {
            "id": "openai",
            "name": "OpenAI",
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "o3-mini"],
            "requires_key": True,
            "placeholder": "sk-proj-..."
        },
        {
            "id": "gemini",
            "name": "Google Gemini",
            "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            "requires_key": True,
            "placeholder": "AIzaSy..."
        },
        {
            "id": "openrouter",
            "name": "OpenRouter",
            "models": [
                "deepseek/deepseek-r1",
                "anthropic/claude-3.5-sonnet",
                "meta-llama/llama-3.3-70b-instruct",
                "google/gemini-2.0-flash-001",
                "qwen/qwen-2.5-72b-instruct"
            ],
            "requires_key": True,
            "placeholder": "sk-or-v1-..."
        },
        {
            "id": "groq",
            "name": "Groq Cloud",
            "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            "requires_key": True,
            "placeholder": "gsk_..."
        },
        {
            "id": "anthropic",
            "name": "Anthropic Claude",
            "models": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
            "requires_key": True,
            "placeholder": "sk-ant-api03-..."
        }
    ]

    @classmethod
    def get_provider_presets(cls) -> List[Dict[str, Any]]:
        return cls.PROVIDER_PRESETS

    @classmethod
    def test_key(cls, provider: str, api_key: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Validates API key by firing a minimal ping prompt to the provider."""
        provider = (provider or "ollama").lower().strip()
        api_key = (api_key or "").strip()

        if provider == "ollama":
            try:
                with httpx.Client(timeout=3.0) as client:
                    res = client.get("http://localhost:11434/api/tags")
                    if res.status_code == 200:
                        return {"valid": True, "message": "Ollama local engine is online and connected."}
                    return {"valid": False, "message": f"Ollama returned status {res.status_code}."}
            except Exception as e:
                return {"valid": False, "message": f"Could not reach Ollama: {str(e)}"}

        if not api_key:
            return {"valid": False, "message": f"API key for {provider} cannot be empty."}

        target_model = model or cls.DEFAULT_MODELS.get(provider, "gpt-4o-mini")

        try:
            with httpx.Client(timeout=10.0) as client:
                if provider == "openai":
                    res = client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": target_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
                    )
                elif provider == "openrouter":
                    res = client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://mirag.me",
                            "X-Title": "Mi:RAG"
                        },
                        json={"model": target_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
                    )
                elif provider == "groq":
                    res = client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": target_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
                    )
                elif provider == "gemini":
                    clean_m = target_model.replace("models/", "")
                    res = client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{clean_m}:generateContent?key={api_key}",
                        headers={"Content-Type": "application/json"},
                        json={"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 5}}
                    )
                elif provider == "anthropic":
                    res = client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json"
                        },
                        json={"model": target_model, "max_tokens": 5, "messages": [{"role": "user", "content": "ping"}]}
                    )
                else:
                    return {"valid": False, "message": f"Unsupported provider '{provider}'."}

                if res.status_code in [200, 201]:
                    return {"valid": True, "message": f"Valid API Key! Successfully connected to {provider.upper()} ({target_model})."}
                else:
                    err_msg = res.text[:200]
                    try:
                        err_json = res.json()
                        err_msg = err_json.get("error", {}).get("message", err_msg)
                    except Exception:
                        pass
                    return {"valid": False, "message": f"{provider.upper()} authentication failed ({res.status_code}): {err_msg}"}
        except Exception as e:
            return {"valid": False, "message": f"Connection error: {str(e)}"}

    @classmethod
    def generate(
        cls,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        ollama_url: str = "http://localhost:11434",
        timeout: float = 90.0
    ) -> str:
        """Executes grounded chat completion across the selected provider."""
        provider = (provider or "ollama").lower().strip()
        model_name = model or cls.DEFAULT_MODELS.get(provider, "llama3.2:3b")

        # 1. Local Ollama Fallback / Default
        if provider == "ollama" or not api_key:
            with httpx.Client(timeout=timeout) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                res = client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temperature}
                    }
                )
                res.raise_for_status()
                return res.json().get("message", {}).get("content", "").strip()

        # 2. OpenAI
        if provider == "openai":
            with httpx.Client(timeout=timeout) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                res = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model_name, "messages": messages, "temperature": temperature}
                )
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()

        # 3. OpenRouter
        if provider == "openrouter":
            with httpx.Client(timeout=timeout) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                res = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://mirag.me",
                        "X-Title": "Mi:RAG Multimodal"
                    },
                    json={"model": model_name, "messages": messages, "temperature": temperature}
                )
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()

        # 4. Groq
        if provider == "groq":
            with httpx.Client(timeout=timeout) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                res = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model_name, "messages": messages, "temperature": temperature}
                )
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()

        # 5. Google Gemini
        if provider == "gemini":
            with httpx.Client(timeout=timeout) as client:
                clean_m = model_name.replace("models/", "")
                full_text = f"System Instructions:\n{system_prompt}\n\nUser Question:\n{user_prompt}" if system_prompt else user_prompt
                res = client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{clean_m}:generateContent?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": full_text}]}],
                        "generationConfig": {"temperature": temperature}
                    }
                )
                res.raise_for_status()
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    return "".join(p.get("text", "") for p in parts).strip()
                return "Gemini returned an empty response."

        # 6. Anthropic Claude
        if provider == "anthropic":
            with httpx.Client(timeout=timeout) as client:
                payload = {
                    "model": model_name,
                    "max_tokens": 4096,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": user_prompt}]
                }
                if system_prompt:
                    payload["system"] = system_prompt

                res = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                res.raise_for_status()
                data = res.json()
                content_blocks = data.get("content", [])
                return "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text").strip()

        raise ValueError(f"Unknown LLM provider: {provider}")
