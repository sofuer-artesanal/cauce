from __future__ import annotations
import os
import requests
from typing import Protocol


class LLMProvider(Protocol):
    def call(self, prompt: str) -> str:
        ...


class OpenRouterLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL")
        self.base_url = base_url or "https://openrouter.ai/api/v1/chat/completions"
        if not self.api_key or not self.model:
            raise EnvironmentError("OpenRouterLLM requires LLM_API_KEY and LLM_MODEL environment variables")

    def call(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 800,
        }
        resp = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        # Expect OpenRouter-like response: choices[0].message.content
        try:
            return j["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Invalid response from LLM: {e} | {j}")
