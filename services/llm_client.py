"""Small provider-compatible LLM adapter with no orchestration-framework dependency."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI


class LLMClient:
    """Expose only the completion operation used by the audit runtime."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        options: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            options["base_url"] = base_url
        self._client = OpenAI(**options)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, *, system: str, user: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return str(response.choices[0].message.content or "").strip()
