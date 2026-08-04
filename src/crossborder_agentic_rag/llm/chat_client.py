"""Provider-agnostic chat-completion clients."""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ChatResult:
    content: str
    provider: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    raw: dict[str, Any] | None = None


class BaseChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        pass

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        pass


class TemplateChatClient:
    def __init__(self, provider: str = "template", model: str | None = "template") -> None:
        self.provider = provider
        self.model = model or "template"

    def complete(self, messages: list[dict[str, str]], temperature: float | None = None, max_tokens: int | None = None) -> ChatResult:
        user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        return ChatResult(
            content=(
                "1. Direct answer\n"
                "This offline template answer is grounded only in the provided evidence.\n"
                "2. Key evidence\nUse the cited evidence items in the prompt ([E1], [E2], etc.).\n"
                "3. Risk analysis if applicable\nReview trademark, patent, and litigation signals before listing.\n"
                "4. Suggested seller actions\nValidate rights, avoid risky claims, and keep records.\n"
                "5. Evidence citations\nCitations should refer to [E1], [E2], etc.\n"
                "6. Limitations / missing evidence\nThis is not legal advice; missing evidence may change the assessment.\n\n"
                f"Prompt preview: {user[:160]}"
            ),
            provider=self.provider,
            model=self.model,
        )

    def complete_structured(self, messages, schema_name, temperature=None, max_tokens=None):
        return {
            "schema_name": schema_name,
            "provider": self.provider,
            "model": self.model,
            "content": self.complete(messages, temperature=temperature, max_tokens=max_tokens).content,
        }


class OpenAICompatibleChatClient:
    def __init__(self, api_key: str | None, base_url: str | None, model: str | None, provider: str = "openai_compatible", timeout: float | None = 60.0, default_temperature: float = 0.0, default_max_tokens: int = 800, disable_thinking: bool = True) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.provider = provider
        self.timeout = timeout
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.disable_thinking = disable_thinking

    def complete(self, messages: list[dict[str, str]], temperature: float | None = None, max_tokens: int | None = None) -> ChatResult:
        if not self.api_key:
            return ChatResult("", self.provider, self.model, error="LLM_API_KEY is not set")
        if not self.model:
            return ChatResult("", self.provider, self.model, error="LLM_MODEL is not set")
        try:
            from openai import OpenAI
            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            client = OpenAI(**kwargs)
            request = {
                "model": self.model,
                "messages": messages,
                "temperature": self.default_temperature if temperature is None else temperature,
                "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
            }
            disable_thinking_applied = False
            if self.disable_thinking:
                request["extra_body"] = {"enable_thinking": False}
            try:
                resp = client.chat.completions.create(**request)
                disable_thinking_applied = self.disable_thinking
            except TypeError:
                request.pop("extra_body", None)
                resp = client.chat.completions.create(**request)
            choices = getattr(resp, "choices", None)
            if choices is None:
                return ChatResult("", self.provider, self.model, error="LLM response choices is None")
            if not choices:
                return ChatResult("", self.provider, self.model, error="LLM response choices is empty")
            msg = getattr(choices[0], "message", None)
            content = getattr(msg, "content", None) if msg is not None else None
            if not content:
                return ChatResult("", self.provider, self.model, error="LLM response content is empty")
            usage_obj = getattr(resp, "usage", None)
            usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else (dict(usage_obj) if isinstance(usage_obj, dict) else None)
            raw = {"choices": [{"message": {"content": str(content)}}]}
            if self.disable_thinking and not disable_thinking_applied:
                raw = {
                    "disable_thinking_requested": True,
                    "disable_thinking_applied": False,
                }
            return ChatResult(str(content), self.provider, self.model, usage=usage, raw=raw)
        except Exception as exc:
            return ChatResult("", self.provider, self.model, error=str(exc))

    def complete_structured(self, messages, schema_name, temperature=None, max_tokens=None):
        result = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        content = result.content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
            if content.startswith("json"):
                content = content[4:].strip()
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"schema_name": schema_name, "content": parsed}
        except (json.JSONDecodeError, TypeError):
            return {
                "schema_name": schema_name,
                "error": "structured_output_parse_failed",
            }


def _first(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def build_chat_client(provider: str | None = None, api_key: str | None = None, base_url: str | None = None, model: str | None = None, timeout: float | None = 60.0, default_temperature: float = 0.0, default_max_tokens: int = 800, disable_thinking: bool = True) -> BaseChatClient:
    resolved_provider = (provider or os.getenv("LLM_PROVIDER") or "template").strip()
    name = resolved_provider.lower().replace("-", "_")
    resolved_key = _first(api_key, os.getenv("LLM_API_KEY"), os.getenv("OPENAI_API_KEY"))
    resolved_base = _first(base_url, os.getenv("LLM_BASE_URL"), os.getenv("OPENAI_BASE_URL"))
    resolved_model = _first(model, os.getenv("LLM_MODEL"), os.getenv("OPENAI_MODEL"))
    if name in {"template", "none"}:
        return TemplateChatClient(provider=resolved_provider, model=resolved_model or "template")
    if name in {"openai_compatible", "openai"}:
        return OpenAICompatibleChatClient(resolved_key, resolved_base, resolved_model, provider=resolved_provider, timeout=timeout, default_temperature=default_temperature, default_max_tokens=default_max_tokens, disable_thinking=disable_thinking)
    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")
