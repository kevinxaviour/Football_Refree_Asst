"""LangChain LLM router with provider fallback and structured outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Type, Optional

from pydantic import BaseModel


@dataclass(frozen=True)
class LLMProvider:
    name: str
    model: str
    base_url: str | None = None


def _provider_order() -> list[LLMProvider]:
    return [
        LLMProvider(name="openai", model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        LLMProvider(
            name="groq", model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        ),
        LLMProvider(
            name="openrouter",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        ),
    ]


def _has_credentials(provider: LLMProvider) -> bool:
    if provider.name == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider.name == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    if provider.name == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return False


def _build_chat_model(provider: LLMProvider):
    if provider.name == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=provider.model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )

    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    kwargs: dict[str, Any] = {}
    if provider.name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        kwargs["base_url"] = provider.base_url
        kwargs["default_headers"] = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "ai-referee-assistant"),
        }

    return ChatOpenAI(
        model=provider.model,
        api_key=api_key,
        temperature=0.1,
        **kwargs,
    )


def invoke_structured(
    *,
    schema: Type[BaseModel],
    messages: Iterable[Any],
    config: dict[str, Any] | None = None,
) -> tuple[BaseModel, str]:
    errors: list[str] = []

    for provider in _provider_order():
        if not _has_credentials(provider):
            continue

        try:
            model = _build_chat_model(provider)
            structured_model = model.with_structured_output(schema)
            result = structured_model.invoke(list(messages), config=config)
            return result, provider.name
        except Exception as exc:
            errors.append(f"{provider.name}: {exc}")

    if not errors:
        raise RuntimeError(
            "No LLM credentials found. Configure OPENAI_API_KEY (primary), "
            "plus GROQ_API_KEY and/or OPENROUTER_API_KEY for fallback."
        )

    raise RuntimeError("All LLM providers failed. " + " | ".join(errors))


def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    chat_history: Optional[list] = None,
) -> str:
    """Backward-compatible text generation helper."""
    from langchain_core.messages import HumanMessage, SystemMessage

    errors: list[str] = []
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=prompt))

    for provider in _provider_order():
        if not _has_credentials(provider):
            continue
        try:
            model = _build_chat_model(provider)
            return model.invoke(messages).content
        except Exception as exc:
            errors.append(f"{provider.name}: {exc}")

    if not errors:
        raise RuntimeError("No LLM credentials found.")
    raise RuntimeError("All LLM providers failed. " + " | ".join(errors))


def get_active_backend() -> str:
    """Return first configured provider in fallback order."""
    for provider in _provider_order():
        if _has_credentials(provider):
            return provider.name
    return "none"
