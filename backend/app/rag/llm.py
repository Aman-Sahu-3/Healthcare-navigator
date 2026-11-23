"""LLM factory. Swap models via LLM_MODEL in .env — no code changes."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


class LLMNotConfiguredError(RuntimeError):
    pass


def get_llm(settings: Settings) -> BaseChatModel:
    if settings.openai_api_key is None:
        raise LLMNotConfiguredError(
            "OPENAI_API_KEY is not set. Add your OpenAI (or Groq) key to the .env file "
            "in the project root."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_s,
        max_retries=2,
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url or None,
    )
