"""Centralised LLM client factory.

All nodes call ``get_chat_model()`` instead of instantiating ``ChatAnthropic``
directly. This ensures every LLM call inherits the configured max_retries and
timeout, so transient failures (429, 5xx, connection errors) are retried by the
Anthropic SDK before bubbling up to the node-level try/except.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from app.config.settings import settings


def get_chat_model() -> ChatAnthropic:
    """Return a ChatAnthropic client configured with project-wide retries + timeout."""
    return ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_tokens,
        max_retries=settings.max_retries,
        timeout=settings.llm_timeout_seconds,
    )
