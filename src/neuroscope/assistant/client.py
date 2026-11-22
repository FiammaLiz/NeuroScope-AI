"""Compatibility shim for the former direct provider client.

Direct ``ask`` calls bypass the desktop's consent and tool-validation boundaries, so
provider requests must now go through ``AssistantOrchestrator``.
"""

from __future__ import annotations

from .providers import OpenAIProvider, ProviderNotConfigured

AssistantNotConfigured = ProviderNotConfigured


class LLMClient:
    """Deprecated name retained to explain why unmediated requests are disabled."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise AssistantNotConfigured(
            "Direct provider calls are disabled. Use the consent-gated assistant panel and AssistantOrchestrator."
        )


__all__ = ["AssistantNotConfigured", "LLMClient", "OpenAIProvider"]
