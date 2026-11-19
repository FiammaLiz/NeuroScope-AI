from __future__ import annotations
import os


class AssistantNotConfigured(RuntimeError):
    pass


class LLMClient:
    """Thin provider adapter. The assistant remains optional/offline by design."""
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL")
        self.api_key = os.getenv("OPENAI_API_KEY")

    def ask(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise AssistantNotConfigured("Set OPENAI_API_KEY to enable the AI assistant.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AssistantNotConfigured("Install the optional dependency with: pip install -e '.[ai]'") from exc
        if not self.model:
            raise AssistantNotConfigured("Set OPENAI_MODEL to the model you want to use.")
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content or ""
