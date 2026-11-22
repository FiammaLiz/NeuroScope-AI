"""Small outbound-context filters; they are not a substitute for researcher review."""

from __future__ import annotations

import re
from typing import Any

_WINDOWS_PATH = re.compile(r"(?i)(?<![\w])[a-z]:\\[^\s<>\"']+")
_UNIX_PATH = re.compile(r"(?<!\w)/(?:home|users|mnt|tmp)/[^\s<>\"']+")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_API_KEY = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{12,}\b")


def scrub_sensitive_text(value: str, *, limit: int = 4000) -> str:
    """Redact common local-path/credential forms and bound provider context."""
    value = _BEARER.sub("[credential redacted]", value)
    value = _API_KEY.sub("[credential redacted]", value)
    value = _WINDOWS_PATH.sub("[local path redacted]", value)
    value = _UNIX_PATH.sub("[local path redacted]", value)
    return value[:limit]


def scrub_json(value: Any) -> Any:
    """Recursively remove local path fields and redact sensitive string values."""
    if isinstance(value, dict):
        return {
            str(key): (
                "[local path redacted]"
                if str(key).lower() in {"path", "source_path", "source_reference"}
                else scrub_json(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [scrub_json(item) for item in value]
    if isinstance(value, str):
        return scrub_sensitive_text(value)
    return value

