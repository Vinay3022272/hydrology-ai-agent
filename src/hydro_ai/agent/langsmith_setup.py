"""
LangSmith setup helpers for Hydro AI.
"""

from __future__ import annotations

import os

from agent.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_ENABLED,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
)


def configure_langsmith() -> dict[str, str | bool]:
    """Configure LangSmith tracing for the current process"""
    if not LANGSMITH_ENABLED:
        return {
            "enabled": False,
            "project": LANGSMITH_PROJECT,
            "reason": "disabled_in_config",
        }

    # LangSmith tracing env vars used by LangChain runtimes.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

    if LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)

    if LANGSMITH_API_KEY:
        os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)

    return {
        "enabled": True,
        "project": LANGSMITH_PROJECT,
        "endpoint": os.environ.get("LANGSMITH_ENDPOINT", ""),
        "api_key_present": bool(os.environ.get("LANGSMITH_API_KEY", "")),
    }
