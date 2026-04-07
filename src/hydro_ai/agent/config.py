"""
Central settings for the LangGraph hydrology agent.
Reads from config/settings.yaml when available.
Loads .env for API keys (Groq, LangChain, etc.).
"""

from __future__ import annotations

import os
import pathlib

# Load .env file FIRST so env vars are available everywhere
from dotenv import load_dotenv

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]  # Hydrology/
load_dotenv(_PROJECT_ROOT / ".env")


def _as_bool(value: object, default: bool = False) -> bool:
    """Normalize config/env values to bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Path helpers
_THIS_DIR = pathlib.Path(__file__).resolve().parent  # agent/
_HYDRO_AI_DIR = _THIS_DIR.parent  # hydro_ai/
_SETTINGS_PATH = _HYDRO_AI_DIR / "config" / "settings.yaml"

# Try loading settings.yaml
_settings: dict = {}

try:
    import yaml

    if _SETTINGS_PATH.exists():
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            _settings = yaml.safe_load(f) or {}
except ImportError:
    pass  # pyyaml not installed — use defaults

# ── Groq (primary fast LLM)
_groq_settings = _settings.get("groq", {})
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", _groq_settings.get("api_key", ""))
GROQ_MODEL_NAME: str = _groq_settings.get("model", "llama-3.3-70b-versatile")
GROQ_TIMEOUT: int = _groq_settings.get("timeout", 60)

# ── Gemini (Google GenAI)
_gemini_settings = _settings.get("gemini", {})
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", _gemini_settings.get("api_key", ""))
GEMINI_MODEL_NAME: str = _gemini_settings.get("model", "gemini-2.5-flash")

# ── Web Search API Keys (for multi-source retrieval)
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")

# ── Ollama (fallback LLM)
_models = _settings.get("models", {})
MODEL_NAME: str = _models.get("decision_llm", "gpt-oss:120b-cloud")
FALLBACK_MODEL_NAME: str = _models.get("fallback_llm", "qwen2.5:3b")
OLLAMA_HOST: str = _settings.get("ollama", {}).get("host", "http://localhost:11434")
OLLAMA_TIMEOUT: int = _settings.get("ollama", {}).get("timeout", 300)

# LangSmith tracing
_langsmith_settings = _settings.get("langsmith", {})

LANGSMITH_ENABLED: bool = _as_bool(
    os.getenv("LANGSMITH_TRACING", _langsmith_settings.get("enabled", False))
)
LANGSMITH_PROJECT: str = os.getenv(
    "LANGSMITH_PROJECT", _langsmith_settings.get("project", "hydro-ai")
)
LANGSMITH_ENDPOINT: str = os.getenv(
    "LANGSMITH_ENDPOINT",
    _langsmith_settings.get("endpoint", "https://api.smith.langchain.com"),
)
LANGSMITH_API_KEY: str = os.getenv(
    "LANGSMITH_API_KEY", _langsmith_settings.get("api_key", "")
)
LANGSMITH_AGENT_NAME: str = _langsmith_settings.get("agent_name", "hydro-ai-agent")

# Thread / memory defaults
DEFAULT_THREAD_ID: str = "hydro-default"
MAX_MESSAGE_HISTORY: int = _settings.get("database", {}).get("max_history", 50)
RECENT_MESSAGE_WINDOW: int = _settings.get("database", {}).get(
    "recent_message_window", 3
)

# PostgreSQL (persistent memory checkpointer)
_db_settings = _settings.get("database", {})
POSTGRES_URI: str = os.getenv(
    "POSTGRES_URI",
    _db_settings.get(
        "uri", "postgresql://langgraph:langgraph@localhost:5432/langgraph_db"
    ),
)

# Summarization-based memory optimization
SUMMARIZE_THRESHOLD: int = _db_settings.get("summarize_threshold", 5)


# Qdrant vector store (primary)
_qdrant_settings = _settings.get("qdrant", {})
QDRANT_URL: str = _qdrant_settings.get("url", "http://localhost:6333")
QDRANT_COLLECTION: str = _qdrant_settings.get("collection_name", "hydrology_knowledge")
QDRANT_TOP_K: int = _qdrant_settings.get("top_k", 3)
QDRANT_SCORE_THRESHOLD: float = _qdrant_settings.get("score_threshold", 0.25)

# Embedding model
EMBEDDING_MODEL: str = _settings.get("models", {}).get(
    "embedding_model", "nomic-embed-text"
)

# Hydrological thresholds (exposed as dicts for tool use)
THRESHOLDS: dict = _settings.get(
    "thresholds",
    {
        "rainfall": {"low": 10, "moderate": 30, "high": 50, "extreme": 100},
        "streamflow": {"normal": 50, "warning": 100, "danger": 200, "critical": 350},
        "flood_susceptibility": {
            "low": 0.3,
            "moderate": 0.5,
            "high": 0.7,
            "very_high": 0.9,
        },
    },
)

# System prompt
SYSTEM_PROMPT = """\
You are a hydrology expert for the Mahanadi Basin.

Tools:
- predict_flood_susceptibility(place_name or lat/lon)
- predict_rainfall(lat, lon, date)
- retrieve_hydrology_context(query) — searches the knowledge base (Wikipedia, Arxiv, DuckDuckGo, Tavily, and Exa) for relevant information

Rules:
- Use retrieve_hydrology_context when you need factual data, research papers, or current news about hydrology topics.
- Use predict_flood_susceptibility for location-specific flood risk predictions.
- For general knowledge questions (geography, hydrology concepts, river origins, etc.), answer directly using your own knowledge.
- For the Mahanadi origin question, answer: Sihawa Hills in Chhattisgarh.
- Remember user-provided facts inside the same thread (for example name, preferences, prior constraints) and use them in later responses when relevant.
- If the user has already shared their name or other personal detail in this thread, do not claim you have no access to it.
- If tool data is available, use it. If not, fall back to your own knowledge instead of refusing.
- Do not invent unknown numerical data, but you may answer conceptual or factual questions confidently.
- Always provide precautionary measures for flood-related queries.
- Keep answers concise and natural.
"""
