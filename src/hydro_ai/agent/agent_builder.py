"""
============================================================
Hydro AI Agent — LangGraph Agent Builder
============================================================
Builds a compiled LangGraph StateGraph with:
  - "agent" node   : LLM with bound tools
  - "tools" node   : ToolNode from langgraph.prebuilt
  - Conditional edge: routes tool_calls → tools, else → END
  - PostgreSQL checkpointer for persistent memory
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from psycopg_pool import ConnectionPool

from agent.config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    GROQ_TIMEOUT,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    LANGSMITH_AGENT_NAME,
    MODEL_NAME,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
    POSTGRES_URI,
    RECENT_MESSAGE_WINDOW,
    SYSTEM_PROMPT,
)
from agent.tools import get_tools
from agent.state import HydroAgentState

logger = logging.getLogger(__name__)


# ── Persistent checkpointer (PostgreSQL) ─────────────────────────────
def _create_checkpointer():
    """Create a PostgresSaver checkpointer using a connection pool."""
    pool = ConnectionPool(
        conninfo=POSTGRES_URI,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # Auto-creates checkpoints + checkpoint_writes tables
    logger.info("PostgreSQL checkpointer ready")
    return checkpointer


_checkpointer = _create_checkpointer()
_cached_agent = None
_active_provider: str = "groq"
_summarizer_llm = None  # Lazy-loaded Ollama instance for summarization


# ── Force Provider State ──
def set_active_provider(provider: str):
    global _active_provider, _cached_agent
    provider = provider.lower()
    if provider in ("groq", "ollama", "gemini") and provider != _active_provider:
        _active_provider = provider
        _cached_agent = None  # Bust cache so it rebuilds!


def _create_llm():
    """
    Try Groq first (fast cloud API, sub-2s).
    If Groq is unavailable or _force_ollama is True, fall back to local Ollama.
    Returns (llm_instance, provider_name).
    """
    #  Fallback to Ollama
    try:
        from langchain_ollama import ChatOllama

        model = FALLBACK_MODEL_NAME or MODEL_NAME
        ollama_llm = ChatOllama(
            model=model,
            base_url=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT,
            num_ctx=8192,
        )
    except Exception as e:
        ollama_llm = None
        ollama_err = str(e)

    if _active_provider == "gemini" and GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            gemini_llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,
            )
            # Smoke test
            gemini_llm.invoke("ping")
            logger.info(f" Using LLM provider: Gemini ({GEMINI_MODEL_NAME})")
            return gemini_llm, "gemini"
        except Exception as e:
            logger.warning(f"Gemini unavailable ({e}), trying fallbacks")

    if _active_provider == "groq" and GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq

            groq_llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                api_key=GROQ_API_KEY,
                temperature=0.3,
                max_retries=0,  # Fail fast on rate limits so we can catch it
                timeout=GROQ_TIMEOUT,
            )
            # Smoke test
            groq_llm.invoke("ping")
            logger.info(f" Using LLM provider: Groq ({GROQ_MODEL_NAME})")
            return groq_llm, "groq"

        except Exception as e:
            logger.warning(
                f"Groq unavailable or rate limited ({e}), falling back to Ollama"
            )

    if _active_provider == "ollama" and ollama_llm:
        logger.info(f" Using LLM provider: Ollama ({model})")
        return ollama_llm, "ollama"

    # ── Fallback ── if requested provider failed, grab whatever works
    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq

            return (
                ChatGroq(model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.3),
                "groq",
            )
        except Exception:
            pass

    if GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return (
                ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL_NAME,
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.3,
                ),
                "gemini",
            )
        except Exception:
            pass

    if ollama_llm:
        logger.info(f" Using fallback LLM provider: Ollama ({model})")
        return ollama_llm, "ollama"
        logger.info(f" Using LLM provider: Ollama ({model})")
        return ollama_llm, "ollama"

    raise RuntimeError(
        "No LLM provider available. " "Set GROQ_API_KEY in .env or start Ollama."
    )


def get_active_provider() -> str:
    """Return the name of the active LLM provider."""
    return _active_provider


def get_summarizer_llm():
    """
    Return a lightweight Ollama LLM dedicated to summarization.
    """
    global _summarizer_llm
    if _summarizer_llm is not None:
        return _summarizer_llm

    from langchain_ollama import ChatOllama

    model = FALLBACK_MODEL_NAME or MODEL_NAME
    _summarizer_llm = ChatOllama(
        model=model,
        base_url=OLLAMA_HOST,
        timeout=OLLAMA_TIMEOUT,
        num_ctx=2048,
    )
    logger.info(f"Summarizer LLM ready: Ollama ({model})")
    return _summarizer_llm


# ══════════════════════════════════════════════════════════════════════
# LangGraph Agent Construction
# ══════════════════════════════════════════════════════════════════════


def _should_continue(state: HydroAgentState) -> Literal["tools", "__end__"]:
    """
    Conditional edge: check if the last AI message has tool_calls.
    If yes → route to "tools" node.  If no → route to END.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def _extract_user_facts(messages: list) -> str:
    """Extract stable user facts from full conversation history (name, preferences)."""
    user_name = None
    preferences: list[str] = []

    name_patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\-']{1,40})",
        r"\bi am\s+([A-Za-z][A-Za-z\-']{1,40})",
        r"\bi'm\s+([A-Za-z][A-Za-z\-']{1,40})",
        r"\bcall me\s+([A-Za-z][A-Za-z\-']{1,40})",
    ]

    for msg in messages:
        if getattr(msg, "type", "") not in ("human", "user"):
            continue

        text = str(getattr(msg, "content", "") or "")
        low = text.lower()

        for pat in name_patterns:
            m = re.search(pat, low, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if candidate:
                    user_name = candidate.title()

        if "remember" in low and len(text) <= 240:
            preferences.append(text.strip())

    facts = []
    if user_name:
        facts.append(f"- User name: {user_name}")
    if preferences:
        # keep only latest 3 short memory cues
        for p in preferences[-3:]:
            facts.append(f"- User memory cue: {p}")

    return "\n".join(facts)


def build_agent(model_name: str | None = None, ollama_host: str | None = None):
    """
    Build and return a compiled LangGraph StateGraph agent.

    The graph has two nodes:
      - "agent"  : calls the LLM (with tools bound)
      - "tools"  : executes tool calls via ToolNode

    Edges:
      START  →  agent
      agent  →  tools   (if tool_calls present)
      agent  →  END     (if no tool_calls — final answer)
      tools  →  agent   (loop back for the LLM to process tool results)
    """
    global _cached_agent, _active_provider

    if _cached_agent is not None:
        return _cached_agent

    # ── 1. LLM — Groq first, Ollama fallback ──
    llm, _active_provider = _create_llm()

    # ── 2. Tools ──
    tools = get_tools()

    # Bind tools to the LLM so it can generate tool_calls
    llm_with_tools = llm.bind_tools(tools)

    # ── 3. System prompt (injected as first message) ──
    full_system_prompt = SYSTEM_PROMPT
    system_message = SystemMessage(content=full_system_prompt)

    #  4. Define graph nodes ──

    def agent_node(state: HydroAgentState) -> dict:
        """
        The agent node: prepend system prompt, call the LLM, return response.
        """
        # Send only recent turns plus rolling summary to control token growth.
        recent_messages = state["messages"][-RECENT_MESSAGE_WINDOW:]
        summary = (state.get("summary") or "").strip()
        user_facts = _extract_user_facts(state.get("messages", []))

        if user_facts:
            facts_block = (
                "\n\nKnown user facts from this thread (trust these if present):\n"
                + user_facts
                + "\nUse them when relevant."
            )
        else:
            facts_block = ""

        if summary:
            summarized_system = SystemMessage(
                content=(
                    full_system_prompt
                    + facts_block
                    + "\n\nConversation summary so far:\n"
                    + summary
                )
            )
            messages = [summarized_system] + recent_messages
        else:
            if facts_block:
                messages = [
                    SystemMessage(content=full_system_prompt + facts_block)
                ] + recent_messages
            else:
                messages = [system_message] + recent_messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ToolNode automatically handles executing tool_calls and returning ToolMessages
    tool_node = ToolNode(tools)

    # ── 5. Build the StateGraph ──
    graph = StateGraph(HydroAgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Add edges
    graph.add_edge(START, "agent")  # Entry point
    graph.add_conditional_edges("agent", _should_continue)  # agent → tools or END
    graph.add_edge("tools", "agent")  # tools → agent (loop back)

    # ── 6. Compile with checkpointer ──
    compiled = graph.compile(checkpointer=_checkpointer)

    _cached_agent = compiled
    return compiled
