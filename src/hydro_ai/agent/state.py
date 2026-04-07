"""
Defines HydroAgentState used by the LangGraph runtime.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class HydroAgentState(TypedDict):
    """State flowing through the hydrology agent graph."""

    # Core message list (auto-merges via add_messages reducer)
    messages: Annotated[list, add_messages]

    # Lifecycle metadata populated by middleware hooks
    run_meta: dict[str, Any]

    # Rolling conversation summary (populated by summarize_and_trim)
    summary: str

    # Index of last summarized message (for rolling summary window)
    last_summarized_index: int
