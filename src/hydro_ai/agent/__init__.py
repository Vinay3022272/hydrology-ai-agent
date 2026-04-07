"""
Hydro AI Agent Package
======================
LangGraph-based hydrology analysis agent with StateGraph,
ToolNode, and conditional edges.

Usage:
    from agent.agent_builder import build_agent
    from agent.run_agent import run_once, stream_once, run_batch
"""

from __future__ import annotations

from agent.agent_builder import build_agent
from agent.run_agent import run_once, stream_once, run_batch, print_stream

__all__ = [
    "build_agent",
    "run_once",
    "stream_once",
    "run_batch",
    "print_stream",
]
