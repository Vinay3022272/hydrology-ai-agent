"""
============================================================
Hydro AI Agent — Runner
============================================================
Convenience functions for invoking the agent:

  run_once      — single synchronous invocation
  stream_once   — streaming generator
  run_batch     — batch multiple queries
  print_stream  — stream and print to stdout
"""

from __future__ import annotations

from typing import Generator

from agent.agent_builder import build_agent
from agent.config import DEFAULT_THREAD_ID
from agent.middleware import (
    on_agent_start,
    on_agent_end,
    log_last_response,
    should_summarize,
    summarize_and_trim,
)


# Single run
def run_once(
    user_query: str,
    thread_id: str = DEFAULT_THREAD_ID,
) -> dict:
    """
    Run the agent with a single query and return the full result.

    Parameters
    ----------
    user_query : str
        The user's hydrology question.
    thread_id : str
        Thread ID for memory continuity.
    """
    agent = build_agent()
    config = {"configurable": {"thread_id": thread_id}}

    # Pre-hook: stamp start time
    pre_state = on_agent_start({"run_meta": {}})

    result = agent.invoke(
        {"messages": [("user", user_query)]},
        config,
    )

    # Post-hooks
    log_last_response(result)
    post_state = on_agent_end(result)

    # Summarize old messages if conversation is getting long
    if should_summarize(result):
        summarize_and_trim(agent, config)

    # Merge run_meta
    result["run_meta"] = {
        **pre_state.get("run_meta", {}),
        **post_state.get("run_meta", {}),
    }

    return result


# Streaming

def stream_once(
    user_query: str,
    thread_id: str = DEFAULT_THREAD_ID,
    stream_mode: str = "updates",
) -> Generator:
    """
    Stream agent output chunk by chunk.

    Parameters
    ----------
    user_query : str
        The user's hydrology question.
    thread_id : str
        Thread ID for memory continuity.
    stream_mode : str
        One of "updates", "messages", or "custom".
    """
    agent = build_agent()
    config = {"configurable": {"thread_id": thread_id}}

    for chunk in agent.stream(
        {"messages": [("user", user_query)]},
        config,
        stream_mode=stream_mode,
    ):
        yield chunk

    # After streaming completes, check if summarization is needed
    snapshot = agent.get_state(config)
    if should_summarize(snapshot.values):
        summarize_and_trim(agent, config)


# Batch

def run_batch(
    queries: list[str],
    thread_prefix: str = "hydro-batch",
) -> list[dict]:
    """
    Run multiple queries in batch.

    Each query gets its own thread ID based on the prefix.
    """
    agent = build_agent()

    inputs = [
        {"messages": [("user", q)]}
        for q in queries
    ]

    configs = [
        {"configurable": {"thread_id": f"{thread_prefix}-{i}"}}
        for i in range(len(queries))
    ]

    results = agent.batch(inputs, config=configs)
    return results


# Print-stream (convenience)

def print_stream(
    user_query: str,
    thread_id: str = DEFAULT_THREAD_ID,
    stream_mode: str = "updates",
) -> None:
    """Stream and print each chunk to stdout."""
    for chunk in stream_once(
        user_query,
        thread_id=thread_id,
        stream_mode=stream_mode,
    ):
        print(chunk)
