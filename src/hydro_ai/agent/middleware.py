"""
Middleware hooks for the Hydro AI LangGraph agent.

Includes:
  - on_agent_start / on_agent_end  — timestamp bookkeeping
  - log_last_response              — debug logger
  - should_summarize               — checks if summarization is needed
  - summarize_and_trim             — summarizes old messages and deletes them
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent.config import SUMMARIZE_THRESHOLD

logger = logging.getLogger(__name__)


# Run summary generation off the request path.
_summary_executor = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="hydro-summary"
)
_active_summary_jobs: set[str] = set()
_summary_jobs_lock = threading.Lock()


#  BEFORE-AGENT HOOK 
def on_agent_start(state: dict) -> dict[str, Any]:
    """Stamp the start time into run_meta."""
    run_meta = dict(state.get("run_meta", None) or {})
    run_meta["started_at"] = datetime.now(timezone.utc).isoformat()
    return {"run_meta": run_meta}


#  AFTER-MODEL HOOK (debug logger) 
def log_last_response(state: dict) -> dict[str, Any] | None:
    """Print the last AI response for debugging."""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))
        print(f"\n[after_model] last response preview ({len(content)} chars):")
        print(content[:500])
    return None


# AFTER-AGENT HOOK 
def on_agent_end(state: dict) -> dict[str, Any]:
    """Stamp the finish time into run_meta."""
    run_meta = dict(state.get("run_meta", None) or {})
    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    return {"run_meta": run_meta}


# ── ROLLING SUMMARY MEMORY OPTIMIZATION ──────────────────────────────
#
# Rolling Summary Pattern:
#   Step 1: S1 = summarize(M1 to M6)           → first summary
#   Step 2: S2 = summarize(S1 + M7 to M12)     → extends summary
#   Step 3: S3 = summarize(S2 + M13 to M18)    → extends again
#
# Ollama always processes: old_summary + ~6 new messages (never full history)
# DB keeps: ALL messages intact (for Streamlit sidebar)


def should_summarize(state: dict) -> bool:
    """
    Returns True if there are enough NEW (un-summarized) messages
    to warrant a new rolling summary.
    """
    messages = state.get("messages", [])
    last_idx = state.get("last_summarized_index", 0)

    # Count how many new messages exist since last summarization
    new_message_count = len(messages) - last_idx
    return new_message_count > SUMMARIZE_THRESHOLD


def summarize_and_trim(agent, config: dict) -> None:
    """
    Rolling summary — only summarize NEW messages since last summary.

    How it works:
    1. Get current state from checkpoint
    2. Find messages added since last summarization (using last_summarized_index)
    3. Send ONLY old_summary + new_messages to Ollama (keeps input small)
    4. Update summary and advance last_summarized_index
    5. DB keeps ALL messages — nothing is deleted

    Example:
      Turn 6:   S1 = summarize(M1–M6)               input: 6 msgs
      Turn 12:  S2 = summarize(S1 + M7–M12)          input: summary + 6 msgs
      Turn 18:  S3 = summarize(S2 + M13–M18)         input: summary + 6 msgs
      → Ollama input stays constant regardless of total conversation length
    """

    def _summarize_job(job_thread_id: str) -> None:
        from agent.agent_builder import get_summarizer_llm

        try:
            snapshot = agent.get_state(config)
            state = snapshot.values
            messages = state.get("messages", [])
            last_idx = state.get("last_summarized_index", 0)

            new_message_count = len(messages) - last_idx
            if new_message_count <= SUMMARIZE_THRESHOLD:
                return

            new_messages = messages[last_idx:]

            logger.info(
                f"Rolling summary: processing {len(new_messages)} new messages "
                f"(msgs {last_idx + 1} to {len(messages)})"
            )

            existing_summary = state.get("summary", "")
            if existing_summary:
                prompt = (
                    f"Existing conversation summary:\n{existing_summary}\n\n"
                    "Extend this summary with the new messages above. "
                    "Keep it concise (3-5 sentences max)."
                )
            else:
                prompt = (
                    "Summarize the conversation above in 3-5 concise sentences. "
                    "Focus on the key topics discussed and any important results."
                )

            summarizer = get_summarizer_llm()
            summary_input = new_messages + [HumanMessage(content=prompt)]

            response = summarizer.invoke(summary_input)
            new_summary = response.content
            logger.info(f"Rolling summary generated ({len(new_summary)} chars)")

            agent.update_state(
                config,
                {
                    "summary": new_summary,
                    "last_summarized_index": len(messages),
                },
            )

            logger.info(
                f"Summary updated. Index advanced to {len(messages)}. "
                f"Full history preserved in DB."
            )
        except Exception as e:
            logger.warning(f"Summarization failed: {e}. Skipping.")
        finally:
            with _summary_jobs_lock:
                _active_summary_jobs.discard(job_thread_id)

    thread_id = str(config.get("configurable", {}).get("thread_id", "default"))
    with _summary_jobs_lock:
        if thread_id in _active_summary_jobs:
            logger.info("Summarization already running for thread_id=%s", thread_id)
            return
        _active_summary_jobs.add(thread_id)

    _summary_executor.submit(_summarize_job, thread_id)
    logger.info("Summarization scheduled in background for thread_id=%s", thread_id)
