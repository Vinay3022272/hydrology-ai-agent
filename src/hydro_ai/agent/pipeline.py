"""
============================================================
Hydro AI Agent — Pipeline Steps
============================================================
Describes the high-level execution plan that is appended to
the system prompt so the LLM knows the intended workflow.
"""

from __future__ import annotations


def get_pipeline_steps() -> list[str]:
    """Return the ordered pipeline steps."""
    return [
        "Step 1: Understand the user's hydrology question.",
        "Step 2: Call trained model tool if prediction/forecast/risk output is needed.",
        "Step 3: Call analysis tools on the model output to derive interpretation.",
        "Step 4: Call retrieval tool to fetch supporting domain context from vector DB.",
        "Step 5: Merge the model output, analysis, and retrieved evidence.",
        "Step 6: Produce final hydrology answer with practical interpretation.",
    ]


def format_pipeline_steps() -> str:
    """Return the pipeline steps as a formatted string."""
    return "\n".join(get_pipeline_steps())
