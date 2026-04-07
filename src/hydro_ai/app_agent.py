"""
============================================================
Hydro AI — Agent CLI Demo
============================================================
Standalone entry point to test the Langchain hydrology agent
from the command line.

Usage:
    cd src/hydro_ai
    python app_agent.py

Requires:
    - Ollama running locally with the configured model
    - pip install langchain-ollama langgraph
"""

from __future__ import annotations

import sys
import os

# Ensure hydro_ai package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.run_agent import run_once, print_stream


def main():
    print("=" * 60)
    print("  Hydro AI — LangGraph Agent Demo")
    print("=" * 60)

    query = "Predict flood susceptibility for this basin and explain the risk drivers."

    # --- Normal run ---
    print("\n=== NORMAL RUN ===\n")
    try:
        result = run_once(query, thread_id="demo-thread-1")
        # Print the last AI message
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            print(getattr(last, "content", str(last)))
        print(f"\nRun meta: {result.get('run_meta', {})}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")

    # --- Streaming run ---
    print("\n=== STREAMING RUN ===\n")
    try:
        print_stream(
            "Run the hydrology workflow and show progress.",
            thread_id="demo-thread-2",
            stream_mode="updates",
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    main()
