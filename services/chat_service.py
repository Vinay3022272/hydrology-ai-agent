import sys
import os

# Ensure the hydro_ai package is in the path
HYDRO_AI_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "hydro_ai")
)
if HYDRO_AI_DIR not in sys.path:
    sys.path.insert(0, HYDRO_AI_DIR)

try:
    from agent.run_agent import run_once

    AGENT_AVAILABLE = True
    AGENT_ERR = None
except ImportError as e:
    print(f"Warning: Could not import LangChain agent: {e}")
    AGENT_AVAILABLE = False
    AGENT_ERR = str(e)

import uuid
from typing import Optional


def _extract_place_name(message: str) -> str:
    """Best-effort extraction of place name from prompts like 'predict flood for Sambalpur'."""
    text = (message or "").strip()
    m = re.search(r"\b(?:for|in|at)\s+([A-Za-z][A-Za-z\s,.-]{1,80})$", text)
    if not m:
        return ""
    place = m.group(1).strip(" .,!?")
    return place


def _tool_fallback_response(message: str) -> str | None:
    """Try direct tool execution when agent runtime fails."""
    msg_lower = message.lower()

    try:
        if "flood" in msg_lower:
            from agent.tools import predict_flood_susceptibility

            place = _extract_place_name(message)
            payload = {"place_name": place} if place else {}
            result = predict_flood_susceptibility.invoke(payload)

            status = result.get("status")
            if status == "success":
                return str(result.get("report", "Flood prediction completed."))
            if status == "geocode_ambiguous":
                cands = result.get("candidates", [])[:3]
                lines = [
                    result.get("message", "Multiple place matches found."),
                    "Top matches:",
                ]
                for c in cands:
                    lines.append(
                        f"- {c.get('name')} ({float(c.get('lat')):.5f}, {float(c.get('lon')):.5f})"
                    )
                return "\n".join(lines)
            return result.get("message", "Flood prediction failed.")

        if "rain" in msg_lower or "precipitation" in msg_lower:
            from agent.tools import predict_rainfall

            place = _extract_place_name(message)
            payload = (
                {"place_name": place, "mode": "point"} if place else {"mode": "basin"}
            )
            result = predict_rainfall.invoke(payload)

            status = result.get("status")
            if status == "success":
                return str(result.get("report", "Rainfall prediction completed."))
            if status == "geocode_ambiguous":
                cands = result.get("candidates", [])[:3]
                lines = [
                    result.get("message", "Multiple place matches found."),
                    "Top matches:",
                ]
                for c in cands:
                    lines.append(
                        f"- {c.get('name')} ({float(c.get('lat')):.5f}, {float(c.get('lon')):.5f})"
                    )
                return "\n".join(lines)
            return result.get("message", "Rainfall prediction failed.")
    except Exception:
        return None

    return None


def process_chat_message(message: str, thread_id: Optional[str] = None) -> dict:
    """
    Processes a chat message.
    Attempts to use the LangChain agent first. If unavailable, falls back to basic routing.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    if AGENT_AVAILABLE:
        try:
            # Call the agent
            result = run_once(message, thread_id=thread_id)

            # Extract the AI's response
            messages = result.get("messages", [])
            ai_response = "I couldn't generate a response."

            # Find the last AIMessage
            for msg in reversed(messages):
                if getattr(msg, "type", "") == "ai":
                    ai_response = _coerce_content_to_text(
                        getattr(msg, "content", str(msg))
                    )
                    break

            return {
                "status": "success",
                "response": ai_response,
                "source": "agent",
                "run_meta": result.get("run_meta", {}),
            }

        except Exception as e:
            print(f"Agent execution failed: {e}. Falling back to basic routing.")
            AGENT_ERR = str(e)
            # Fall through to basic routing

    # Fallback basic keyword routing
    msg_lower = message.lower()

    direct_tool_response = _tool_fallback_response(message)
    if direct_tool_response:
        return {
            "status": "success",
            "response": direct_tool_response,
            "source": "tool_fallback",
        }

    if "flood" in msg_lower:
        response = (
            "It looks like you're asking about floods. You can use the 'Flood Susceptibility' "
            "page to select a location in the Mahanadi Basin and get a precise risk prediction."
        )
    elif "rain" in msg_lower or "precipitation" in msg_lower:
        response = (
            "It looks like you're asking about rainfall. The rainfall prediction module "
            "is currently under development. Please check back later."
        )
    else:
        err_msg = (
            f" (Internal Error: {AGENT_ERR})"
            if not AGENT_AVAILABLE or "AGENT_ERR" in locals()
            else ""
        )
        response = (
            "I am the Hydro AI Agent. I can help answer questions about the Mahanadi Basin, "
            f"flood susceptibility, and rainfall predictions. How can I help you today?{err_msg}"
        )

    return {"status": "success", "response": response, "source": "fallback"}


import json
import math
import re
from typing import Generator


def _json_dumps(obj):
    """Serialize obj to JSON, converting NaN/Inf to null."""
    try:
        return json.dumps(obj)
    except (ValueError, OverflowError):
        # Fallback: allow NaN/Infinity in output, then text-replace with null
        raw = json.dumps(obj, allow_nan=True)
        raw = re.sub(r"\bNaN\b", "null", raw)
        raw = re.sub(r"-?Infinity", "null", raw)
        return raw


def _coerce_content_to_text(content) -> str:
    """Normalize model content blocks/lists into a plain text string."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):
        return "".join(_coerce_content_to_text(part) for part in content)

    if isinstance(content, dict):
        # Common content block shape: {"type": "text", "text": "..."}
        if "text" in content:
            return _coerce_content_to_text(content.get("text"))
        if "content" in content:
            return _coerce_content_to_text(content.get("content"))
        return str(content)

    return str(content)


# Map agent tool names to user-friendly status labels
_TOOL_STATUS_LABELS = {
    "retrieve_hydrology_context": "🔍 Searching knowledge base...",
    "predict_flood_susceptibility": "🌊 Running flood prediction...",
    "predict_rainfall": "🌧️ Running rainfall prediction...",
}


def _tool_status_label(tool_name: str) -> str:
    """Return a user-friendly status label for a tool name."""
    return _TOOL_STATUS_LABELS.get(tool_name, f"⚙️ Running {tool_name}...")


def stream_chat_message(
    message: str, thread_id: Optional[str] = None, provider: str = "groq"
) -> Generator[str, None, None]:
    """
    Streams a chat message.
    Yields JSON string chunks ending with a newline.
    Types: "meta", "status" (live step updates), "token" (AI content), "error".
    Includes a 60-second timeout guard so the user never waits forever.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    if not AGENT_AVAILABLE:
        # Fallback
        res = process_chat_message(message, thread_id)
        yield _json_dumps(
            {"content": res["response"], "thread_id": thread_id, "type": "token"}
        ) + "\n"
        return

    try:
        from agent.run_agent import stream_once
        from agent.agent_builder import get_active_provider, set_active_provider

        # User-selected provider override
        set_active_provider(provider)

        active_provider = get_active_provider()
        yield _json_dumps({"thread_id": thread_id, "type": "meta"}) + "\n"
        yield _json_dumps(
            {"content": f"🤔 Thinking... (via {active_provider})", "type": "status"}
        ) + "\n"

        _seen_tools = set()  # avoid duplicate status for same tool

        import threading
        import time

        # Allow up to 300s for local Ollama, 180s for Cloud APIs to account for heavy tool execution
        _STREAM_TIMEOUT = 300 if provider == "ollama" else 180
        _timed_out = threading.Event()
        _stream_started = time.monotonic()

        for chunk_obj in stream_once(
            message, thread_id=thread_id, stream_mode="messages"
        ):
            # Check timeout
            if time.monotonic() - _stream_started > _STREAM_TIMEOUT:
                yield _json_dumps(
                    {
                        "content": f"\n\n **Response timed out** (>{_STREAM_TIMEOUT}s). The model may be overloaded. Please try again.",
                        "type": "error",
                    }
                ) + "\n"
                return

            if isinstance(chunk_obj, tuple) and len(chunk_obj) > 0:
                msg_chunk = chunk_obj[0]
                msg_type = getattr(msg_chunk, "type", "")

                # LangGraph metadata can indicate active node (agent/tools).
                msg_meta = (
                    chunk_obj[1]
                    if len(chunk_obj) > 1 and isinstance(chunk_obj[1], dict)
                    else {}
                )
                node_name = str(msg_meta.get("langgraph_node", "")).lower()
                if node_name == "tools" and "tools_phase" not in _seen_tools:
                    _seen_tools.add("tools_phase")
                    yield _json_dumps(
                        {
                            "content": "🔍 Searching knowledge base / running tools...",
                            "type": "status",
                        }
                    ) + "\n"

                # --- Detect tool calls and emit status events ---
                tool_calls = getattr(msg_chunk, "tool_calls", None) or []
                tool_call_chunks = getattr(msg_chunk, "tool_call_chunks", None) or []

                for tc in tool_calls + tool_call_chunks:
                    tool_name = (
                        tc.get("name", "")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", "")
                    )
                    if tool_name and tool_name not in _seen_tools:
                        _seen_tools.add(tool_name)
                        label = _tool_status_label(tool_name)
                        yield _json_dumps({"content": label, "type": "status"}) + "\n"

                # ToolMessage fallback: catches cases where tool_calls are not present in chunks.
                if msg_type in ("tool", "ToolMessage"):
                    tool_name = getattr(msg_chunk, "name", "") or "tool"
                    if tool_name not in _seen_tools:
                        _seen_tools.add(tool_name)
                        label = _tool_status_label(tool_name)
                        yield _json_dumps({"content": label, "type": "status"}) + "\n"

                # --- Forward only final AI response tokens ---
                is_tool_call = bool(tool_calls or tool_call_chunks)
                if msg_type in ("AIMessageChunk", "ai") and not is_tool_call:
                    content = _coerce_content_to_text(getattr(msg_chunk, "content", ""))
                    if content:
                        # Emit a composing status on first token
                        if "composing" not in _seen_tools:
                            _seen_tools.add("composing")
                            yield _json_dumps(
                                {"content": "✍️ Composing answer...", "type": "status"}
                            ) + "\n"
                        yield _json_dumps({"content": content, "type": "token"}) + "\n"

    except Exception as e:
        err_str = str(e).lower()

        # Gemini tool-calling can fail in streaming with thought_signature errors,
        # while regular invoke still succeeds. Retry once via non-stream path.
        if (
            "thought_signature" in err_str
            or "functioncall parts" in err_str
            or "missing a thought_signature" in err_str
        ):
            try:
                from agent.run_agent import run_once

                result = run_once(message, thread_id=thread_id)
                messages = result.get("messages", [])
                ai_response = "I couldn't generate a response."
                for msg in reversed(messages):
                    if getattr(msg, "type", "") == "ai":
                        ai_response = _coerce_content_to_text(
                            getattr(msg, "content", str(msg))
                        )
                        break

                yield _json_dumps({"content": ai_response, "type": "token"}) + "\n"
                return
            except Exception as retry_err:
                e = retry_err
                err_str = str(retry_err).lower()

        if "429" in err_str or "rate limit" in err_str:
            print(f"Groq Rate Limit hit! Instructing user to toggle to Ollama.")
            yield _json_dumps(
                {
                    "content": "\n\n **Groq API Rate Limit Hit!** Please switch the **Model Provider** toggle in the sidebar to **Ollama (Free, Local)** and click 'Ask' again.",
                    "type": "error",
                }
            ) + "\n"
        else:
            print(f"Agent stream failed: {e}")
            res = process_chat_message(message, thread_id)
            if res.get("source") == "tool_fallback":
                yield _json_dumps({"content": res["response"], "type": "token"}) + "\n"
                return

            yield _json_dumps(
                {
                    "content": f"\n\n**Note:** Agent encountered an error ({e}). Showing fallback response:\n\n{res['response']}",
                    "type": "error",
                }
            ) + "\n"
