import requests
import streamlit as st

# Base URL for the FastAPI backend
API_BASE_URL = "http://localhost:8000"


def get_health_status() -> bool:
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def predict_flood(lat: float, lon: float) -> dict | None:
    """Call the flood prediction endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/flood", json={"lat": lat, "lon": lon}, timeout=30
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None


def predict_rainfall(
    lat: float | None = None,
    lon: float | None = None,
    date: str | None = None,
    mode: str = "point",
) -> dict | None:
    """Call the rainfall prediction endpoint."""
    try:
        payload = {"date": date, "mode": mode}
        if lat is not None:
            payload["lat"] = lat
        if lon is not None:
            payload["lon"] = lon
        response = requests.post(
            f"{API_BASE_URL}/predict/rainfall", json=payload, timeout=180
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None


import json
from typing import Optional, Generator


def stream_chat_message(
    message: str, thread_id: Optional[str] = None, provider: str = "ollama"
) -> Generator[dict, None, None]:
    """
    Stream a message from the chat agent.
    Yields dicts: {"type": "status"|"token"|"error", "content": "..."}
    """
    try:
        payload = {"message": message, "provider": provider}
        if thread_id:
            payload["thread_id"] = thread_id
        response = requests.post(
            f"{API_BASE_URL}/chat/stream", json=payload, timeout=300, stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                msg_type = data.get("type", "")
                content = data.get("content", "")
                if msg_type in ("status", "token", "error") and content:
                    yield {"type": msg_type, "content": content}
    except requests.exceptions.Timeout:
        yield {"type": "error", "content": "The agent took too long to respond."}
    except Exception as e:
        yield {"type": "error", "content": f"Error connecting to backend: {e}"}


def stream_chat_message_sse(
    message: str, thread_id: Optional[str] = None, provider: str = "ollama"
) -> Generator[dict, None, None]:
    """
    Stream chat using Server-Sent Events endpoint.
    Yields dicts: {"type": "status"|"token"|"error", "content": "..."}
    """
    try:
        payload = {"message": message, "provider": provider}
        if thread_id:
            payload["thread_id"] = thread_id

        response = requests.post(
            f"{API_BASE_URL}/chat/sse",
            json=payload,
            timeout=300,
            stream=True,
            headers={"Accept": "text/event-stream"},
        )
        response.raise_for_status()

        for raw in response.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if not payload:
                continue
            data = json.loads(payload)
            msg_type = data.get("type", "")
            content = data.get("content", "")
            if msg_type in ("status", "token", "error") and content:
                yield {"type": msg_type, "content": content}

    except requests.exceptions.Timeout:
        yield {"type": "error", "content": "The agent took too long to respond."}
    except Exception as e:
        # Transparent fallback to the existing NDJSON streaming endpoint
        yield from stream_chat_message(message, thread_id=thread_id, provider=provider)


def get_all_threads() -> list:
    """Fetch all chat threads."""
    try:
        response = requests.get(f"{API_BASE_URL}/chat/threads", timeout=15)
        data = response.json()
        if data.get("status") == "success":
            return data.get("threads", [])
        return []
    except Exception as e:
        print(f"Error fetching threads: {e}")
        return []


def get_thread_messages(thread_id: str) -> dict:
    """Fetch all messages for a clear thread."""
    try:
        response = requests.get(f"{API_BASE_URL}/chat/threads/{thread_id}", timeout=30)
        data = response.json()
        if data.get("status") == "success":
            return data
        return {"messages": [], "summary": ""}
    except Exception as e:
        print(f"Error fetching thread details: {e}")
        return {"messages": [], "summary": ""}
