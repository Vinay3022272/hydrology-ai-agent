from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.flood_service import get_flood_prediction
from services.rainfall_service import get_rainfall_prediction
from services.chat_service import stream_chat_message
from fastapi.responses import StreamingResponse
from typing import Generator

app = FastAPI(title="Hydro AI API", version="1.0.0")

# Allow Streamlit to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Request Models


class FloodRequest(BaseModel):
    lat: float
    lon: float


class RainfallRequest(BaseModel):
    lat: float = None
    lon: float = None
    date: str = None
    mode: str = "point"


from typing import Optional


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    provider: Optional[str] = "groq"


#  Routes
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/predict/flood")
def predict_flood(request: FloodRequest):
    """Run flood susceptibility prediction for a lat/lon point."""
    result = get_flood_prediction(request.lat, request.lon)
    return result


@app.post("/predict/rainfall")
def predict_rainfall(request: RainfallRequest):
    """Placeholder for rainfall prediction."""
    result = get_rainfall_prediction(request.dict())
    return result


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream a chat message via LangChain agent or fallback routing."""
    return StreamingResponse(
        stream_chat_message(request.message, request.thread_id, request.provider),
        media_type="application/x-ndjson",
    )


def _to_sse(jsonl_stream: Generator[str, None, None]) -> Generator[str, None, None]:
    """Convert newline-delimited JSON chunks to SSE data events."""
    for chunk in jsonl_stream:
        payload = chunk.strip()
        if payload:
            yield f"data: {payload}\n\n"


@app.post("/chat/sse")
def chat_sse(request: ChatRequest):
    """SSE stream for chat responses (event-stream)."""
    base_stream = stream_chat_message(
        request.message, request.thread_id, request.provider
    )
    return StreamingResponse(
        _to_sse(base_stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


from services.chat_history_service import get_all_threads_history, get_thread_messages


@app.get("/chat/threads")
def get_threads():
    """Get all chat threads with summaries."""
    return get_all_threads_history()


@app.get("/chat/threads/{thread_id}")
def get_thread(thread_id: str):
    """Get all messages for a given thread."""
    return get_thread_messages(thread_id)
