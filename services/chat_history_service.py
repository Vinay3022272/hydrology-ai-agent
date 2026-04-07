import sys
import os
import psycopg
import logging

# Ensure the hydro_ai package is in the path
HYDRO_AI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "hydro_ai"))
if HYDRO_AI_DIR not in sys.path:
    sys.path.insert(0, HYDRO_AI_DIR)

from agent.config import POSTGRES_URI
from agent.agent_builder import build_agent

# Set up logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_threads_history():
    """Returns a list of dicts: [{'thread_id': '...', 'title': '...'}, ...]"""
    threads = []
    
    try:
        logger.info(f"Connecting to Postgres to fetch threads: {POSTGRES_URI}")
        
        # 1. Get unique thread IDs recently updated
        with psycopg.connect(POSTGRES_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT thread_id, max(checkpoint_id) as last_updated "
                    "FROM checkpoints "
                    "GROUP BY thread_id "
                    "ORDER BY last_updated DESC "
                    "LIMIT 20"
                )
                rows = cur.fetchall()
                thread_ids = [row[0] for row in rows]
                
        logger.info(f"Found {len(thread_ids)} threads in DB: {thread_ids}")
        
        if not thread_ids:
            return {"status": "success", "threads": []}

        # 2. Build agent once to fetch summaries/state
        agent = build_agent()
        
        for tid in thread_ids:
            config = {"configurable": {"thread_id": tid}}
            snapshot = agent.get_state(config)
            
            if snapshot and snapshot.values:
                state = snapshot.values
                messages = state.get("messages", [])
                
                title = state.get("summary", "")
                if not title:
                    # Find first human message
                    for msg in messages:
                        if getattr(msg, "type", "") == "human":
                            content = getattr(msg, "content", "Empty convo")
                            title = str(content)[:40].strip() + ("..." if len(str(content)) > 40 else "")
                            break
                            
                if not title:
                    title = f"Thread {tid[:8]}"
                    
                threads.append({
                    "thread_id": tid,
                    "title": title
                })
                
        logger.info(f"Returning {len(threads)} threads to frontend.")
        return {"status": "success", "threads": threads}
        
    except Exception as e:
        logger.error(f"Critical error in get_all_threads_history: {e}")
        return {"status": "error", "message": str(e), "threads": []}

def get_thread_messages(thread_id: str):
    """Returns the full messages history for a thread."""
    try:
        agent = build_agent()
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = agent.get_state(config)
        
        if not snapshot or not snapshot.values:
            return {"status": "error", "message": "Thread not found", "messages": []}
            
        messages = snapshot.values.get("messages", [])
        
        # Serialize messages for the frontend
        serialized_msgs = []
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            content = getattr(msg, "content", "")
            
            role = "assistant" if msg_type in ("ai", "AIMessageChunk") else "user" if msg_type == "human" else msg_type
            
            # Avoid sending tool calls raw
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                continue
                
            if role in ("user", "assistant") and content:
                serialized_msgs.append({"role": role, "content": content})
                
        return {"status": "success", "messages": serialized_msgs, "summary": snapshot.values.get("summary", "")}
        
    except Exception as e:
        logger.error(f"Error in get_thread_messages: {e}")
        return {"status": "error", "message": str(e), "messages": []}
