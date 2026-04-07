import streamlit as st
import time

st.markdown(
    """
<style>
    .chat-header {
        text-align: center;
        padding: 1rem 0 1rem;
    }
    .chat-header h1 {
        color: #8e44ad;
        font-size: 2rem;
    }
    .chat-header p {
        color: #7f8c8d;
        font-size: 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="chat-header">
    <h1>💬 AI Chat Assistant</h1>
    <p>Ask questions about floods, rainfall, or general hydrology in the Mahanadi Basin</p>
</div>
""",
    unsafe_allow_html=True,
)

import uuid
from utils.api_client import get_all_threads, get_thread_messages

# --- Sidebar: Chat History ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    provider_options = ["groq", "gemini", "ollama"]
    current_provider = st.session_state.get("llm_provider", "ollama")
    if current_provider not in provider_options:
        current_provider = "ollama"

    st.session_state.llm_provider = st.selectbox(
        "Model Provider",
        options=provider_options,
        format_func=lambda x: {
            "groq": "Groq (Fast, Cloud)",
            "gemini": "Gemini (Google GenAI)",
            "ollama": "Ollama (gemini-3-flash-preview)",
        }.get(x, x),
        index=provider_options.index(current_provider),
        help="Select the AI model provider. Groq is fast but rate limited. Gemini offers deep context. Ollama uses gemini-3-flash-preview.",
    )
    st.markdown("---")

    st.markdown("### 📚 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Load threads from backend (cached in session state to prevent slow UI reloading)
    if "threads_list" not in st.session_state:
        # Show a temporary spinner because this network request can be slow
        with st.spinner("Loading history..."):
            st.session_state.threads_list = get_all_threads()

    threads = st.session_state.threads_list

    if not threads:
        st.info("No past conversations found.")
    else:
        for t in threads:
            tid = t["thread_id"]
            title = t.get("title", f"Thread {tid[:8]}")

            # Truncate title so it fits on exactly one line (Streamlit sidebar width)
            if len(title) > 22:
                title = title[:22] + "..."

            # Highlight current thread
            is_active = st.session_state.get("thread_id") == tid
            btn_type = "primary" if is_active else "secondary"

            if st.button(
                title, key=f"btn_{tid}", use_container_width=True, type=btn_type
            ):
                # Load thread history
                history = get_thread_messages(tid)
                st.session_state.thread_id = tid
                st.session_state.messages = history.get("messages", [])
                st.rerun()

# --- Main Chat UI ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me about flood susceptibility in Cuttack..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare assistant response container
    with st.chat_message("assistant"):
        from utils.api_client import stream_chat_message_sse

        try:
            response_tokens = []
            completed_steps = []
            current_step = None

            # Placeholder for the step tracker — re-rendered as steps come in
            steps_placeholder = st.empty()

            _SPINNER_CSS = """<style>
@keyframes spin { to { transform: rotate(360deg); } }
.step-spinner {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid #ccc; border-top-color: #8e44ad;
    border-radius: 50%; animation: spin 0.6s linear infinite;
    vertical-align: middle; margin-right: 6px;
}
</style>"""

            def _render_steps():
                """Re-render all steps: completed (✅) + current (animated spinner)."""
                html = _SPINNER_CSS
                for s in completed_steps:
                    html += f'<p style="margin:4px 0">✅ &nbsp;{s}</p>'
                if current_step:
                    html += f'<p style="margin:4px 0"><span class="step-spinner"></span>{current_step}</p>'
                steps_placeholder.markdown(html, unsafe_allow_html=True)

            for item in stream_chat_message_sse(
                prompt,
                thread_id=st.session_state.thread_id,
                provider=st.session_state.get("llm_provider", "ollama"),
            ):
                if item["type"] == "status":
                    # Mark previous current step as completed
                    if current_step:
                        completed_steps.append(current_step)
                    current_step = item["content"]
                    _render_steps()

                elif item["type"] == "token":
                    response_tokens.append(item["content"])

                elif item["type"] == "error":
                    response_tokens.append(item["content"])

            # Mark the last step as completed
            if current_step:
                completed_steps.append(current_step)
                current_step = None

            # Final render with all steps marked completed
            _render_steps()

            # Render the full AI response below the steps
            if response_tokens:
                full_response = "".join(response_tokens)
                st.markdown("---")
                st.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

                # Fetch fresh chat history in the background to update the sidebar
                st.session_state.threads_list = get_all_threads()
            else:
                st.warning(
                    "Failed to generate response. Check if backend API is running."
                )
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
