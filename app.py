"""
Streamlit chat app with feedback for the RAG pipeline.

Run with:
    uv run streamlit run app.py

Features:
- Course selector
- Question input → RAG answer via local Ollama
- Response time, relevance score, token usage display
- Thumbs up / thumbs down feedback buttons
- Conversation history stored in session state
"""

import uuid
from datetime import datetime

import streamlit as st

from src import FaqHttpLoader, MinsearchIndex, OllamaClient, RAGBase
from src.monitoring import get_answer
from src.rag import INSTRUCTIONS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "granite4.1:3b"

COURSES = [
    "data-engineering-zoomcamp",
    "machine-learning-zoomcamp",
    "mlops-zoomcamp",
    "llm-zoomcamp",
]

# INSTRUCTIONS imported from src.rag — includes "I don't know" fallback

# ---------------------------------------------------------------------------
# Cached pipeline initialisation (runs once per session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading FAQ index…")
def load_pipeline() -> tuple[RAGBase, OllamaClient]:
    """Load documents, build the search index, and initialise the pipeline.

    Cached by Streamlit so the index is built only once per server process.
    """
    loader = FaqHttpLoader()
    documents = loader.load()
    index = MinsearchIndex(documents)
    llm_client = OllamaClient(num_ctx=16384)
    rag = RAGBase(
        index=index,
        llm_client=llm_client,
        llm_model=OLLAMA_MODEL,
        instructions=INSTRUCTIONS,
    )
    return rag, llm_client


# ---------------------------------------------------------------------------
# In-memory storage helpers (replace with PostgreSQL in the next lesson)
# ---------------------------------------------------------------------------


def save_conversation(
    conversation_id: str,
    question: str,
    answer_data: dict,
    course: str,
) -> None:
    """Append a conversation record to session-state history."""
    st.session_state.setdefault("history", []).append(
        {
            "id": conversation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "course": course,
            "question": question,
            **answer_data,
            "feedback": None,
        }
    )


def save_feedback(conversation_id: str, feedback: int) -> None:
    """Update the feedback field for the most recent conversation record."""
    for record in reversed(st.session_state.get("history", [])):
        if record["id"] == conversation_id:
            record["feedback"] = feedback
            break


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Course Assistant", page_icon="🎓", layout="centered")
st.title("🎓 Course Assistant")
st.caption(f"Powered by Ollama · model: `{OLLAMA_MODEL}`")

rag, llm_client = load_pipeline()

# Session state initialisation
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# --- Input widgets ---
course = st.selectbox("Select a course:", COURSES)
user_input = st.text_input(
    "Enter your question:", placeholder="e.g. How do I set up Docker?"
)

if st.button("Ask", type="primary", disabled=not user_input.strip()):
    with st.spinner("Thinking…"):
        answer_data = get_answer(
            rag=rag,
            llm_client=llm_client,
            question=user_input,
            course=course,
            model=OLLAMA_MODEL,
        )

    st.session_state.last_answer = answer_data

    st.success(answer_data["answer"])

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Response time", f"{answer_data['response_time']:.2f}s")
    col2.metric("Relevance", answer_data["relevance"])
    col3.metric("Tokens", answer_data["total_tokens"])

    with st.expander("Details"):
        st.write(f"**Relevance explanation:** {answer_data['relevance_explanation']}")
        st.write(
            f"**Tokens:** {answer_data['prompt_tokens']} prompt + "
            f"{answer_data['completion_tokens']} completion"
        )
        st.write(f"**Model:** {answer_data['model_used']}")

    save_conversation(st.session_state.conversation_id, user_input, answer_data, course)
    # Rotate conversation ID so the next question gets a fresh one
    st.session_state.conversation_id = str(uuid.uuid4())

# --- Feedback buttons (always visible after at least one answer) ---
if st.session_state.get("history"):
    last_id = st.session_state.history[-1]["id"]
    st.divider()
    st.write("Was this answer helpful?")
    fb_col1, fb_col2, _ = st.columns([1, 1, 6])
    if fb_col1.button("👍", key="thumbs_up"):
        save_feedback(last_id, 1)
        st.toast("Thanks for the positive feedback!", icon="👍")
    if fb_col2.button("👎", key="thumbs_down"):
        save_feedback(last_id, -1)
        st.toast("Thanks — we'll use this to improve.", icon="👎")

# --- Conversation history (collapsible) ---
if st.session_state.get("history"):
    with st.expander(f"Conversation history ({len(st.session_state.history)} entries)"):
        for rec in reversed(st.session_state.history):
            fb_icon = {1: "👍", -1: "👎", None: "—"}[rec.get("feedback")]
            st.markdown(
                f"**Q:** {rec['question']}  \n"
                f"**A:** {rec['answer']}  \n"
                f"*{rec['course']} · {rec['relevance']} · "
                f"{rec['response_time']:.1f}s · {fb_icon}*"
            )
            st.divider()
