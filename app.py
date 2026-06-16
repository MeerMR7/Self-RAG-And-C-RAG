import os
import traceback
import concurrent.futures

import streamlit as st
from dotenv import load_dotenv

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Self Correcting AI (Self-RAG and C-RAG)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- ENV / SECRETS ----------------
load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# ---------------- SAFE IMPORTS ----------------
try:
    from rag_pipeline import run_pipeline
    PIPELINE_IMPORT_ERROR = None
except Exception:
    run_pipeline = None
    PIPELINE_IMPORT_ERROR = traceback.format_exc()

try:
    from ingest import ingest_documents
    INGEST_IMPORT_ERROR = None
except Exception:
    ingest_documents = None
    INGEST_IMPORT_ERROR = traceback.format_exc()


# ---------------- HELPERS ----------------
@st.cache_resource
def get_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=2)


def run_pipeline_with_timeout(question: str, timeout_seconds: int = 90):
    """
    Prevents infinite 'Thinking...' state.
    If pipeline takes too long, app shows a timeout message.
    """
    executor = get_executor()
    future = executor.submit(run_pipeline, question)
    return future.result(timeout=timeout_seconds)


def normalize_result(result):
    """
    Accepts different pipeline result formats safely.
    """
    if not isinstance(result, dict):
        return {
            "generation": str(result),
            "documents": [],
            "workflow_steps": [],
        }

    return {
        "generation": result.get("generation")
        or result.get("answer")
        or result.get("response")
        or "No answer generated.",
        "documents": result.get("documents") or result.get("sources") or [],
        "workflow_steps": result.get("workflow_steps") or [],
    }


def show_sources(documents):
    seen = set()

    for doc in documents:
        try:
            source = doc.metadata.get("source", "Vector Store")
        except Exception:
            source = str(doc)

        if source not in seen:
            seen.add(source)
            st.markdown(f"- {source}")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🤖 Smart Self Correcting AI")
    st.caption("Self-RAG and C-RAG")
    st.divider()

    st.subheader("System Status")

    if os.getenv("GOOGLE_API_KEY"):
        st.success("GOOGLE_API_KEY loaded")
    else:
        st.error("GOOGLE_API_KEY missing")

    if os.getenv("TAVILY_API_KEY"):
        st.success("TAVILY_API_KEY loaded")
    else:
        st.warning("TAVILY_API_KEY missing — web search fallback disabled")

    if run_pipeline is None:
        st.error("Pipeline import failed")
    else:
        st.success("Pipeline loaded")

    if ingest_documents is None:
        st.warning("Ingest file not loaded")

    st.divider()

    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        if ingest_documents is None:
            st.error("ingest.py could not be imported.")
            if INGEST_IMPORT_ERROR:
                st.code(INGEST_IMPORT_ERROR)
        else:
            with st.spinner("Processing documents..."):
                try:
                    ingest_documents()
                    st.success("Knowledge base updated successfully.")
                except Exception:
                    st.error("Document processing failed.")
                    st.code(traceback.format_exc())

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown(
        """
        **Project Info**
        - Model: Gemini
        - Method: Self-RAG and C-RAG
        - Vector DB: ChromaDB
        - Web Search: Tavily
        - Framework: Streamlit
        """
    )


# ---------------- MAIN UI ----------------
st.title("Smart Self Correcting AI (Self-RAG and C-RAG)")
st.caption(
    "An intelligent chatbot using Self-RAG and C-RAG to retrieve, verify, correct, and generate accurate answers."
)
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

if PIPELINE_IMPORT_ERROR:
    st.error("Pipeline import error:")
    st.code(PIPELINE_IMPORT_ERROR)


# ---------------- DISPLAY OLD CHAT ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("workflow_steps"):
            with st.expander("Reasoning Steps", expanded=False):
                for step in message["workflow_steps"]:
                    st.markdown(f"- {step}")

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources", expanded=False):
                show_sources(message["sources"])


# ---------------- CHAT INPUT ----------------
prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if run_pipeline is None:
            answer = "Pipeline is not loaded. Please check the import error shown above."
            st.error(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "workflow_steps": [],
                    "sources": [],
                }
            )

        elif not os.getenv("GOOGLE_API_KEY"):
            answer = "GOOGLE_API_KEY is missing. Add it in Streamlit Secrets."
            st.error(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "workflow_steps": [],
                    "sources": [],
                }
            )

        else:
            with st.spinner("Thinking..."):
                try:
                    raw_result = run_pipeline_with_timeout(prompt, timeout_seconds=90)
                    result = normalize_result(raw_result)

                    answer = result["generation"]
                    steps = result["workflow_steps"]
                    documents = result["documents"]

                    st.markdown(answer)

                    if steps:
                        with st.expander("Reasoning Steps", expanded=False):
                            for step in steps:
                                st.markdown(f"- {step}")

                    if documents:
                        with st.expander("Sources", expanded=False):
                            show_sources(documents)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "workflow_steps": steps,
                            "sources": documents,
                        }
                    )

                except concurrent.futures.TimeoutError:
                    answer = (
                        "The chatbot took too long to respond. "
                        "This usually happens when the model, embeddings, vector database, or web search is slow. "
                        "Please try again with a shorter question."
                    )

                    st.error(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "workflow_steps": [],
                            "sources": [],
                        }
                    )

                except Exception:
                    full_error = traceback.format_exc()

                    st.error("Pipeline error occurred.")
                    st.code(full_error)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Pipeline error occurred. Check the error shown above.",
                            "workflow_steps": [],
                            "sources": [],
                        }
                    )
