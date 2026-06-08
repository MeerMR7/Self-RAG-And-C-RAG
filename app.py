import streamlit as st
import os

# ── Inject secrets before any imports ────────────────────────────────────────
os.environ["GROQ_API_KEY"]   = st.secrets.get("GROQ_API_KEY",   os.getenv("GROQ_API_KEY", ""))
os.environ["TAVILY_API_KEY"] = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))

from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import run_pipeline
from ingest import ingest_documents

# ── Auto-ingest if chroma_db missing ─────────────────────────────────────────
if not os.path.exists("./chroma_db"):
    with st.spinner("⚙️ Building knowledge base from documents..."):
        try:
            ingest_documents()
            st.success("✅ Knowledge base ready!")
        except Exception as e:
            st.warning(f"⚠️ Auto-ingest failed: {e}")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Self-Correcting AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d0d0d; color: #ececec; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #1a1a1a;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border: 1px solid #2a2a2a;
}

/* Chat input fixed at bottom */
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    background-color: #0d0d0d;
    padding: 12px 0;
    border-top: 1px solid #222;
    z-index: 999;
}

/* Input box */
[data-testid="stChatInput"] textarea {
    background-color: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 15px !important;
}

/* Add bottom padding so messages don't hide behind input */
[data-testid="stVerticalBlock"] { padding-bottom: 100px; }

/* Reasoning step boxes */
.step-box {
    background: #161b2e;
    border-left: 3px solid #3b82f6;
    padding: 8px 14px;
    margin: 5px 0;
    border-radius: 6px;
    font-size: 0.83em;
    color: #93c5fd;
    font-family: 'Courier New', monospace;
}

/* Source boxes */
.source-box {
    background: #1a1a1a;
    border: 1px solid #2d2d2d;
    padding: 6px 12px;
    margin: 4px 0;
    border-radius: 6px;
    font-size: 0.82em;
    color: #86efac;
}

/* Expander styling */
[data-testid="stExpander"] {
    background-color: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background-color: #1d4ed8;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: #2563eb;
}

/* Fix white box at bottom */
.stChatFloatingInputContainer {
    background-color: #0d0d0d !important;
    border-top: 1px solid #222 !important;
}

/* Fix input field */
.stChatInputContainer {
    background-color: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
}

/* Force dark background everywhere */
.main, .block-container {
    background-color: #0d0d0d !important;
}

/* Success/Error */
.stSuccess { background-color: #052e16 !important; }
.stError { background-color: #2d0a0a !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Smart RAG AI")
    st.caption("Self-RAG · CRAG · LangGraph")
    st.divider()

    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        os.makedirs("./data", exist_ok=True)
        for f in uploaded_files:
            with open(f"./data/{f.name}", "wb") as out:
                out.write(f.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) saved!")

    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing documents..."):
            try:
                ingest_documents()
                st.success("✅ Knowledge base updated!")
            except Exception as e:
                st.error(f"❌ {e}")

    st.divider()
    st.markdown("""
    **Pipeline Info**
    - 🤖 Model: `llama-3.1-8b-instant`
    - 🗄️ Vector DB: `ChromaDB`
    - 🌐 Web Search: `Tavily`
    - 🔗 Framework: `LangGraph`
    """)
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Chat Area ────────────────────────────────────────────────────────────
st.markdown("## 🧠 Smart Self-Correcting AI")
st.caption("Powered by Self-RAG & CRAG — watch the AI think, verify, and correct itself")
st.divider()

# ── Init session state ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Welcome message ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #555;">
        <div style="font-size: 48px;">🧠</div>
        <div style="font-size: 20px; font-weight: 600; color: #888; margin-top: 12px;">Ask me anything</div>
        <div style="font-size: 14px; color: #444; margin-top: 8px;">I'll retrieve, verify, and self-correct my answers in real time</div>
    </div>
    """, unsafe_allow_html=True)

# ── Display chat history ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if message.get("workflow_steps"):
                with st.expander("🔄 Reasoning Steps", expanded=False):
                    for step in message["workflow_steps"]:
                        st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    seen = set()
                    for doc in message["sources"]:
                        src = doc.metadata.get("source", "Vector Store")
                        if src not in seen:
                            seen.add(src)
                            st.markdown(f'<div class="source-box">📄 {src}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                result = run_pipeline(prompt)

                with st.expander("🔄 Reasoning Steps", expanded=True):
                    for step in result["workflow_steps"]:
                        st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

                st.markdown(result["generation"])

                if result["documents"]:
                    with st.expander("📚 Sources", expanded=False):
                        seen = set()
                        for doc in result["documents"]:
                            src = doc.metadata.get("source", "Vector Store")
                            if src not in seen:
                                seen.add(src)
                                st.markdown(f'<div class="source-box">📄 {src}</div>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["generation"],
                    "workflow_steps": result["workflow_steps"],
                    "sources": result["documents"]
                })

            except Exception as e:
                err_msg = str(e)
                st.error(f"❌ Error: {err_msg}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {err_msg}",
                    "workflow_steps": [],
                    "sources": []
                })
