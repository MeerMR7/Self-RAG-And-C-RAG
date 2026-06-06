import streamlit as st
import os
from dotenv import load_dotenv

# CRITICAL FIX: Load .env BEFORE importing modules that read env vars at import time
load_dotenv()

from rag_pipeline import run_pipeline
from ingest import ingest_documents

st.set_page_config(page_title="Smart Self-Correcting AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0f0f0f; color: #f0f0f0; }
.step-box { background: #1a1a2e; border-left: 3px solid #4a9eff; padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 0.85em; color: #b0c4de; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 Smart RAG AI")
    st.caption("Self-RAG · CRAG · LangGraph")
    st.divider()

    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        os.makedirs("./documents", exist_ok=True)
        for f in uploaded_files:
            with open(f"./documents/{f.name}", "wb") as out:
                out.write(f.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")

    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                ingest_documents()
                st.success("✅ Documents stored in ChromaDB!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.divider()
    st.markdown("**Model:** Llama3-8b (Groq)  \n**Vector DB:** ChromaDB  \n**Web Search:** Tavily  \n**Framework:** LangGraph")
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🧠 Smart Self-Correcting AI")
st.caption("Powered by Self-RAG & CRAG — watch the AI think, verify, and correct itself")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
                            st.markdown(f"📄 `{src}`")
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Running Self-RAG + CRAG Pipeline..."):
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
                                st.markdown(f"📄 `{src}`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["generation"],
                    "workflow_steps": result["workflow_steps"],
                    "sources": result["documents"]
                })
            except Exception as e:
                st.error(f"❌ Error: {e}")
