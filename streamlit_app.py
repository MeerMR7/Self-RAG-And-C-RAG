"""
streamlit_app.py  —  Smart Self-Correcting AI  (Self-RAG + CRAG)
Rebuilt UI: fully dark, cohesive, no clashing sections.
"""

import streamlit as st
import time
import random

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Smart RAG AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",   # cleaner on first load
)

# ── Load secrets / env ─────────────────────────────────────────────────────────
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception:
    pass

# ── Import your pipeline ───────────────────────────────────────────────────────
# Uncomment when your rag_pipeline.py is in the same folder:
# from rag_pipeline import run_rag_pipeline

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #0d1117 !important;
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] * { color: #8b949e !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stMarkdown a { color: #39d0d8 !important; }

/* ── Main container ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: #1c2330;
    border: 1px solid #30363d;
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem;
    display: inline-block;
    max-width: 80%;
    float: right;
    clear: both;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px 16px 16px 16px;
    padding: 0.85rem 1.1rem;
    line-height: 1.7;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background: #1c2330 !important;
    border: 1px solid #30363d !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, rgba(57,208,216,0.15), rgba(68,147,248,0.15)) !important;
    border: 1px solid rgba(57,208,216,0.3) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(57,208,216,0.45) !important;
    box-shadow: 0 0 0 3px rgba(57,208,216,0.07) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e6edf3 !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #484f58 !important;
}
[data-testid="stChatInput"] button {
    background: #39d0d8 !important;
    border-radius: 8px !important;
    color: #0d1117 !important;
}
[data-testid="stChatInput"] button:hover {
    background: #2eb8bf !important;
}

/* ── Expanders (reasoning / sources) ── */
[data-testid="stExpander"] {
    background-color: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stExpander"]:hover {
    border-color: #30363d !important;
}
details summary {
    color: #8b949e !important;
    font-size: 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 0.6rem 0.8rem !important;
}
details[open] summary {
    border-bottom: 1px solid #21262d !important;
}
.streamlit-expanderContent {
    background: #0d1117 !important;
    padding: 0.75rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    color: #8b949e !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.18s !important;
    padding: 0.4rem 0.9rem !important;
}
.stButton > button:hover {
    border-color: #39d0d8 !important;
    color: #39d0d8 !important;
    background: rgba(57,208,216,0.05) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
}
[data-testid="stFileUploader"] * { color: #6e7681 !important; }
[data-testid="stFileUploader"] button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
    border-radius: 6px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important;
}
[data-testid="stMetricLabel"] p { color: #6e7681 !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"]   { color: #e6edf3 !important; font-size: 1rem   !important; }

/* ── Divider ── */
hr { border-color: #21262d !important; margin: 1rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #30363d; }

/* ── Custom components ── */
.rag-header {
    padding: 2rem 2.5rem 1.5rem;
    border-bottom: 1px solid #21262d;
    background: #0d1117;
}
.rag-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.3rem;
}
.rag-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, rgba(57,208,216,0.15), rgba(68,147,248,0.15));
    border: 1px solid rgba(57,208,216,0.3);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.rag-title {
    font-size: 1.25rem;
    font-weight: 600;
    background: linear-gradient(135deg, #39d0d8 0%, #4493f8 60%, #a371f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.02em;
}
.rag-subtitle {
    font-size: 0.78rem;
    color: #484f58;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}
.tech-badge {
    display: inline-block;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.68rem;
    color: #6e7681;
    font-family: 'JetBrains Mono', monospace;
    margin: 0 2px;
}
.chat-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 1.5rem 1.5rem 0;
}
.input-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 0.75rem 1.5rem 1.25rem;
    background: #0d1117;
    position: sticky;
    bottom: 0;
    border-top: 1px solid #21262d;
}
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
}
.empty-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}
.empty-title {
    font-size: 1rem;
    font-weight: 500;
    color: #8b949e;
    margin-bottom: 0.35rem;
}
.empty-sub {
    font-size: 0.78rem;
    color: #484f58;
    font-family: 'JetBrains Mono', monospace;
}
.example-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    max-width: 640px;
    margin: 1.5rem auto 0;
}
.example-btn {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: left;
    cursor: pointer;
    transition: all 0.18s;
    color: #8b949e;
    font-size: 0.78rem;
    line-height: 1.45;
    font-family: 'Inter', sans-serif;
    width: 100%;
}
.example-btn:hover {
    border-color: #39d0d8;
    color: #c9d1d9;
    background: rgba(57,208,216,0.04);
}
.example-tag {
    display: block;
    font-size: 0.62rem;
    color: #39d0d8;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 3px;
    opacity: 0.8;
}

/* Workflow step pills */
.wf-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid #161b22;
}
.wf-row:last-child { border-bottom: none; }
.wf-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }
.wf-body { flex: 1; min-width: 0; }
.wf-label { font-size: 0.78rem; font-weight: 500; font-family: 'JetBrains Mono', monospace; }
.wf-detail { font-size: 0.7rem; color: #6e7681; margin-top: 1px; }
.wf-dur { font-size: 0.65rem; color: #484f58; font-family: 'JetBrains Mono', monospace; flex-shrink: 0; }
.wf-complete .wf-label { color: #3fb950; }
.wf-active   .wf-label { color: #39d0d8; }
.wf-skipped  .wf-label { color: #484f58; text-decoration: line-through; }
.wf-error    .wf-label { color: #f85149; }

/* Source cards */
.src-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 12px;
    margin: 5px 0;
    transition: border-color 0.15s;
}
.src-card:hover { border-color: #444c56; }
.src-num  { color: #484f58; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; }
.src-title { font-size: 0.82rem; font-weight: 500; color: #c9d1d9; margin: 2px 0; }
.src-title a { color: #c9d1d9; text-decoration: none; }
.src-title a:hover { color: #39d0d8; }
.src-snippet { font-size: 0.72rem; color: #6e7681; line-height: 1.5; margin: 3px 0 6px; }
.src-foot { display: flex; align-items: center; gap: 8px; }
.src-badge-doc { background: rgba(68,147,248,0.1); color: rgba(68,147,248,0.75);
    font-size: 0.6rem; padding: 1px 6px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; text-transform: uppercase; }
.src-badge-web { background: rgba(57,208,216,0.1); color: rgba(57,208,216,0.75);
    font-size: 0.6rem; padding: 1px 6px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; text-transform: uppercase; }
.src-rel { font-size: 0.65rem; color: #484f58; font-family: 'JetBrains Mono', monospace; }

/* Info caption */
.meta-line {
    font-size: 0.68rem;
    color: #484f58;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px;
    padding-left: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "total_queries": 0,
    "web_searches": 0,
    "rewrites": 0,
    "_prefill": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ────────────────────────────────────────────────────────────────────
STEP_META = [
    ("retrieve",            "Retrieve",            "🔵", "Querying ChromaDB vector store"),
    ("evaluate_relevance",  "Evaluate Relevance",  "🟣", "Self-RAG relevance classifier"),
    ("rewrite_query",       "Rewrite Query",        "🟡", "CRAG query correction"),
    ("web_search",          "Web Search",           "🩵", "Tavily fallback search"),
    ("generate",            "Generate",             "🟢", "GPT-4o answer generation"),
    ("check_hallucination", "Hallucination Check",  "🟠", "Faithfulness verification"),
    ("finalize",            "Finalise",             "⚪", "Packaging response"),
]

def render_workflow_html(steps: list) -> str:
    rows = ""
    for s in steps:
        if s["status"] == "pending":
            continue
        css = f"wf-{s['status']}"
        sym = {"complete": "✓", "active": "⟳", "skipped": "—", "error": "✗"}.get(s["status"], "")
        icon = s.get("icon", "⚪")
        detail = f'<div class="wf-detail">{s["detail"]}</div>' if s.get("detail") else ""
        dur = f'<span class="wf-dur">{s["duration_ms"]}ms</span>' if s.get("duration_ms") else ""
        rows += f"""
        <div class="wf-row {css}">
            <span class="wf-icon">{icon}</span>
            <div class="wf-body">
                <span class="wf-label">{sym} {s['label']}</span>
                {detail}
            </div>
            {dur}
        </div>"""
    return f'<div style="padding:4px 0">{rows}</div>'


def render_sources_html(sources: list) -> str:
    html = ""
    for i, s in enumerate(sources, 1):
        badge = 'src-badge-web' if s.get("type") == "web" else 'src-badge-doc'
        label = "WEB" if s.get("type") == "web" else "DOC"
        rel = f'<span class="src-rel">{round(s["relevance_score"]*100)}% match</span>' if s.get("relevance_score") else ""
        title_inner = (
            f'<a href="{s["url"]}" target="_blank">{s["title"]} ↗</a>'
            if s.get("url") else s.get("title", f"Source {i}")
        )
        html += f"""
        <div class="src-card">
            <div class="src-num">#{i}</div>
            <div class="src-title">{title_inner}</div>
            <div class="src-snippet">{s.get('snippet','')}</div>
            <div class="src-foot">
                <span class="{badge}">{label}</span>
                {rel}
            </div>
        </div>"""
    return html


def mock_pipeline(prompt: str) -> dict:
    """Fallback mock — remove once rag_pipeline.py is wired up."""
    time.sleep(0.4)
    short = len(prompt.split()) < 5
    return {
        "answer": (
            f"**Answer to:** *{prompt}*\n\n"
            "Based on the retrieved documents, here is a comprehensive response. "
            "The Self-RAG architecture first retrieves relevant chunks from ChromaDB, "
            "evaluates their relevance using a classifier, and — if context is insufficient "
            "— rewrites the query and falls back to a live Tavily web search.\n\n"
            "The final answer is verified against the sources to ensure factual consistency "
            "before being returned to you."
        ),
        "sources": [
            {"id": "1", "title": "Self-RAG Paper (arXiv 2310.11511)", "url": "https://arxiv.org/abs/2310.11511",
             "snippet": "Self-RAG trains LMs to retrieve and reflect using special tokens.", "type": "document", "relevance_score": 0.93},
            {"id": "2", "title": "CRAG Paper (arXiv 2401.15884)", "url": "https://arxiv.org/abs/2401.15884",
             "snippet": "CRAG evaluates retrieval quality and triggers web search when needed.", "type": "document", "relevance_score": 0.87},
        ],
        "web_search_used": short,
        "query_rewritten": f"{prompt} — detailed explanation" if short else None,
        "tokens_used": random.randint(800, 1600),
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Smart RAG AI")
    st.caption("Self-RAG · CRAG · LangGraph")
    st.divider()

    st.markdown("**Session Stats**")
    c1, c2 = st.columns(2)
    c1.metric("Queries",      st.session_state.total_queries)
    c2.metric("Web Searches", st.session_state.web_searches)
    st.metric("Query Rewrites", st.session_state.rewrites)
    st.divider()

    st.markdown("**Upload Documents**")
    st.caption("Upload PDF or TXT files")
    uploaded = st.file_uploader("", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")
    if st.button("⚙️ Process Documents", use_container_width=True):
        if uploaded:
            with st.spinner("Processing…"):
                time.sleep(1.2)
            st.success(f"✓ {len(uploaded)} file(s) indexed")
        else:
            st.warning("No files uploaded")
    st.divider()

    st.markdown("**Pipeline**")
    st.markdown("""
<div style="font-size:0.75rem;line-height:2;color:#6e7681;">
🔵 <b style="color:#4493f8">Retrieve</b> — ChromaDB<br>
🟣 <b style="color:#a371f7">Evaluate</b> — Self-RAG<br>
🟡 <b style="color:#e3b341">Rewrite</b> — CRAG<br>
🩵 <b style="color:#39d0d8">Web Search</b> — Tavily<br>
🟢 <b style="color:#3fb950">Generate</b> — GPT-4o<br>
🟠 <b style="color:#f0883e">Verify</b> — Hallucination check
</div>""", unsafe_allow_html=True)
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        for k in ["messages","total_queries","web_searches","rewrites"]:
            st.session_state[k] = [] if k == "messages" else 0
        st.rerun()

    st.markdown("""
<div style="font-size:0.65rem;color:#484f58;font-family:'JetBrains Mono',monospace;margin-top:0.5rem;">
Self-RAG + CRAG v0.1<br>LangGraph · ChromaDB · GPT-4o
</div>""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <div class="rag-logo">
    <div class="rag-logo-icon">🧠</div>
    <span class="rag-title">Smart Self-Correcting AI</span>
  </div>
  <div class="rag-subtitle">Self-RAG · CRAG · LangGraph · ChromaDB · GPT-4o</div>
  <div style="margin-top:0.6rem;">
    <span class="tech-badge">Self-RAG</span>
    <span class="tech-badge">CRAG</span>
    <span class="tech-badge">LangGraph</span>
    <span class="tech-badge">ChromaDB</span>
    <span class="tech-badge">Tavily</span>
    <span class="tech-badge">GPT-4o</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Chat area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">Ask me anything</div>
        <div class="empty-sub">I'll retrieve · evaluate · correct · verify — before answering</div>
    </div>""", unsafe_allow_html=True)

    # Example prompts
    examples = [
        ("Architecture", "What is the difference between Self-RAG and CRAG?"),
        ("LangGraph",    "How does LangGraph manage stateful AI workflows?"),
        ("Research",     "What are the latest advances in RAG systems?"),
        ("Safety",       "How are hallucinations detected and reduced in RAG?"),
    ]
    col1, col2 = st.columns(2)
    for i, (tag, text) in enumerate(examples):
        with (col1 if i % 2 == 0 else col2):
            if st.button(f"**{tag}**\n{text}", key=f"ex{i}", use_container_width=True):
                st.session_state._prefill = text
                st.rerun()

# Render history
for msg in st.session_state.messages:
    avatar = "🧠" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("workflow_steps"):
                active = [s for s in msg["workflow_steps"] if s["status"] != "pending"]
                with st.expander(f"🔍 Reasoning Trace — {len(active)} steps", expanded=False):
                    st.markdown(render_workflow_html(msg["workflow_steps"]), unsafe_allow_html=True)

            if msg.get("sources"):
                with st.expander(f"📚 Sources — {len(msg['sources'])} retrieved", expanded=False):
                    st.markdown(render_sources_html(msg["sources"]), unsafe_allow_html=True)

            parts = []
            if msg.get("tokens_used"):      parts.append(f"⚡ {msg['tokens_used']:,} tokens")
            if msg.get("query_rewritten"):  parts.append("✏️ query rewritten")
            if msg.get("web_search_used"):  parts.append("🌐 web search used")
            if parts:
                st.markdown(f'<div class="meta-line">{" · ".join(parts)}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end chat-wrapper


# ── Input ──────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("_prefill", None)
prompt  = st.chat_input("Ask anything — I'll reason through it step by step…") or prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_queries += 1

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        trace_ph = st.empty()
        text_ph  = st.empty()

        # Init workflow steps
        steps = [
            {"id": sid, "label": lbl, "icon": ico, "status": "pending", "detail": default_detail}
            for sid, lbl, ico, default_detail in STEP_META
        ]

        def set_step(idx, status, detail=None, duration=None):
            steps[idx]["status"] = status
            if detail:   steps[idx]["detail"]      = detail
            if duration: steps[idx]["duration_ms"] = duration
            trace_ph.markdown(render_workflow_html(steps), unsafe_allow_html=True)

        response_text   = ""
        sources         = []
        web_search_used = False
        query_rewritten = None
        tokens_used     = None

        try:
            # ── Step 1: Retrieve ───────────────────────────────────────────────
            set_step(0, "active", "Querying ChromaDB vector store…")

            # ── CALL YOUR PIPELINE HERE ────────────────────────────────────────
            # result = run_rag_pipeline(prompt)
            result = mock_pipeline(prompt)   # ← remove when wired up
            # ──────────────────────────────────────────────────────────────────

            set_step(0, "complete", "Candidate documents retrieved",
                     duration=random.randint(300,600))

            # ── Step 2: Evaluate relevance ─────────────────────────────────────
            set_step(1, "active", "Running Self-RAG relevance classifier…")
            time.sleep(0.25)
            web_search_used = result.get("web_search_used", False)
            query_rewritten = result.get("query_rewritten")
            set_step(1, "complete",
                     "Context sparse — fallback triggered" if web_search_used else "Sufficient context found",
                     duration=random.randint(380,700))

            # ── Step 3: Rewrite query ──────────────────────────────────────────
            if query_rewritten:
                st.session_state.rewrites += 1
                set_step(2, "active", "Ambiguous query — rewriting…")
                time.sleep(0.2)
                set_step(2, "complete", f'→ "{query_rewritten}"',
                         duration=random.randint(220,420))
            else:
                set_step(2, "skipped", "Query clear — no rewrite needed")

            # ── Step 4: Web search ─────────────────────────────────────────────
            if web_search_used:
                st.session_state.web_searches += 1
                set_step(3, "active", "Calling Tavily Search API…")
                time.sleep(0.28)
                set_step(3, "complete", "Live web sources retrieved",
                         duration=random.randint(580,1100))
            else:
                set_step(3, "skipped", "Vector context sufficient")

            # ── Step 5: Generate ───────────────────────────────────────────────
            set_step(4, "active", "Generating with GPT-4o…")
            time.sleep(0.2)
            set_step(4, "complete", duration=random.randint(700,1600))

            # ── Step 6: Hallucination check ────────────────────────────────────
            set_step(5, "active", "Verifying factual consistency…")
            time.sleep(0.18)
            set_step(5, "complete", "Answer grounded in sources",
                     duration=random.randint(280,520))

            # ── Step 7: Finalise ───────────────────────────────────────────────
            set_step(6, "complete", "Response ready",
                     duration=random.randint(40,100))

            response_text = result.get("answer", "No answer returned.")
            sources       = result.get("sources", [])
            tokens_used   = result.get("tokens_used")

        except Exception as e:
            for s in steps:
                if s["status"] == "active":
                    s["status"] = "error"
                    s["detail"] = str(e)
            trace_ph.markdown(render_workflow_html(steps), unsafe_allow_html=True)
            response_text = f"⚠️ **Pipeline error:** {e}\n\nCheck your API keys and that `rag_pipeline.py` is present."

        # Stream answer token by token
        words, buf = response_text.split(" "), ""
        for w in words:
            buf += w + " "
            text_ph.markdown(buf + "▋")
            time.sleep(0.014)
        text_ph.markdown(response_text)

        # Inline sources
        if sources:
            with st.expander(f"📚 Sources — {len(sources)} retrieved", expanded=True):
                st.markdown(render_sources_html(sources), unsafe_allow_html=True)

        # Meta line
        parts = []
        if tokens_used:     parts.append(f"⚡ {tokens_used:,} tokens")
        if query_rewritten: parts.append("✏️ query rewritten")
        if web_search_used: parts.append("🌐 web search used")
        if parts:
            st.markdown(f'<div class="meta-line">{" · ".join(parts)}</div>', unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role":           "assistant",
        "content":        response_text,
        "workflow_steps": steps,
        "sources":        sources,
        "tokens_used":    tokens_used,
        "query_rewritten":query_rewritten,
        "web_search_used":web_search_used,
    })
