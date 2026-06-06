import os

# ── Load secrets (Streamlit Cloud) or .env (local) ────────────────────────────
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key, val in st.secrets.items():
            os.environ[key] = str(val)
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
# ──────────────────────────────────────────────────────────────────────────────

from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

llm             = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
web_search_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=3)

# ── State ──────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:       str
    documents:      List[Document]
    generation:     str
    workflow_steps: List[str]
    web_searched:   bool

# ── Nodes ──────────────────────────────────────────────────────────────────────
def load_vectorstore():
    embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, collection_name="rag_documents")
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def retrieve(state):
    steps = state.get("workflow_steps", [])
    steps.append("📥 Retrieving relevant documents from vector store...")
    try:
        retriever = load_vectorstore()
        docs      = retriever.invoke(state["question"])
    except Exception:
        docs = []
        steps.append("⚠️ No documents found — will use web search.")
    return {**state, "documents": docs, "workflow_steps": steps}

def grade_documents(state):
    steps = state["workflow_steps"]
    steps.append("🔍 Evaluating document relevance (Self-RAG grader)...")
    if not state["documents"]:
        return {**state, "documents": [], "workflow_steps": steps}
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a relevance grader. Reply only 'yes' or 'no'."),
        ("human",  "Question: {question}\nDocument: {document}\nIs this relevant?")
    ])
    grader       = grade_prompt | llm
    relevant_docs = []
    for doc in state["documents"]:
        try:
            result = grader.invoke({"question": state["question"], "document": doc.page_content[:500]})
            if "yes" in result.content.lower():
                relevant_docs.append(doc)
        except Exception:
            relevant_docs.append(doc)
    steps.append(f"✅ {len(relevant_docs)}/{len(state['documents'])} documents passed relevance check.")
    return {**state, "documents": relevant_docs, "workflow_steps": steps}

def rewrite_query(state):
    steps = state["workflow_steps"]
    steps.append("✏️ Weak context — rewriting query for better retrieval...")
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite this question to be more specific. Return only the rewritten question."),
        ("human",  "Question: {question}")
    ])
    try:
        result = (rewrite_prompt | llm).invoke({"question": state["question"]})
        steps.append(f"✏️ Rewritten: '{result.content.strip()}'")
        return {**state, "question": result.content.strip(), "workflow_steps": steps}
    except Exception:
        return {**state, "workflow_steps": steps}

def web_search(state):
    steps = state["workflow_steps"]
    steps.append("🌐 Executing web search fallback via Tavily...")
    try:
        results  = web_search_tool.invoke({"query": state["question"]})
        web_docs = [
            Document(page_content=r["content"], metadata={"source": r["url"], "type": "web"})
            for r in results if "content" in r
        ]
        steps.append(f"🌐 Web search returned {len(web_docs)} results.")
        return {**state, "documents": state.get("documents", []) + web_docs,
                "workflow_steps": steps, "web_searched": True}
    except Exception as e:
        steps.append(f"❌ Web search failed: {e}")
        return {**state, "workflow_steps": steps, "web_searched": True}

def generate(state):
    steps   = state["workflow_steps"]
    steps.append("💬 Generating final answer from grounded context...")
    context = (
        "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(state["documents"])])
        if state["documents"] else "No context found."
    )
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the context. Be accurate and concise."),
        ("human",  "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    try:
        result = (generate_prompt | llm).invoke({"context": context, "question": state["question"]})
        return {**state, "generation": result.content, "workflow_steps": steps}
    except Exception as e:
        return {**state, "generation": f"Error: {e}", "workflow_steps": steps}

def check_hallucination(state):
    steps = state["workflow_steps"]
    steps.append("🛡️ Checking for hallucinations & evaluating answer quality...")
    if not state["documents"]:
        steps.append("⚠️ Skipped — no context available.")
        return {**state, "workflow_steps": steps}
    context = "\n\n".join([doc.page_content for doc in state["documents"][:3]])
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", "Is the answer grounded in the context? Reply only 'yes' or 'no'."),
        ("human",  "Context: {context}\n\nAnswer: {answer}")
    ])
    try:
        result = (hallucination_prompt | llm).invoke({
            "context": context[:1500],
            "answer":  state["generation"][:500]
        })
        if "yes" in result.content.lower():
            steps.append("✅ Answer is grounded. No hallucinations detected.")
        else:
            steps.append("⚠️ Potential hallucination detected.")
    except Exception as e:
        steps.append(f"⚠️ Check skipped: {e}")
    return {**state, "workflow_steps": steps}

# ── Graph ──────────────────────────────────────────────────────────────────────
def decide_after_grading(state):
    return "rewrite_query" if not state["documents"] else "generate"

def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve",           retrieve)
    graph.add_node("grade_documents",    grade_documents)
    graph.add_node("rewrite_query",      rewrite_query)
    graph.add_node("web_search",         web_search)
    graph.add_node("generate",           generate)
    graph.add_node("check_hallucination",check_hallucination)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"rewrite_query": "rewrite_query", "generate": "generate"}
    )
    graph.add_edge("rewrite_query",       "web_search")
    graph.add_edge("web_search",          "generate")
    graph.add_edge("generate",            "check_hallucination")
    graph.add_edge("check_hallucination", END)
    return graph.compile()

# ── Public entry point ─────────────────────────────────────────────────────────
def run_pipeline(question: str) -> dict:
    app    = build_graph()
    result = app.invoke(RAGState(
        question=question,
        documents=[],
        generation="",
        workflow_steps=[],
        web_searched=False
    ))
    return result
