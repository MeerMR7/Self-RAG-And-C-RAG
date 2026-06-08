import os
from dotenv import load_dotenv
from typing import List, TypedDict
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Lazy initialisation (never crashes on import) ────────────────────────────
@lru_cache(maxsize=1)
def _get_llm():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets (or your .env file for local runs)."
        )
    return ChatGroq(model="llama-3.1-8b-instant", api_key=key)

@lru_cache(maxsize=1)
def _get_web_search_tool():
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. "
            "Add it to Streamlit Secrets (or your .env file for local runs)."
        )
    return TavilySearchResults(api_key=key)

@lru_cache(maxsize=1)
def _get_retriever():
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ── State ────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    workflow_steps: List[str]
    tries: int

# ── Prompts / Chains (use lazy getters) ──────────────────────────────────────
def _retrieval_grader():
    return (
        ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing relevance of a retrieved document to a user question. If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. Give a binary score 'yes' or 'no'."),
            ("human", "Document:\n{document}\n\nQuestion:\n{question}")
        ])
        | _get_llm()
        | StrOutputParser()
    )

def _hallucination_grader():
    return (
        ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no'."),
            ("human", "Facts:\n{documents}\n\nAnswer:\n{generation}")
        ])
        | _get_llm()
        | StrOutputParser()
    )

def _answer_grader():
    return (
        ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no'."),
            ("human", "Question:\n{question}\n\nAnswer:\n{generation}")
        ])
        | _get_llm()
        | StrOutputParser()
    )

def _rag_chain():
    return (
        ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the retrieved context to answer the question. If you don't know, say you don't know. Keep it concise (3 sentences max)."),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:")
        ])
        | _get_llm()
        | StrOutputParser()
    )

# ── Nodes ────────────────────────────────────────────────────────────────────
def retrieve(state: GraphState):
    docs = _get_retriever().invoke(state["question"])
    state["workflow_steps"].append(f"🔍 Retrieved {len(docs)} documents from vector store.")
    return {"documents": docs, "question": state["question"], "workflow_steps": state["workflow_steps"], "tries": state.get("tries", 0)}

def grade_documents(state: GraphState):
    docs = state["documents"]
    q = state["question"]
    steps = state["workflow_steps"]
    filtered = []
    grader = _retrieval_grader()
    for d in docs:
        score = grader.invoke({"document": d.page_content, "question": q})
        if "yes" in score.lower():
            filtered.append(d)
    steps.append(f"📊 Graded documents: {len(filtered)}/{len(docs)} relevant.")
    return {"documents": filtered, "question": q, "workflow_steps": steps, "tries": state.get("tries", 0)}

def generate(state: GraphState):
    q = state["question"]
    docs = state["documents"]
    steps = state["workflow_steps"]
    tries = state.get("tries", 0) + 1
    context = "\n\n".join([d.page_content for d in docs]) if docs else "No relevant context found."
    generation = _rag_chain().invoke({"question": q, "context": context})
    steps.append(f"✍️ Generated answer (attempt {tries}).")
    return {"documents": docs, "question": q, "generation": generation, "workflow_steps": steps, "tries": tries}

def web_search(state: GraphState):
    q = state["question"]
    steps = state["workflow_steps"]
    results = _get_web_search_tool().invoke({"query": q})
    web_docs = []
    for r in results:
        if isinstance(r, dict):
            content = r.get("content", r.get("snippet", ""))
            url = r.get("url", "Web Search")
            web_docs.append(Document(page_content=content, metadata={"source": url}))
    steps.append(f"🌐 Web search returned {len(web_docs)} results.")
    return {"documents": web_docs, "question": q, "workflow_steps": steps, "tries": state.get("tries", 0)}

def grade_generation(state: GraphState):
    q = state["question"]
    docs = state["documents"]
    gen = state["generation"]
    steps = state["workflow_steps"]
    tries = state.get("tries", 0)

    # Prevent infinite loops
    if tries >= 2:
        steps.append("⏹️ Max retries reached. Returning best-effort answer.")
        return {"documents": docs, "question": q, "generation": gen, "workflow_steps": steps, "tries": tries, "grounded": True, "useful": True}

    if not docs:
        steps.append("⚠️ No documents to ground against. Skipping hallucination check.")
        return {"documents": docs, "question": q, "generation": gen, "workflow_steps": steps, "tries": tries, "grounded": True, "useful": True}

    h_score = _hallucination_grader().invoke({"documents": "\n\n".join([d.page_content for d in docs]), "generation": gen})
    if "yes" in h_score.lower():
        steps.append("✅ Answer is grounded in documents.")
        a_score = _answer_grader().invoke({"question": q, "generation": gen})
        if "yes" in a_score.lower():
            steps.append("✅ Answer is useful.")
            return {"documents": docs, "question": q, "generation": gen, "workflow_steps": steps, "tries": tries, "grounded": True, "useful": True}
        else:
            steps.append("❌ Answer is not useful. Triggering corrective retrieval.")
            return {"documents": docs, "question": q, "generation": gen, "workflow_steps": steps, "tries": tries, "grounded": True, "useful": False}
    else:
        steps.append("❌ Answer is not grounded / hallucinated. Triggering corrective retrieval.")
        return {"documents": docs, "question": q, "generation": gen, "workflow_steps": steps, "tries": tries, "grounded": False}

# ── Edges ────────────────────────────────────────────────────────────────────
def decide_to_generate(state: GraphState):
    return "generate" if state["documents"] else "web_search"

def decide_after_generation(state: GraphState):
    if state.get("grounded") and state.get("useful"):
        return END
    return "web_search"

# ── Graph ──────────────────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)
workflow.add_node("grade_generation", grade_generation)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {"generate": "generate", "web_search": "web_search"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "grade_generation")
workflow.add_conditional_edges("grade_generation", decide_after_generation, {"web_search": "web_search", END: END})

_app = workflow.compile()

# ── Public API ───────────────────────────────────────────────────────────────
def run_pipeline(question: str):
    result = _app.invoke({
        "question": question,
        "generation": "",
        "documents": [],
        "workflow_steps": [],
        "tries": 0
    })
    return {
        "generation": result["generation"],
        "documents": result["documents"],
        "workflow_steps": result["workflow_steps"]
    }
