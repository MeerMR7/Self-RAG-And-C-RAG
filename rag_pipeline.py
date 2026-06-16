import os
from functools import lru_cache
from typing import List, Dict, Any

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------- CONFIG ----------------
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------- LLM ----------------
@lru_cache(maxsize=1)
def get_llm():
    google_key = os.environ.get("GOOGLE_API_KEY")

    if not google_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to Streamlit Secrets or your local .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=google_key,
        temperature=0.2,
        max_output_tokens=800,
        request_timeout=35,
        max_retries=2,
    )


# ---------------- VECTOR RETRIEVER ----------------
@lru_cache(maxsize=1)
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})


# ---------------- WEB SEARCH ----------------
def get_web_search_tool():
    tavily_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_key:
        return None

    return TavilySearchResults(
        api_key=tavily_key,
        max_results=3,
    )


# ---------------- SELF-RAG: RETRIEVE ----------------
def retrieve_documents(question: str, steps: List[str]) -> List[Document]:
    if not os.path.exists(CHROMA_DIR):
        steps.append("No ChromaDB found. Local document retrieval skipped.")
        return []

    try:
        retriever = get_retriever()
        docs = retriever.invoke(question)

        if docs:
            steps.append(f"Retrieved {len(docs)} documents from local vector database.")
        else:
            steps.append("No documents found in local vector database.")

        return docs

    except Exception as e:
        steps.append(f"Local retrieval failed: {str(e)}")
        return []


# ---------------- SELF-RAG: RELEVANCE CHECK ----------------
def simple_relevance_filter(question: str, docs: List[Document], steps: List[str]) -> List[Document]:
    if not docs:
        return []

    question_words = set(question.lower().split())
    relevant_docs = []

    for doc in docs:
        text = doc.page_content.lower()
        score = sum(1 for word in question_words if word in text)

        if score > 0:
            relevant_docs.append(doc)

    if relevant_docs:
        steps.append(f"Self-RAG relevance check kept {len(relevant_docs)} relevant documents.")
        return relevant_docs

    steps.append("Self-RAG relevance check found weak local context.")
    return []


# ---------------- C-RAG: WEB CORRECTION ----------------
def web_search_documents(question: str, steps: List[str]) -> List[Document]:
    tool = get_web_search_tool()

    if tool is None:
        steps.append("Tavily API key missing. Web correction skipped.")
        return []

    try:
        results = tool.invoke({"query": question})
        web_docs = []

        for result in results:
            if isinstance(result, dict):
                content = result.get("content") or result.get("snippet") or ""
                url = result.get("url") or "Web Search"

                if content:
                    web_docs.append(
                        Document(
                            page_content=content[:1500],
                            metadata={"source": url},
                        )
                    )

        steps.append(f"C-RAG web correction returned {len(web_docs)} web results.")
        return web_docs

    except Exception as e:
        steps.append(f"Web correction failed: {str(e)}")
        return []


# ---------------- ANSWER GENERATION ----------------
def generate_answer(question: str, docs: List[Document], steps: List[str]) -> str:
    if docs:
        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Document')}\n{doc.page_content[:1500]}"
                for doc in docs
            ]
        )
    else:
        context = (
            "No external context was available. "
            "Answer using general academic knowledge and clearly avoid unsupported claims."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an academic FYP chatbot for a project named:
Smart Self Correcting AI (Self-RAG and C-RAG).

Your job:
- Answer clearly and professionally.
- Use the provided context when available.
- If context is weak or missing, still provide a helpful general answer.
- Keep the answer easy to understand for students.
- Do not get stuck.
- Do not say only "I don't know" unless truly necessary.
""",
            ),
            (
                "human",
                """
Question:
{question}

Context:
{context}

Final Answer:
""",
            ),
        ]
    )

    chain = prompt | get_llm() | StrOutputParser()

    answer = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    steps.append("Generated final response successfully.")
    return answer


# ---------------- MAIN PIPELINE ----------------
def run_pipeline(question: str) -> Dict[str, Any]:
    steps = []
    documents = []

    steps.append("Started Smart Self Correcting AI pipeline.")

    # Step 1: Local RAG retrieval
    local_docs = retrieve_documents(question, steps)

    # Step 2: Self-RAG relevance checking
    relevant_docs = simple_relevance_filter(question, local_docs, steps)

    if relevant_docs:
        documents.extend(relevant_docs)
        steps.append("Self-RAG accepted local retrieved context.")
    else:
        steps.append("Self-RAG rejected or found weak local context.")

    # Step 3: C-RAG correction through web search if local context is weak
    if not documents:
        steps.append("C-RAG correction triggered.")
        web_docs = web_search_documents(question, steps)

        if web_docs:
            documents.extend(web_docs)
            steps.append("C-RAG added corrected external context.")
        else:
            steps.append("C-RAG could not add external context. Using LLM fallback answer.")

    # Step 4: Generate final answer
    answer = generate_answer(question, documents, steps)

    steps.append("Pipeline completed successfully.")

    return {
        "generation": answer,
        "documents": documents,
        "workflow_steps": steps,
    }


# Compatibility alias for older app files
run_rag_pipeline = run_pipeline
