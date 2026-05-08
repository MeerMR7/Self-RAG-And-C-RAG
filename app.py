import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

# --- AGENT STATE ---
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

# --- NODES ---
def retrieve(state):
    db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    docs = db.similarity_search(state["question"], k=2)
    return {"documents": [d.page_content for d in docs]}

def generate(state):
    llm = ChatOpenAI(model="gpt-4o-mini")
    # Simple logic: if no docs found, use web search (CRAG logic)
    if not state["documents"]:
        search = TavilySearchResults(k=2)
        web_results = search.invoke({"query": state["question"]})
        context = str(web_results)
    else:
        context = "\n".join(state["documents"])
    
    res = llm.invoke(f"Question: {state['question']}\nContext: {context}")
    return {"generation": res.content}

# --- GRAPH ASSEMBLY ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
agent = workflow.compile()

# --- STREAMLIT UI ---
st.title("🚀 Smart AI Agent")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("Thinking..."):
        result = agent.invoke({"question": prompt})
        answer = result["generation"]
        
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
