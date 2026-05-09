import streamlit as st
import os
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. SET YOUR PROJECT NAME ---
PROJECT_NAME = "🛡️ Secure-Doc AI: Self-Correcting RAG"
st.set_page_config(page_title=PROJECT_NAME, layout="wide")
st.title(PROJECT_NAME)

# --- 2. SECRETS LOADING ---
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if openai_key and tavily_key:
    client = OpenAI(api_key=openai_key)
    search_tool = TavilySearchResults(api_key=tavily_key, k=3)
    st.sidebar.success("🔑 All Systems Connected")

    # --- 3. PDF CHECKER ---
    if not os.path.exists("data"):
        os.makedirs("data")
    
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    
    if pdf_files:
        st.sidebar.info(f"📂 Reading: {pdf_files[0]}")
        # Note: In a full RAG, we would process embeddings here. 
        # For now, we are setting up the Agentic Loop.
    else:
        st.sidebar.warning("📂 Please upload a PDF to the 'data' folder.")

    # --- 4. CHAT LOGIC ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about the document or search the web..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent Thinking (Retrieving -> Grading -> Generating)..."):
                # LOGIC: If no PDF data, perform C-RAG (Web Search)
                if not pdf_files:
                    st.caption("Self-Correction: No local docs found. Triggering Web Search...")
                    search_results = search_tool.invoke({"query": prompt})
                    context = str(search_results)
                else:
                    context = "Local PDF Data" # Placeholder for processed PDF text

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": f"Use this context: {context}"}] + 
                             [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
