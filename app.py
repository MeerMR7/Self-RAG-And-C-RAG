import streamlit as st
import os
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pypdf import PdfReader

# --- 1. SETUP ---
PROJECT_NAME = "🛡️ Secure-Doc AI: Self-Correcting RAG"
st.set_page_config(page_title=PROJECT_NAME, layout="wide")
st.title(PROJECT_NAME)

# --- 2. LOAD SECRETS ---
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not openai_key or not tavily_key:
    st.error("❌ Missing API Keys in GitHub Secrets! Please RESTART your Codespace.")
    st.stop()

client = OpenAI(api_key=openai_key)
search_tool = TavilySearchResults(api_key=tavily_key, k=3)

# --- 3. DIRECT FILE LOADER ---
# This looks for your specific file in the main directory
PDF_FILENAME = "Academic-Policy-Manual-for-Students3.pdf"

def load_my_pdf(filename):
    if os.path.exists(filename):
        try:
            reader = PdfReader(filename)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    return None

pdf_text = load_my_pdf(PDF_FILENAME)

# --- 4. SIDEBAR STATUS ---
st.sidebar.success("🔑 All Systems Connected")
if pdf_text:
    st.sidebar.info(f"📂 Reading: {PDF_FILENAME}")
else:
    st.sidebar.error(f"❌ Could not find {PDF_FILENAME} in the main folder!")

# --- 5. CHAT SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. SELF-CORRECTING LOGIC ---
if prompt := st.chat_input("Ask me about the Academic Policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Document & Web..."):
            
            context = ""
            # Logic: If PDF is loaded, use it. Otherwise, use Web Search.
            if pdf_text:
                # We give the AI the PDF text as context
                context = pdf_text[:5000] # Taking the first 5000 characters
                st.caption("🔍 Analyzing Document...")
            else:
                st.caption("🌐 PDF Missing. Searching Web...")
                web_results = search_tool.invoke({"query": prompt})
                context = str(web_results)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Use this context to answer: {context}"},
                    *st.session_state.messages
                ]
            )
            
            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
