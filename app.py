import streamlit as st
import os
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pypdf import PdfReader

# --- 1. SETUP ---
PROJECT_NAME = "🛡️ Secure-Doc AI: Grok-Powered RAG"
st.set_page_config(page_title=PROJECT_NAME, layout="wide")
st.title(PROJECT_NAME)

# --- 2. LOAD SECRETS ---
# This version works whether you used GitHub Secrets or Streamlit Secrets
grok_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")

if not grok_key or not tavily_key:
    st.error("❌ Missing API Keys! Ensure XAI_API_KEY and TAVILY_API_KEY are in your Secrets.")
    st.stop()

# Initialize the Grok Client via OpenAI SDK compatibility
client = OpenAI(
    api_key=grok_key,
    base_url="https://api.x.ai/v1"
)
search_tool = TavilySearchResults(api_key=tavily_key, k=3)

# --- 3. DIRECT FILE LOADER ---
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
st.sidebar.success("🚀 Grok Engine Connected")
if pdf_text:
    st.sidebar.info(f"📂 Reading: {PDF_FILENAME}")
else:
    st.sidebar.error(f"❌ Could not find {PDF_FILENAME} in the folder!")

# --- 5. CHAT SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. AGENTIC LOGIC ---
if prompt := st.chat_input("Ask me about the Academic Policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Grok is processing..."):
            
            context = ""
            # Enhanced Context: Grok-4.3 handles massive context, 
            # so we use a much larger slice of the PDF (approx 40 pages)
            if pdf_text:
                context = pdf_text[:100000] 
                st.caption("🔍 Analyzing Document with Grok-4.3...")
            else:
                st.caption("🌐 PDF Missing. Grok is searching Web...")
                web_results = search_tool.invoke({"query": prompt})
                context = str(web_results)

            try:
                response = client.chat.completions.create(
                    model="grok-4.3", # Using the 2026 flagship model
                    messages=[
                        {"role": "system", "content": f"You are Grok. Use this context to answer precisely: {context}"},
                        *st.session_state.messages
                    ],
                    temperature=0.1 # Lower temperature for factual accuracy
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"API Error: {str(e)}")
