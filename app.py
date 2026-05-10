import streamlit as st
import os
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pypdf import PdfReader

# --- 1. SETUP ---
PROJECT_NAME = "🛡️ Secure-Doc AI: Groq & Tavily RAG"
st.set_page_config(page_title=PROJECT_NAME, layout="wide")
st.title(PROJECT_NAME)

# --- 2. LOAD SECRETS ---
# Using the exact names from your screenshot
groq_key = os.getenv("GROK_API_KEY") or st.secrets.get("GROK_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")

if not groq_key or not tavily_key:
    st.error("❌ Missing Keys! Ensure your Secrets are named 'GROK_API_KEY' and 'TAVILY_API_KEY' exactly.")
    st.stop()

# Initialize the Groq Client (using the 'gsk_' key)
# Note: We use the OpenAI library because Groq is fully compatible
client = OpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1"
)

# Initialize Tavily Search
search_tool = TavilySearchResults(api_key=tavily_key, k=3)

# --- 3. PDF LOADER ---
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
st.sidebar.success("⚡ Groq LPU Connected")
st.sidebar.info("🌐 Tavily Search Active")
if pdf_text:
    st.sidebar.write(f"📂 Document Loaded: {PDF_FILENAME}")
else:
    st.sidebar.warning(f"⚠️ {PDF_FILENAME} not found in the main folder.")

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
        with st.spinner("Analyzing..."):
            
            # Step A: Get Document Context (First 30k characters for Groq)
            doc_context = pdf_text[:30000] if pdf_text else "No local document found."
            
            # Step B: Get Web Context via Tavily
            st.caption("🌐 Validating with Web Search...")
            web_results = search_tool.invoke({"query": prompt})
            
            # Step C: Generate Final Answer using Groq's high-speed Llama model
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", # Current flagship on Groq
                    messages=[
                        {
                            "role": "system", 
                            "content": (
                                "You are an Academic Advisor. Answer using the Document context first. "
                                "If the information is missing or the Web Search provides more recent "
                                "updates (like 2025/2026 dates), use the Web results to correct the answer."
                                f"\n\nDOCUMENT: {doc_context}"
                                f"\n\nWEB SEARCH: {web_results}"
                            )
                        },
                        *st.session_state.messages
                    ],
                    temperature=0.2
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Groq API Error: {str(e)}")
