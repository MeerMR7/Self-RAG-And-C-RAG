import streamlit as st
import os
from openai import OpenAI  # We use the SDK as a bridge to xAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pypdf import PdfReader

# --- 1. SETUP ---
st.set_page_config(page_title="Grok-Doc AI", layout="wide")
st.title("🛡️ Secure-Doc AI: Grok-Powered RAG")

# --- 2. LOAD SECRETS ---
# Ensure these names match your "Secrets" exactly
grok_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")

if not grok_key or not tavily_key:
    st.error("❌ Missing Keys: Ensure XAI_API_KEY and TAVILY_API_KEY are in Secrets.")
    st.stop()

# Initialize Grok (using OpenAI-compatible provider)
client = OpenAI(
    api_key=grok_key,
    base_url="https://api.x.ai/v1"
)

# Initialize Tavily
search_tool = TavilySearchResults(api_key=tavily_key, k=3)

# --- 3. PDF LOADER ---
PDF_FILENAME = "Academic-Policy-Manual-for-Students3.pdf"

def load_pdf(filename):
    if os.path.exists(filename):
        try:
            reader = PdfReader(filename)
            return "".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return None

pdf_text = load_pdf(PDF_FILENAME)

# --- 4. SIDEBAR ---
st.sidebar.success("🚀 Grok-4.3 Connected")
st.sidebar.info("🌐 Tavily Search Active")
if pdf_text:
    st.sidebar.write(f"📂 Loaded: {PDF_FILENAME}")
else:
    st.sidebar.warning("⚠️ PDF Not Found - Defaulting to Web Search only.")

# --- 5. CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. AGENTIC LOGIC ---
if prompt := st.chat_input("Ask about academic policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Grok is analyzing..."):
            
            # Context Gathering
            doc_context = pdf_text[:80000] if pdf_text else "No document available."
            
            # Step 1: Search Web via Tavily for real-time validation
            st.caption("🌐 Consulting Tavily for real-time updates...")
            web_data = search_tool.invoke({"query": prompt})
            
            # Step 2: Grok Synthesis
            system_prompt = (
                "You are Grok-4.3. You are an expert academic advisor. "
                "Use the PROVIDED DOCUMENT for official rules, but if the WEB SEARCH "
                "contains more recent or conflicting dates/info, prioritize the most logical/recent answer. "
                f"\n\nOFFICIAL DOCUMENT: {doc_context}"
                f"\n\nWEB SEARCH RESULTS: {web_data}"
            )

            try:
                response = client.chat.completions.create(
                    model="grok-4.3", # 2026 Flagship Model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages
                    ],
                    temperature=0.2
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Grok API Error: {e}")
