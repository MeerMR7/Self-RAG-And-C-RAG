import streamlit as st
import os
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pypdf import PdfReader

# --- 1. SETUP ---
PROJECT_NAME = "🛡️ Secure-Doc AI: Self-Correcting RAG"
st.set_page_config(page_title=PROJECT_NAME, layout="wide")
st.title(PROJECT_NAME)

# Folder for your PDFs (Changed from 'data' to avoid your error)
PDF_FOLDER = "knowledge_base"
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

# --- 2. LOAD SECRETS ---
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not openai_key or not tavily_key:
    st.error("❌ Missing API Keys in GitHub Secrets!")
    st.stop()

client = OpenAI(api_key=openai_key)
search_tool = TavilySearchResults(api_key=tavily_key, k=3)

# --- 3. PDF PROCESSING FUNCTION ---
def get_pdf_text(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if not files:
        return None, None
    
    combined_text = ""
    first_file = files[0]
    reader = PdfReader(os.path.join(folder, first_file))
    for page in reader.pages:
        combined_text += page.extract_text()
    return combined_text, first_file

# --- 4. SIDEBAR STATUS ---
st.sidebar.success("🔑 Keys Connected")
pdf_content, pdf_name = get_pdf_text(PDF_FOLDER)

if pdf_name:
    st.sidebar.info(f"📂 Active Doc: {pdf_name}")
else:
    st.sidebar.warning("📂 No PDF found. Upload to 'knowledge_base' folder.")

# --- 5. CHAT SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. THE AGENTIC LOGIC (THE "FULL" PART) ---
if prompt := st.chat_input("Ask about the doc or the web..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Knowledge Base..."):
            
            context = ""
            # CRAG LOGIC: Check PDF first
            if pdf_content and len(pdf_content) > 10:
                # Simple check: Does the prompt's keywords exist in the PDF?
                if any(word.lower() in pdf_content.lower() for word in prompt.split()[:3]):
                    context = pdf_content[:4000] # Use PDF text as context
                    st.caption("✅ Info found in PDF.")
                else:
                    st.caption("⚠️ Not in PDF. Self-Correcting via Web Search...")
                    web_data = search_tool.invoke({"query": prompt})
                    context = str(web_data)
            else:
                st.caption("🌐 No PDF available. Using Web Search...")
                web_data = search_tool.invoke({"query": prompt})
                context = str(web_data)

            # FINAL GENERATION
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. Use this context: {context}"},
                    *st.session_state.messages
                ]
            )
            
            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
