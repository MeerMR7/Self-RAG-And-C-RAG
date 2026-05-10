import streamlit as st
import os
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pypdf import PdfReader

# --- 1. SETUP ---
st.set_page_config(page_title="Voice Groq-Doc AI", layout="wide")

# --- 2. LOAD SECRETS ---
groq_key = os.getenv("GROK_API_KEY") or st.secrets.get("GROK_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")

if not groq_key or not tavily_key:
    st.error("❌ Missing Keys! Check your Secrets for GROK_API_KEY and TAVILY_API_KEY.")
    st.stop()

client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
search_tool = TavilySearchResults(api_key=tavily_key, k=3)

# --- 3. PDF LOADER ---
PDF_FILENAME = "Academic-Policy-Manual-for-Students3.pdf"

@st.cache_data
def load_pdf(filename):
    if os.path.exists(filename):
        reader = PdfReader(filename)
        return "".join([p.extract_text() for p in reader.pages])
    return None

pdf_text = load_pdf(PDF_FILENAME)

# --- 4. VOICE & TEXT INPUT ---
st.sidebar.header("🎤 Voice Input")
audio_file = st.sidebar.audio_input("Record your question")

# Transcription Logic
voice_prompt = None
if audio_file:
    with st.spinner("Transcribing your voice..."):
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", audio_file.read()),
            model="whisper-large-v3-turbo", # Fast & Accurate
            response_format="text"
        )
        voice_prompt = transcription

# --- 5. CHAT SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Determine which prompt to use (Voice takes priority if recorded)
text_prompt = st.chat_input("Or type your question here...")
final_prompt = voice_prompt if voice_prompt else text_prompt

# --- 6. AGENTIC LOGIC ---
if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Grok/Llama is thinking..."):
            
            # Context
            doc_context = pdf_text[:30000] if pdf_text else ""
            web_data = search_tool.invoke({"query": final_prompt})
            
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": f"Use Doc: {doc_context}\nWeb: {web_data}"},
                        *st.session_state.messages
                    ]
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")
