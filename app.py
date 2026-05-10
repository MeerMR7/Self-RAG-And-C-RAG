import streamlit as st
import os
from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pypdf import PdfReader

# --- 1. SETUP ---
st.set_page_config(page_title="Voice Groq-Doc AI", layout="wide")
st.title("🎙️ Secure-Doc AI: Voice & Web RAG")

# --- 2. LOAD SECRETS ---
# Match the exact names from your screenshot
groq_key = os.getenv("GROK_API_KEY") or st.secrets.get("GROK_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")

if not groq_key or not tavily_key:
    st.error("❌ Missing Keys! Ensure Secrets are named 'GROK_API_KEY' and 'TAVILY_API_KEY'.")
    st.stop()

client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
search_tool = TavilySearchResults(api_key=tavily_key, k=2)

# --- 3. PDF LOADER ---
PDF_FILENAME = "Academic-Policy-Manual-for-Students3.pdf"

@st.cache_data
def load_pdf(filename):
    if os.path.exists(filename):
        try:
            reader = PdfReader(filename)
            return "".join([p.extract_text() for p in reader.pages])
        except Exception:
            return None
    return None

pdf_text = load_pdf(PDF_FILENAME)

# --- 4. VOICE INPUT SIDEBAR ---
st.sidebar.header("🎤 Mir Jo Awaaz")
audio_file = st.sidebar.audio_input("Click to record your question")

voice_prompt = None
if audio_file:
    try:
        with st.spinner("Transcribing..."):
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )
            voice_prompt = transcription
    except Exception as e:
        st.sidebar.error(f"Voice Error: {e}")

# --- 5. CHAT SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

text_input = st.chat_input("Or type your message here...")
# Use voice if available, otherwise use text
final_prompt = voice_prompt if voice_prompt else text_input

# --- 6. OPTIMIZED LOGIC ---
if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # Limit PDF text to 8k characters to avoid 429 Errors
            context = pdf_text[:8000] if pdf_text else "No document found."
            
            try:
                # Search the web
                web_data = search_tool.invoke({"query": final_prompt})
                
                # Use Llama-3.1-8b for MUCH higher rate limits
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant", 
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You are a helpful assistant. Use this Doc: {context}\n\nWeb info: {web_data}"
                        },
                        *st.session_state.messages
                    ],
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                if "429" in str(e):
                    st.error("⚠️ Rate limit hit. I've switched to a smaller model to help, but Groq's free tier is busy. Try again in a minute!")
                else:
                    st.error(f"Error: {e}")
