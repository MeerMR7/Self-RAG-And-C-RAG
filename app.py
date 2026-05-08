import streamlit as st
import os
from openai import OpenAI

# 1. NEW NAME FOR YOUR PROJECT
st.set_page_config(page_title="Self-Correcting RAG", layout="wide")
st.title("🛡️ Secure-Doc AI: Self-Correcting RAG")

# 2. GRABBING SECRETS FROM GITHUB
# This pulls the keys you put in "Secrets"
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not openai_key:
    st.error("❌ OpenAI Key not found in Secrets. Please restart your Codespace or enter it in the sidebar.")
    with st.sidebar:
        openai_key = st.text_input("Manual OpenAI Key", type="password")

if openai_key:
    client = OpenAI(api_key=openai_key)
    st.success("✅ OpenAI API Key Connected!")
    
    # Simple check for your PDF folder
    if os.path.exists("data") and os.listdir("data"):
        st.info(f"📂 PDF Found: {os.listdir('data')[0]}")
    else:
        st.warning("📂 No PDF found. Please upload your PDF to the 'data' folder in the sidebar.")

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your PDF or the web..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # This is the "Self-Correction" spot
            # If PDF is missing, it will use the LLM (and later we add Tavily search)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            )
            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
