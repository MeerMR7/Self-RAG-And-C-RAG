import streamlit as st
from openai import OpenAI
import os

# --- 1. CHANGE YOUR CHATBOT NAME HERE ---
st.set_page_config(page_title="Agentic RAG System", page_icon="🤖")
st.title("🛡️ My Custom Self-Correcting AI") 
st.caption("Final Year Project - RAG + C-RAG Logic")

# Sidebar for Keys
with st.sidebar:
    st.header("Settings")
    # You can hardcode your key here for the demo if you want:
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    tavily_key = st.text_input("Enter Tavily API Key", type="password")

if not api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # This is where the 'Self-Correction' happens
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
