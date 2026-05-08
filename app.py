import streamlit as st
import os
from openai import OpenAI

# Change the name of your chatbot here
st.title("🛡️ My Secure Self-Correcting AI")

# This looks for the keys you put in GitHub Secrets
# (Make sure they are named exactly 'OPENAI_API_KEY' and 'TAVILY_API_KEY' in GitHub)
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# If the secret isn't found, it shows the manual box as a backup
if not openai_key:
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

if openai_key:
    client = OpenAI(api_key=openai_key)
    st.success("✅ Connected to OpenAI!")
    
    # Simple check for your PDF
    if os.path.exists("data") and os.listdir("data"):
        st.info(f"📂 Found PDF: {os.listdir('data')[0]}")
    
    # ... rest of your chat logic ...
else:
    st.warning("Please enter your API key or ensure it is set in GitHub Secrets.")
