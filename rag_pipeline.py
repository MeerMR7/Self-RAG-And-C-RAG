# ── ADDED at the very top ──────────────────────────────
import os

try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key, val in st.secrets.items():
            os.environ[key] = str(val)   # ← copies Streamlit secrets → os.environ
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()                        # ← still works locally with .env
except Exception:
    pass
# ──────────────────────────────────────────────────────
