"""Load configuration from a local .env file, or from Streamlit Cloud's
secrets manager when this app is deployed there instead of run locally.
"""

import os

from dotenv import load_dotenv


def _get(key):
    # On Streamlit Community Cloud, keys added in the app's "Secrets" panel
    # show up in st.secrets. Locally (no secrets.toml), reading st.secrets
    # can raise -- caught here so it falls back to .env without extra setup.
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


def load_config():
    load_dotenv()
    return {
        "NEWSAPI_KEY": (_get("NEWSAPI_KEY") or "").strip() or None,
        "COINGECKO_API_KEY": (_get("COINGECKO_API_KEY") or "").strip() or None,
    }
