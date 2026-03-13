from __future__ import annotations
from pathlib import Path
import streamlit as st
import hashlib

st.set_page_config(page_title="Calculated Risk", page_icon="📈", layout="wide")


def hash_pw(password: str):
    return hashlib.sha256(password.encode()).hexdigest()
st.write("Login build: beta-password-set-2")
# USER DATABASE
# -----------------------------
USERS = {
    "murphey": hash_pw("ECvx554u"),
    "rmcoy001": hash_pw("Dwl0F8Wf"),
    "beta001": hash_pw("Dwl0F8Wg"),
    "beta002": hash_pw("Dwl0F8Wh"),
    "beta003": hash_pw("Dwl0F8Wfi"),
    "beta004": hash_pw("Dwl0F8Wj"),
    "beta005": hash_pw("Dwl0F8Wk"),
    "beta006": hash_pw("Dwl0F8Wl"),
    "beta007": hash_pw("Dwl0F8Wm"),
    "beta008": hash_pw("Dwl0F8Wn"),
    "beta009": hash_pw("Dwl0F8Wo"),
    "beta010": hash_pw("Dwl0F8Wp"),
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.authenticated:
    st.title("🔐 Calculated Risk Beta Access")
    st.caption("🧪 Private Beta — Authorized Testers Only")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == hash_pw(password):
            st.session_state.authenticated = True
            st.session_state.user = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()

st.sidebar.success(f"Logged in as {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

st.title("Calculated Risk")
st.write("Recovered rebuild shell")

ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"

files = {
    "Lines": EXPORTS / "lines_live_normalized.csv",
    "Model Probs": EXPORTS / "model_probs.csv",
    "Bets Log": EXPORTS / "bets_log.csv",
    "Bankroll": EXPORTS / "bankroll.csv",
}

st.subheader("System Status")

for name, path in files.items():
    if path.exists():
        st.success(f"{name}: OK")
    else:
        st.warning(f"{name}: Missing")

st.divider()
st.info("Use the sidebar to navigate the beta tools.")