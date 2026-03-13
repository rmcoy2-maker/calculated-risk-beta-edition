from __future__ import annotations
from pathlib import Path
import streamlit as st
import hashlib

st.set_page_config(page_title="Calculated Risk", page_icon="📈", layout="wide")


def hash_pw(password: str):
    return hashlib.sha256(password.encode()).hexdigest()


USERS = {
    "murphey": "6abb4f247be995cbc6619c92da440bc16ee03407ecce16d6f4a2357cdadf8d1f",
    "rmcoy001": "28d80bec4ce5e09b72d4e085cd51df1444081d8cd9aca6eac85015311e1b1ca9",
    "beta001": "df621b7d4b7be66f5098490c97447ff73d7410e3fbe5e3e5410430dd5f5fac48",
    "beta002": "afd4612e3a276b49c0d626818317c57c318685726cec64c3f6bae968e6f7c101",
    "beta003": "305f04205ac28f8aae1b2882a0eec206c15c7ac804044ceea081defa9fcc185a",
    "beta004": "d665aac5a6bf521ec21e7d532d73eb94755734e1bad2d6e3118fa9119accadd4",
    "beta005": "95f4b7738761ccec8c0a7b0c91f480cfecf18788ae22eb29b89c5148d9f86d9a",
    "beta006": "67692a2f477d3be238e07a7f7c679861e54a359d666038a2eeac075029896ab2",
    "beta007": "117476de2f2e1e38e72622f7a63b0edf8976c629d11ba6fe228ca481962fb826",
    "beta008": "b7107b458eaa182043a2466a87fa53dc906478a80100ef5e3988ca1dbf881d73",
    "beta009": "ef54a110ffdd135a419fa325b12d56bb4a85a568c3c7077bdce0c6143e533798",
    "beta010": "d715dddb14292761ee19d5f20c4b1341d2e6d032ef20c805af48fc3157e989b1",
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.authenticated:
    st.title("🔐 Calculated Risk Beta Access")

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