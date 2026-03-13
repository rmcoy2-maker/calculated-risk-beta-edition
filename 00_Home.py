from __future__ import annotations
from pathlib import Path
import streamlit as st
import hashlib

st.set_page_config(page_title="Calculated Risk", page_icon="📈", layout="wide")

# -----------------------------
# PASSWORD HASH FUNCTION
# -----------------------------
def hash_pw(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
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

# -----------------------------
# SESSION STATE
# -----------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None

# -----------------------------
# LOGIN SCREEN
# -----------------------------
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
# -----------------------------
# LOGGED IN VIEW
# -----------------------------
st.sidebar.success(f"Logged in as {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

# -----------------------------
# MAIN APP
# -----------------------------
st.title("Calculated Risk")

st.write("Recovered rebuild shell")

ROOT = Path(__file__).resolve().parents[2]
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

pages_dir = Path(__file__).resolve().parent / "pages"

def add_page(filename: str, label: str):
    if (pages_dir / filename).exists():
        st.page_link(f"pages/{filename}", label=label)

add_page("04_Bankroll_Tracker.py", "💰 Bankroll Tracker")
add_page("10_Edge_Scanner.py", "🎯 Edge Scanner")
add_page("11_Hedge_Finder.py", "⚖️ Hedge Finder")
add_page("21_Reports.py", "📝 Reports")
add_page("22_Reports_Hub.py", "📊 Reports Hub")
add_page("23_Doc_Odds_Live_Board.py", "📡 Doc Odds Live Board")
add_page("24_Doc_Odds_AI.py", "🤖 Doc Odds AI")
add_page("25_Bet_Engine.py", "🧠 Bet Engine")
add_page("09_Analytics_Hub.py", "📊 Analytics Hub")
add_page("09_Lines_Explorer.py", "📉 Lines Explorer")
add_page("09_All_Picks_Explorer.py", "⚙️ All Picks Explorer")
add_page("12_Micro_Calculations.py", "🧮 Micro Calculations")
add_page("13_Your_Log.py", "📖 Your Log")
add_page("14_Your_History.py", "📜 Your History")
add_page("93_Analytics.py", "📈 Premium Analytics")
add_page("94_Legal_Terms_Privacy.py", "⚖️ Legal / Privacy")
add_page("95_Models_Browser.py", "🧠 Models Browser")
add_page("96_Compare_Models.py", "🔬 Compare Models")
add_page("96_Data_Diagnostics.py", "🧪 Data Diagnostics")
add_page("97_Settle_Diagnostics.py", "🧪 Settle Diagnostics")
add_page("98_Diagnostics.py", "🧪 Diagnostics")
add_page("02_User_manual.py", "📘 User Manual")
add_page("01_Shortcuts.py", "⚡ Shortcuts")
add_page("00_Line_Shop.py", "📋 Line Shop")
add_page("03_Backtest.py", "📚 Backtest")
add_page("06_Ghost_Parlay_Calc.py", "👻 Ghost Parlay Calc")
add_page("07_Parlay_Scored_Explorer.py", "🧾 Parlay Scored Explorer")
add_page("08_Settle_Parlay.py", "✅ Settle Parlay")