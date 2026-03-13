from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# -------------------------------------------------------------------
# Force real project root onto sys.path
# File assumed at: edge-finder/serving_ui_recovered/app/pages/24_Doc_Odds_AI.py
# or: edge-finder/serving_ui/app/pages/24_Doc_Odds_AI.py
# -------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()

# pages -> app -> serving_ui_recovered/app? adjust safely
PAGES_DIR = THIS_FILE.parent
APP_DIR = PAGES_DIR.parent

# Try to locate project root robustly
CANDIDATES = [
    APP_DIR.parent.parent,  # edge-finder if file is in serving_ui/app/pages
    APP_DIR.parent,         # edge-finder if file is in app/pages
]

ROOT = None
for candidate in CANDIDATES:
    if (candidate / "exports").exists() or (candidate / "tools").exists():
        ROOT = candidate
        break

if ROOT is None:
    ROOT = CANDIDATES[0]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS_DIR = ROOT / "tools"
EXPORTS_DIR = ROOT / "exports"
MODELS_DIR = ROOT / "models"

# -------------------------------------------------------------------
# Optional auth/debug stubs
# -------------------------------------------------------------------
def require_eligibility(*args, **kwargs):
    return True


def current_user():
    return None


def login():
    pass


def logout():
    pass


def show_logout():
    pass


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def go_to_page(page_path: str) -> None:
    """
    Native Streamlit page navigation.
    Example:
        go_to_page("pages/10_Edge_Scanner.py")
    """
    try:
        st.switch_page(page_path)
    except Exception as e:
        st.warning(f"Navigation unavailable: {e}")


def init_chat_state() -> None:
    if "doc_odds_messages" not in st.session_state:
        st.session_state.doc_odds_messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to Doc Odds AI. Ask about markets, reports, bankroll logic, "
                    "workflow questions, or model interpretation."
                ),
            }
        ]


def render_chat_history() -> None:
    for msg in st.session_state.doc_odds_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def generate_local_response(prompt: str) -> str:
    """
    Local placeholder response layer.
    Replace later with your real OpenAI / internal app logic.
    """
    text = prompt.strip().lower()

    if any(word in text for word in ["report", "reports", "3v1"]):
        return (
            "Reports mode detected. Use Reports Hub to generate weekly editions, "
            "or ask me to summarize what each report slot is meant to do."
        )

    if any(word in text for word in ["bankroll", "unit", "stake"]):
        return (
            "Bankroll topic detected. Keep stake sizing tied to edge quality, variance, "
            "and your governor rules rather than chasing absolute hit rate."
        )

    if any(word in text for word in ["market", "markets", "prop", "spread", "total"]):
        return (
            "Markets topic detected. Check whether you want game-level markets, player props, "
            "or grouped views before comparing edge and price."
        )

    if any(word in text for word in ["model", "models", "calibration", "ece"]):
        return (
            "Model topic detected. Start with calibration, confidence buckets, and whether the "
            "price-discovery logic matches the market type you are evaluating."
        )

    if any(word in text for word in ["hello", "hi", "hey"]):
        return "Hey Murphey — Doc Odds AI is online."

    return (
        "I’m ready. Ask me about Edge Finder workflow, reports, model interpretation, "
        "markets, bankroll structure, or page routing."
    )


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
def app() -> None:
    st.set_page_config(
        page_title="Doc Odds AI",
        page_icon="🤖",
        layout="wide",
    )

    require_eligibility()
    init_chat_state()

    st.title("🤖 Doc Odds AI")
    st.caption(
        f"Calculated Risk assistant layer • {datetime.now():%Y-%m-%d %H:%M}"
    )

    with st.sidebar:
        st.subheader("Doc Odds AI")
        st.write("Project root:")
        st.code(str(ROOT))

        st.markdown("---")
        st.subheader("Quick Navigation")

        if st.button("📈 Edge Scanner"):
            go_to_page("pages/10_Edge_Scanner.py")

        if st.button("📊 Markets Explorer"):
            go_to_page("pages/30_Markets_Explorer.py")

        if st.button("🗃️ Reports Hub"):
            go_to_page("pages/22_Reports_Hub.py")

        if st.button("🏦 Bankroll Tracker"):
            go_to_page("pages/04_Bankroll_Tracker.py")

        if st.button("🏠 Home"):
            try:
                st.switch_page("00_Home.py")
            except Exception:
                go_to_page("pages/00_Home.py")

        st.markdown("---")
        st.subheader("Status")
        st.write(f"Tools dir exists: {'Yes' if TOOLS_DIR.exists() else 'No'}")
        st.write(f"Exports dir exists: {'Yes' if EXPORTS_DIR.exists() else 'No'}")
        st.write(f"Models dir exists: {'Yes' if MODELS_DIR.exists() else 'No'}")

    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Chat")
        render_chat_history()

        user_prompt = st.chat_input("Ask Doc Odds AI something...")
        if user_prompt:
            st.session_state.doc_odds_messages.append(
                {"role": "user", "content": user_prompt}
            )

            with st.chat_message("user"):
                st.markdown(user_prompt)

            reply = generate_local_response(user_prompt)
            st.session_state.doc_odds_messages.append(
                {"role": "assistant", "content": reply}
            )

            with st.chat_message("assistant"):
                st.markdown(reply)

    with right:
        st.markdown("### Quick Actions")
        st.info(
            "This page is now syntax-clean and dependency-light. "
            "It uses native Streamlit navigation and avoids `streamlit_extras`."
        )

        st.markdown("### Suggested Uses")
        st.write("- Ask workflow questions")
        st.write("- Route to major app pages")
        st.write("- Add OpenAI integration later")
        st.write("- Add report summary logic later")

        if st.button("Clear chat"):
            st.session_state.doc_odds_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Chat cleared. Ask about markets, reports, bankroll, "
                        "models, or app workflow."
                    ),
                }
            ]
            st.rerun()


if __name__ == "__main__":
    app()