import streamlit as st

def premium_enabled() -> bool:
    auth = st.session_state.get("auth", {})
    username = (
        auth.get("username")
        or auth.get("user")
        or st.session_state.get("user")
        or ""
    )
    username = str(username).strip().lower()

    premium_users = {
        "murphey",
        "rmcoy2",
        "beta001",
        "beta002",
        "beta003",
    }

    if username in premium_users or username.startswith("beta"):
        return True

    role = str(auth.get("role", "")).strip().lower()
    return role in {"premium", "pro", "elite", "hof", "admin"}
st.session_state["user"] = username
st.session_state["auth"] = {
    "authenticated": True,
    "username": username,
    "role": "premium" if str(username).lower().startswith("beta") else "user",
}