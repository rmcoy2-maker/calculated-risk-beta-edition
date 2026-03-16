from __future__ import annotations


def page_gate():
    """
    Lightweight page guard for recovered/local environments.
    Returns (auth, st) so pages can use them safely.
    """
    import streamlit as st

    try:
        from app.lib.auth import login, show_logout  # type: ignore
    except Exception:
        def login(required: bool = False):
            class _Auth:
                ok = True
                authenticated = True
            return _Auth()

        def show_logout():
            return None

    try:
        from app.lib.compliance_gate import require_eligibility  # type: ignore
    except Exception:
        def require_eligibility(*args, **kwargs):
            return True

    auth = login(required=False)

    if not getattr(auth, "ok", True):
        st.stop()

    try:
        show_logout()
    except Exception:
        pass

    try:
        require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})
    except Exception:
        pass

    return auth, st