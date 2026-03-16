from __future__ import annotations

from pathlib import Path
import streamlit as st


def run_page_check(page_file: str, page_title: str) -> None:
    try:
        p = Path(page_file).resolve()
    except Exception:
        return

    app_dir = p.parents[1]
    pages_dir = app_dir / "pages"
    project_root = p.parents[3] if len(p.parents) > 3 else p.parents[-1]
    exports_dir = project_root / "exports"

    st.sidebar.markdown("### System Check")
    st.sidebar.caption(f"Page: {p.name}")
    st.sidebar.caption(f"Title: {page_title}")

    checks = {
        "Page file": p.exists(),
        "Pages dir": pages_dir.exists(),
        "Exports dir": exports_dir.exists(),
        "Lines file": (exports_dir / "lines_live_normalized.csv").exists(),
        "Model probs": (exports_dir / "model_probs.csv").exists(),
        "Bets log": (exports_dir / "bets_log.csv").exists(),
        "Bankroll": (exports_dir / "bankroll.csv").exists(),
    }

    for label, ok in checks.items():
        if ok:
            st.sidebar.success(f"{label}: OK")
        else:
            st.sidebar.warning(f"{label}: Missing")

    home_like = []
    for f in app_dir.glob("*.py"):
        if "home" in f.stem.lower():
            home_like.append(f.name)
    for f in pages_dir.glob("*.py"):
        if "home" in f.stem.lower():
            home_like.append(f"pages/{f.name}")

    if len(home_like) > 1:
        st.sidebar.error("Duplicate Home-style files detected")
        with st.sidebar.expander("Review files"):
            for item in sorted(home_like):
                st.write(item)
