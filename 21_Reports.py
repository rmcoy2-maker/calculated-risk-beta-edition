from __future__ import annotations

import streamlit as st

def app() -> None:
    st.set_page_config(
        page_title="Reports",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Reports")
    st.info(
        "The legacy report engine file (`app_reports.py`) is not present in this rebuild. "
        "Use Reports Hub for report generation and scheduling."
    )

    st.markdown("### Available Reports Tools")
    st.write("- Generate single 3v1 editions")
    st.write("- Generate all editions for a week")
    st.write("- Run weekly report pipeline")
    st.write("- Check report PDF status")

    if st.button("Open Reports Hub"):
        try:
            st.switch_page("pages/22_Reports_Hub.py")
        except Exception as e:
            st.error(f"Could not open Reports Hub: {e}")

if __name__ == "__main__":
    app()