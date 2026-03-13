from __future__ import annotations

from pathlib import Path
import json
import tempfile
import streamlit as st

st.set_page_config(page_title="Shortcuts", page_icon="⚡", layout="wide")
st.title("⚡ Shortcuts")


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "streamlit_app.py").exists():
            return p
    return Path.cwd()


def get_exports_dir() -> Path:
    root = find_repo_root()

    candidates = [
        root / "exports",
        Path(tempfile.gettempdir()) / "calculated_risk_exports",
    ]

    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            continue

    return Path(tempfile.gettempdir())


exports = get_exports_dir()
shortcuts_file = exports / "shortcuts.json"

if shortcuts_file.exists():
    try:
        shortcuts = json.loads(shortcuts_file.read_text(encoding="utf-8"))
    except Exception:
        shortcuts = {}
else:
    shortcuts = {}

default_books = shortcuts.get("books", [])

books_text = st.text_area(
    "Favorite books (comma separated)",
    value=", ".join(default_books),
    height=120,
)

if st.button("Save shortcuts"):
    books = [x.strip() for x in books_text.split(",") if x.strip()]
    payload = {"books": books}
    try:
        shortcuts_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        shortcuts = payload
        st.success(f"Saved to {shortcuts_file}")
    except Exception as e:
        st.error(f"Could not save shortcuts: {e}")

st.subheader("Current shortcuts")
st.json(shortcuts if shortcuts else {"books": []})