from __future__ import annotations

from pathlib import Path
import json
import streamlit as st

st.set_page_config(page_title="Shortcuts", page_icon="⚡", layout="wide")

st.title("⚡ Shortcuts")

root = Path(__file__).resolve().parents[3]
exports = root / "exports"
exports.mkdir(parents=True, exist_ok=True)
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
    shortcuts_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    st.success(f"Saved to {shortcuts_file}")

st.subheader("Current shortcuts")
st.json(shortcuts if shortcuts else {"books": []})
