from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Line Shop", page_icon="📋", layout="wide")

st.title("📋 Line Shop")

root = Path(__file__).resolve().parents[3]
exports = root / "exports"

candidates = [
    exports / "lines_live_normalized.csv",
    exports / "lines_live_fixed.csv",
    exports / "lines_live.csv",
]

line_file = next((p for p in candidates if p.exists()), None)

if line_file is None:
    st.warning("No live lines file found in exports.")
    st.stop()

try:
    df = pd.read_csv(line_file, low_memory=False)
except Exception as exc:
    st.error(f"Could not read lines file: {exc}")
    st.stop()

st.caption(f"Source: {line_file}")
st.write(f"Rows: {len(df):,}")

book_col = next((c for c in ["book", "sportsbook"] if c in df.columns), None)
market_col = "market" if "market" in df.columns else None

view = df.copy()

if book_col:
    books = sorted(view[book_col].dropna().astype(str).unique().tolist())
    selected_books = st.multiselect("Book", books, default=books)
    view = view[view[book_col].astype(str).isin(selected_books)]

if market_col:
    markets = sorted(view[market_col].dropna().astype(str).unique().tolist())
    selected_markets = st.multiselect("Market", markets, default=markets)
    view = view[view[market_col].astype(str).isin(selected_markets)]

show_cols = [c for c in ["game_id", "sport", "market", "selection", "side", "book", "odds", "line"] if c in view.columns]
if not show_cols:
    show_cols = list(view.columns[:12])

st.dataframe(view[show_cols], use_container_width=True, height=600)
