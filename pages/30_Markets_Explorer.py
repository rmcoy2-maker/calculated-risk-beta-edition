# serving_ui/app/pages/30_Markets_Explorer.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve()

def find_repo_root() -> Path:
    for p in [HERE.parent] + list(HERE.parents):
        if (p / "streamlit_app.py").exists():
            return p
    return Path.cwd()

ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from core_engine.tools.market_grouping import prepare_markets_for_app
except Exception:
    def prepare_markets_for_app(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "event_id" not in out.columns:
            if {"home_team", "away_team"}.issubset(out.columns):
                out["event_id"] = (
                    out["home_team"].astype(str) + " vs " + out["away_team"].astype(str)
                )
            else:
                out["event_id"] = out.index.astype(str)

        sort_cols = [c for c in ["event_id", "market", "selection", "book", "price"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, na_position="last")

        return out

DATA_DIR = ROOT / "exports" / "markets"

@st.cache_data(show_spinner=False)
def load_markets(week: int) -> pd.DataFrame:
    path = DATA_DIR / f"nfl_week_{week:02d}_markets_full.csv"

    if not path.exists():
        raise FileNotFoundError(f"Markets file not found: {path}")

    return pd.read_csv(path)


# -------------------------------------------------
# Streamlit page
# -------------------------------------------------
def app():
    st.title("📊 Markets Explorer")

    week = st.number_input("NFL Week", 1, 22, 12)

    try:
        df_raw = load_markets(int(week))
    except Exception as e:
        st.error(str(e))
        return

    df = prepare_markets_for_app(df_raw)

    event_ids = sorted(df["event_id"].unique())
    event_id = st.selectbox("Game", event_ids)

    df_game = df[df["event_id"] == event_id]

    st.subheader("Markets + Player Props")
    st.dataframe(df_game, use_container_width=True, hide_index=True)