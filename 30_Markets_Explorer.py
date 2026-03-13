# serving_ui/app/pages/30_Markets_Explorer.py

from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# -------------------------------------------------
# Ensure project root is on path
# -------------------------------------------------
CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_engine.tools.market_grouping import prepare_markets_for_app

# -------------------------------------------------
# Data paths
# -------------------------------------------------
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