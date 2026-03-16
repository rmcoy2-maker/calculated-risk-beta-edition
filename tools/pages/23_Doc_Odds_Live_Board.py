# --- imports unchanged ---
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from tools.true_line_utils import compute_true_line

# page
st.set_page_config(page_title="Doc Odds Live Board", layout="wide")

st.title("📊 Doc Odds Live Board")

EXPORTS = Path("exports")
LINES_CSV = EXPORTS / "lines_live.csv"
MODEL_CSV = EXPORTS / "model_probs.csv"


def american_to_decimal(odds):
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def load_lines(path):

    df = pd.read_csv(path)

    df["Decimal"] = df["American"].apply(american_to_decimal)

    return df


def load_model_probs(path):

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    return df


df_lines = load_lines(LINES_CSV)

df_model = load_model_probs(MODEL_CSV)

if not df_model.empty:

    df = df_lines.merge(
        df_model,
        on=["game_id", "market", "selection"],
        how="left"
    ).rename(columns={"prob": "Model Prob"})

else:

    df = df_lines.copy()
    df["Model Prob"] = np.nan


# -----------------------------
# TRUE LINE ENGINE
# -----------------------------

df = compute_true_line(df, prob_col="Model Prob", odds_col="American")


# -----------------------------
# EV
# -----------------------------

df["EV"] = df["Model Prob"] * df["Decimal"] - 1


# -----------------------------
# BEST PRICE
# -----------------------------

best = df.sort_values("American", ascending=False).drop_duplicates(
    ["game_id", "market", "selection"]
)

# -----------------------------
# DISPLAY
# -----------------------------

cols = [
    "league",
    "game_id",
    "market",
    "selection",
    "American",
    "true_line",
    "market_prob",
    "true_prob",
    "edge_prob",
    "value_tier",
    "governor",
    "EV"
]

cols = [c for c in cols if c in best.columns]

st.dataframe(
    best[cols].sort_values("edge_prob", ascending=False),
    use_container_width=True
)

st.caption("True Line Engine + Governor active")