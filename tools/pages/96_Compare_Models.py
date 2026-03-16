from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from tools.true_line_utils import compute_true_line

st.set_page_config(page_title="Compare Models", layout="wide")

st.title("📊 Compare Models")

EXPORTS = Path("exports")

path = EXPORTS / "edges_models.csv"

if not path.exists():
    st.error("edges_models.csv not found")
    st.stop()

df = pd.read_csv(path)

df = compute_true_line(df, prob_col="p_win", odds_col="american_odds")

df["ev"] = df["p_win"] * df["_payout_decimal"] - (1 - df["p_win"])

# -----------------------------
# MODEL SUMMARY
# -----------------------------

summary = (
    df.groupby("_model")
    .agg(
        rows=("ev", "count"),
        ev_mean=("ev", "mean"),
        edge_mean=("edge_prob", "mean"),
        elite=("value_tier", lambda x: (x == "Elite").sum()),
        strong=("value_tier", lambda x: (x == "Strong").sum())
    )
    .sort_values("ev_mean", ascending=False)
)

st.subheader("Model Summary")

st.dataframe(summary)

# -----------------------------
# EDGE DISTRIBUTION
# -----------------------------

st.subheader("Edge Distribution")

st.bar_chart(
    df["value_tier"].value_counts()
)

# -----------------------------
# SAMPLE
# -----------------------------

st.subheader("Sample Rows")

cols = [
    "_model",
    "p_win",
    "american_odds",
    "true_line",
    "edge_prob",
    "value_tier"
]

cols = [c for c in cols if c in df.columns]

st.dataframe(df[cols].head(200))