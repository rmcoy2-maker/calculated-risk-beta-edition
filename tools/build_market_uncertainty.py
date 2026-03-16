import pandas as pd
import numpy as np

df = pd.read_csv("analysis_out/true_line/true_line_board.csv", low_memory=False)

num_cols = [
    "open_prob",
    "mid_prob",
    "close_prob",
    "open_to_close_prob_move",
    "mid_to_close_prob_move",
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Basic move metrics
df["abs_open_close_move"] = abs(df["open_to_close_prob_move"])
df["abs_mid_close_move"] = abs(df["mid_to_close_prob_move"])

# Disagreement between model and market
df["model_market_gap"] = abs(df["true_prob_shrunk_40"] - df["market_prob"])

# Market uncertainty score
df["market_uncertainty_score"] = (
    df["abs_open_close_move"].fillna(0) * 2 +
    df["abs_mid_close_move"].fillna(0) +
    df["model_market_gap"].fillna(0)
)

# Buckets
df["uncertainty_bucket"] = pd.cut(
    df["market_uncertainty_score"],
    bins=[-1,0.02,0.05,0.10,1],
    labels=["Low","Medium","High","Extreme"]
)

summary = (
    df.groupby(["Season","uncertainty_bucket"])
      .agg(
        bets=("profit","size"),
        wins=("actual_win","sum"),
        hit_rate=("actual_win","mean"),
        roi=("profit","mean")
      )
      .reset_index()
)

print("\nMarket Uncertainty Summary\n")
print(summary)

df.to_csv("analysis_out/true_line_board_uncertainty.csv", index=False)
summary.to_csv("analysis_out/market_uncertainty_summary.csv", index=False)