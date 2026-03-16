import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, low_memory=False)

for col in ["profit", "actual_win", "edge_prob_shrunk_40", "model_edge_vs_close", "hybrid_score", "Week"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# one side per game first
df = (
    df.sort_values("hybrid_score", ascending=False)
      .groupby("game_id", dropna=False)
      .head(1)
      .copy()
)

# safer production-style rules
safe = df[
    df["odds_bucket"].isin(["Medium Favorite", "Light Favorite"]) &
    df["market_confirmation"].isin(["confirmed", "faded_late"]) &
    (df["model_edge_vs_close"] > 0.01) &
    (df["edge_prob_shrunk_40"] >= 0.01) &
    (df["edge_prob_shrunk_40"] <= 0.04)
].copy()

# stronger governor late season
safe["late_season"] = safe["Week"] >= 9
safe = safe[
    (~safe["late_season"]) |
    ((safe["late_season"]) & (safe["edge_prob_shrunk_40"] >= 0.02))
].copy()

# top 2 per week
safe = (
    safe.sort_values(["Season", "Week", "hybrid_score"], ascending=[True, True, False])
        .groupby(["Season", "Week"], dropna=False)
        .head(2)
        .copy()
)

summary = (
    safe.groupby("Season", dropna=False)
        .agg(
            bets=("profit", "size"),
            wins=("actual_win", "sum"),
            hit_rate=("actual_win", "mean"),
            roi=("profit", "mean"),
            avg_edge_shr40=("edge_prob_shrunk_40", "mean"),
            avg_close_edge=("model_edge_vs_close", "mean"),
        )
        .reset_index()
)

print("\nSafe Selector Summary\n")
print(summary.to_string(index=False))

safe.to_csv("analysis_out/safe_selector_bets.csv", index=False)
summary.to_csv("analysis_out/safe_selector_summary.csv", index=False)

print("\nSaved:")
print("analysis_out/safe_selector_bets.csv")
print("analysis_out/safe_selector_summary.csv")