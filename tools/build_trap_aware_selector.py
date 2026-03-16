import pandas as pd
import numpy as np

df = pd.read_csv("analysis_out/true_line_board_with_traps.csv", low_memory=False)

num_cols = [
    "Week",
    "market_prob",
    "true_prob_shrunk_40",
    "edge_prob_shrunk_40",
    "model_edge_vs_close",
    "hybrid_score",
    "profit",
    "actual_win",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Base production universe
base = df[
    (df["favorite_trap_flag"] == 0) &
    (df["odds_bucket"].isin(["Medium Favorite", "Light Favorite"])) &
    (df["market_confirmation"].isin(["confirmed", "faded_late"])) &
    (df["edge_prob_shrunk_40"] >= 0.01) &
    (df["edge_prob_shrunk_40"] <= 0.04) &
    (df["model_edge_vs_close"] > 0.01)
].copy()

# Extra late-season caution
base = base[
    (base["Week"] < 9) |
    ((base["Week"] >= 9) & (base["edge_prob_shrunk_40"] >= 0.02))
].copy()

# Top 2 per week
safe = (
    base.sort_values(["Season", "Week", "hybrid_score"], ascending=[True, True, False])
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
            avg_prob=("true_prob_shrunk_40", "mean"),
            avg_market_prob=("market_prob", "mean"),
            avg_edge=("edge_prob_shrunk_40", "mean"),
            avg_close_edge=("model_edge_vs_close", "mean"),
        )
        .reset_index()
)

print("\nTrap-Aware Selector Summary\n")
print(summary.to_string(index=False))

safe.to_csv("analysis_out/trap_aware_selector_bets.csv", index=False)
summary.to_csv("analysis_out/trap_aware_selector_summary.csv", index=False)

print("\nSaved:")
print("analysis_out/trap_aware_selector_bets.csv")
print("analysis_out/trap_aware_selector_summary.csv")