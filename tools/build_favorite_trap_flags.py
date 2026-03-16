import pandas as pd
import numpy as np

df = pd.read_csv("analysis_out/true_line/true_line_board.csv", low_memory=False)

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

# Keep only real market-joined rows
df = df.dropna(
    subset=[
        "market_prob",
        "true_prob_shrunk_40",
        "edge_prob_shrunk_40",
        "model_edge_vs_close",
        "hybrid_score",
    ]
).copy()

# One side per game
df = (
    df.sort_values("hybrid_score", ascending=False)
      .groupby("game_id", dropna=False)
      .head(1)
      .copy()
)

df["is_favorite"] = df["market_prob"] > 0.5

df["trap_flag_inflated_favorite"] = (
    df["is_favorite"] &
    (df["true_prob_shrunk_40"] >= 0.70) &
    (df["market_confirmation"] != "confirmed")
).astype(int)

df["trap_flag_extreme_disagreement"] = (
    df["is_favorite"] &
    (df["edge_prob_shrunk_40"] > 0.04)
).astype(int)

df["trap_flag_late_favorite"] = (
    df["is_favorite"] &
    (df["Week"] >= 9) &
    (df["true_prob_shrunk_40"] >= 0.65)
).astype(int)

df["trap_flag_weak_close_support"] = (
    df["is_favorite"] &
    (df["true_prob_shrunk_40"] >= 0.65) &
    (df["model_edge_vs_close"] <= 0.01)
).astype(int)

df["trap_flag_unknown_context"] = (
    df["odds_bucket"].fillna("Unknown").eq("Unknown")
).astype(int)

trap_cols = [c for c in df.columns if c.startswith("trap_flag_")]
df["trap_count"] = df[trap_cols].sum(axis=1)

df["favorite_trap_flag"] = (df["trap_count"] >= 1).astype(int)
df["favorite_trap_heavy"] = (df["trap_count"] >= 2).astype(int)

summary = (
    df.groupby(["Season", "favorite_trap_flag"], dropna=False)
      .agg(
          bets=("profit", "size"),
          wins=("actual_win", "sum"),
          hit_rate=("actual_win", "mean"),
          roi=("profit", "mean"),
          avg_prob=("true_prob_shrunk_40", "mean"),
          avg_market_prob=("market_prob", "mean"),
          avg_edge=("edge_prob_shrunk_40", "mean"),
      )
      .reset_index()
)

print("\nFavorite Trap Summary\n")
print(summary.to_string(index=False))

df.to_csv("analysis_out/true_line_board_with_traps.csv", index=False)
summary.to_csv("analysis_out/favorite_trap_summary.csv", index=False)

print("\nSaved:")
print("analysis_out/true_line_board_with_traps.csv")
print("analysis_out/favorite_trap_summary.csv")