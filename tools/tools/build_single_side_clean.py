import pandas as pd

df = pd.read_csv("analysis_out/true_line/true_line_board.csv", low_memory=False)

for col in ["profit", "actual_win", "hybrid_score", "edge_prob_shrunk_40", "model_edge_vs_close", "market_prob"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows missing critical ranking / market fields
df = df.dropna(subset=["hybrid_score", "market_prob", "edge_prob_shrunk_40", "model_edge_vs_close"]).copy()

best = (
    df.sort_values("hybrid_score", ascending=False)
      .groupby("game_id", dropna=False)
      .head(1)
      .copy()
)

summary = (
    best.groupby("Season", dropna=False)
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

print("\nClean single-side summary\n")
print(summary.to_string(index=False))

best.to_csv("analysis_out/single_side_clean_board.csv", index=False)
summary.to_csv("analysis_out/single_side_clean_summary.csv", index=False)

print("\nSaved:")
print("analysis_out/single_side_clean_board.csv")
print("analysis_out/single_side_clean_summary.csv")