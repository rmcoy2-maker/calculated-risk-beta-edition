import pandas as pd

df = pd.read_csv("analysis_out/true_line/true_line_board.csv", low_memory=False)

df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")

# choose best side per game
best = (
    df.sort_values("hybrid_score", ascending=False)
      .groupby("game_id")
      .head(1)
)

print("Rows after single-side filter:", len(best))

summary = (
    best.groupby("Season")
        .agg(
            bets=("profit","size"),
            wins=("actual_win","sum"),
            hit_rate=("actual_win","mean"),
            roi=("profit","mean")
        )
)

print("\nPerformance after one-side-per-game filter\n")
print(summary)

best.to_csv("analysis_out/single_side_board.csv", index=False)
summary.to_csv("analysis_out/single_side_performance.csv")