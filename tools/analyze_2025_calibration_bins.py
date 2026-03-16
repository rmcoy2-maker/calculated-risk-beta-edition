import pandas as pd

df = pd.read_csv("analysis_out/true_line/true_line_board.csv", low_memory=False)

df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")
df["true_prob_shrunk_40"] = pd.to_numeric(df["true_prob_shrunk_40"], errors="coerce")

# clean + one side per game
df = df.dropna(subset=["true_prob_shrunk_40", "actual_win", "hybrid_score"]).copy()
df = (
    df.sort_values("hybrid_score", ascending=False)
      .groupby("game_id", dropna=False)
      .head(1)
      .copy()
)

df = df[df["Season"] == 2025].copy()

df["prob_bin"] = pd.cut(
    df["true_prob_shrunk_40"],
    bins=[0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0],
    include_lowest=True
)

out = (
    df.groupby("prob_bin", dropna=False)
      .agg(
          bets=("actual_win", "size"),
          wins=("actual_win", "sum"),
          hit_rate=("actual_win", "mean"),
          avg_prob=("true_prob_shrunk_40", "mean"),
      )
      .reset_index()
)

print("\n2025 calibration bins\n")
print(out.to_string(index=False))

out.to_csv("analysis_out/2025_calibration_bins.csv", index=False)
print("\nSaved: analysis_out/2025_calibration_bins.csv")