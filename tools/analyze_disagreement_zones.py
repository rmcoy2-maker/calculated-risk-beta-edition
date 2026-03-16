import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, low_memory=False)

df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")
df["edge_prob_shrunk_40"] = pd.to_numeric(df["edge_prob_shrunk_40"], errors="coerce")
df["model_edge_vs_close"] = pd.to_numeric(df["model_edge_vs_close"], errors="coerce")
df["hybrid_score"] = pd.to_numeric(df["hybrid_score"], errors="coerce")

# enforce one side per game
df = (
    df.sort_values("hybrid_score", ascending=False)
      .groupby("game_id", dropna=False)
      .head(1)
      .copy()
)

df["shr40_zone"] = pd.cut(
    df["edge_prob_shrunk_40"],
    bins=[-999, 0.0, 0.01, 0.02, 0.04, 999],
    labels=["<0", "0-0.01", "0.01-0.02", "0.02-0.04", ">0.04"],
    include_lowest=True
)

df["close_edge_zone"] = pd.cut(
    df["model_edge_vs_close"],
    bins=[-999, 0.0, 0.01, 0.02, 0.04, 999],
    labels=["<0", "0-0.01", "0.01-0.02", "0.02-0.04", ">0.04"],
    include_lowest=True
)

out1 = (
    df.groupby(["Season", "shr40_zone"], dropna=False)
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

out2 = (
    df.groupby(["Season", "close_edge_zone"], dropna=False)
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

print("\nShrunk 40 Disagreement Zones\n")
print(out1.to_string(index=False))

print("\nClose Edge Zones\n")
print(out2.to_string(index=False))

out1.to_csv("analysis_out/disagreement_zone_shr40.csv", index=False)
out2.to_csv("analysis_out/disagreement_zone_close.csv", index=False)
print("\nSaved:")
print("analysis_out/disagreement_zone_shr40.csv")
print("analysis_out/disagreement_zone_close.csv")