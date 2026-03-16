import pandas as pd
import numpy as np

df = pd.read_csv("analysis_out/true_line/true_line_board.csv", low_memory=False)

for c in ["actual_win","true_prob_shrunk_40","hybrid_score"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["true_prob_shrunk_40","actual_win","hybrid_score"]).copy()

# one side per game
df = (
    df.sort_values("hybrid_score", ascending=False)
      .groupby("game_id")
      .head(1)
)

def shrink(p, factor):
    return 0.5 + factor*(p-0.5)

results = []

for f in [0.50,0.60,0.70,0.80,0.90,1.0]:

    p = shrink(df["true_prob_shrunk_40"], f)

    implied = df["market_prob"]

    edge = p - implied

    bets = df[edge > 0.01]

    summary = bets.groupby("Season").agg(
        bets=("actual_win","size"),
        wins=("actual_win","sum"),
        hit=("actual_win","mean")
    )

    summary["shrink"] = f
    results.append(summary.reset_index())

out = pd.concat(results)

print(out)