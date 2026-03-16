import pandas as pd
import numpy as np

df = pd.read_csv("analysis_out/true_line_board_with_traps.csv", low_memory=False)

for c in ["profit", "actual_win", "Week", "true_prob_shrunk_40", "market_prob"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

trap_cols = [c for c in df.columns if c.startswith("trap_flag_")]

rows = []
for col in trap_cols:
    g = (
        df.groupby(["Season", col], dropna=False)
          .agg(
              bets=("profit", "size"),
              wins=("actual_win", "sum"),
              hit_rate=("actual_win", "mean"),
              roi=("profit", "mean"),
              avg_prob=("true_prob_shrunk_40", "mean"),
              avg_market_prob=("market_prob", "mean"),
          )
          .reset_index()
    )
    g["trap_type"] = col
    rows.append(g)

out = pd.concat(rows, ignore_index=True)

print("\nTrap Flag Performance\n")
print(out.to_string(index=False))

out.to_csv("analysis_out/trap_flag_performance.csv", index=False)
print("\nSaved: analysis_out/trap_flag_performance.csv")