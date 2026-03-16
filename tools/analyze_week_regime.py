import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input)

df["market_prob"] = pd.to_numeric(df["market_prob"], errors="coerce")
df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")

df["side"] = np.where(df["market_prob"] > 0.5, "Favorite", "Underdog")

df["week_bucket"] = pd.cut(
    df["Week"],
    bins=[0,3,8,18],
    labels=["Weeks1_3","Weeks4_8","Weeks9_plus"]
)

out = (
    df.groupby(["Season","week_bucket","side"])
    .agg(
        bets=("actual_win","size"),
        wins=("actual_win","sum"),
        hit_rate=("actual_win","mean")
    )
    .reset_index()
)

print(out)