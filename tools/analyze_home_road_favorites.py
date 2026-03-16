import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, low_memory=False)

df["market_prob"] = pd.to_numeric(df["market_prob"], errors="coerce")
df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")

# determine favorite
df["is_favorite"] = df["market_prob"] > 0.5

# determine home vs away side
# assuming column names HomeTeam and AwayTeam exist
# and a column "Team" identifies the bet side
if "Team" in df.columns:
    df["home_away"] = np.where(df["Team"] == df["HomeTeam"], "Home", "Road")
else:
    # fallback: assume favorite is home if market_prob > opponent
    df["home_away"] = "Unknown"

df["week_bucket"] = pd.cut(
    df["Week"],
    bins=[0,3,8,18],
    labels=["Weeks1_3","Weeks4_8","Weeks9_plus"]
)

favorites = df[df["is_favorite"]]

out = (
    favorites
    .groupby(["Season","week_bucket","home_away"])
    .agg(
        bets=("actual_win","size"),
        wins=("actual_win","sum"),
        hit_rate=("actual_win","mean")
    )
    .reset_index()
)

print("\nFavorite Performance: Home vs Road\n")
print(out)