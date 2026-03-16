import pandas as pd
import numpy as np

scores = pd.read_csv("2017-2025_scores.csv")

# Determine winner
scores["home_win"] = scores["HomeScore"] > scores["AwayScore"]

# If spread exists use it to determine favorite
if "Spread" in scores.columns:
    scores["favorite"] = np.where(scores["Spread"] < 0, "home", "away")
else:
    raise ValueError("Spread column required")

scores["favorite_win"] = np.where(
    (scores["favorite"] == "home") & (scores["home_win"]),
    1,
    np.where(
        (scores["favorite"] == "away") & (~scores["home_win"]),
        1,
        0
    )
)

summary = (
    scores.groupby("Season")
    .agg(
        games=("favorite_win","size"),
        fav_wins=("favorite_win","sum")
    )
)

summary["favorite_win_pct"] = summary["fav_wins"] / summary["games"]

print("\nLeague Favorite Win Rate\n")
print(summary)

summary.to_csv("analysis_out/league_favorite_win_rate.csv")