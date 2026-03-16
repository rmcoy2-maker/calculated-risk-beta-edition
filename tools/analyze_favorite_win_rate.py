import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def american_to_implied_prob(odds):
    if pd.isna(odds):
        return np.nan
    o = float(odds)
    if o > 0:
        return 100 / (o + 100)
    else:
        return abs(o) / (abs(o) + 100)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    # determine market favorite
    df["market_prob"] = df["market_odds"].apply(american_to_implied_prob)

    # favorite if probability > .5
    df["is_favorite"] = df["market_prob"] > 0.5

    df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")

    summary = (
        df[df["is_favorite"]]
        .groupby("Season")
        .agg(
            games=("actual_win", "size"),
            wins=("actual_win", "sum"),
        )
    )

    summary["favorite_win_pct"] = summary["wins"] / summary["games"]

    print("\nFavorite Win Rates\n")
    print(summary)

    summary.to_csv("analysis_out/favorite_win_rate_by_season.csv")


if __name__ == "__main__":
    main()