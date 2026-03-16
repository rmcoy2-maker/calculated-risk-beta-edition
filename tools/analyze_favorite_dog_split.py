import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    df["market_prob"] = pd.to_numeric(df["market_prob"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")

    df["side_type"] = np.where(df["market_prob"] > 0.5, "Favorite", "Underdog")

    out = (
        df.groupby(["Season", "side_type"], dropna=False)
        .agg(
            bets=("profit", "size"),
            wins=("actual_win", "sum"),
            hit_rate=("actual_win", "mean"),
            roi=("profit", "mean"),
            avg_market_prob=("market_prob", "mean"),
        )
        .reset_index()
    )

    print(out)
    out.to_csv("analysis_out/favorite_dog_split.csv", index=False)

if __name__ == "__main__":
    main()