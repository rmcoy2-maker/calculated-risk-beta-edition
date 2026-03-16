from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BETS_PATH = PROJECT_ROOT / "exports" / "backtest_bets_moneyline.csv"
OUT_DIR = PROJECT_ROOT / "exports" / "backtest_2025_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_MIN = 0.09


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        out = pd.DataFrame(
            {
                "bets": [df["game_id"].count()],
                "wins": [df["actual_win"].sum()],
                "profit_units": [df["profit"].sum()],
                "avg_model_prob": [df["model_prob"].mean()],
                "avg_implied_prob": [df["implied_prob"].mean()],
                "avg_odds": [df["closing_odds"].mean()],
            }
        )
    else:
        out = (
            df.groupby(group_cols, observed=True, dropna=False)
            .agg(
                bets=("game_id", "count"),
                wins=("actual_win", "sum"),
                profit_units=("profit", "sum"),
                avg_model_prob=("model_prob", "mean"),
                avg_implied_prob=("implied_prob", "mean"),
                avg_odds=("closing_odds", "mean"),
            )
            .reset_index()
        )

    out["win_rate"] = out["wins"] / out["bets"]
    out["roi"] = out["profit_units"] / out["bets"]
    out["calibration_gap"] = out["win_rate"] - out["avg_model_prob"]
    return out


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["model_prob_bucket"] = pd.cut(
        out["model_prob"],
        bins=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00],
        right=False,
        include_lowest=True,
    )

    out["odds_bucket"] = pd.cut(
        out["closing_odds"],
        bins=[-np.inf, -300, -200, -150, -110, 110, 150, 200, 300, np.inf],
        labels=[
            "<=-300",
            "-300 to -200",
            "-200 to -150",
            "-150 to -110",
            "-110 to +110",
            "+110 to +150",
            "+150 to +200",
            "+200 to +300",
            ">=+300",
        ],
        right=False,
    )

    return out


def main() -> None:
    print("Loading bets...")
    df = pd.read_csv(BETS_PATH, low_memory=False)

    required = [
        "game_id",
        "Season",
        "closing_odds",
        "model_prob",
        "implied_prob",
        "edge",
        "actual_win",
        "profit",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"backtest_bets_moneyline.csv missing columns: {missing}")

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df["closing_odds"] = pd.to_numeric(df["closing_odds"], errors="coerce")
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["implied_prob"] = pd.to_numeric(df["implied_prob"], errors="coerce")
    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")
    df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

    df = df[df["edge"] >= EDGE_MIN].copy()
    df = df[df["Season"] == 2025].copy()

    print(f"Rows in 2025 sample after edge filter: {len(df):,}")

    if df.empty:
        print("No 2025 rows found after filtering.")
        return

    df = add_buckets(df)

    overall = summarize(df, [])
    by_model_prob = summarize(df, ["model_prob_bucket"])
    by_odds = summarize(df, ["odds_bucket"])
    by_model_prob_odds = summarize(df, ["model_prob_bucket", "odds_bucket"])

    overall.to_csv(OUT_DIR / "overall_2025.csv", index=False)
    by_model_prob.to_csv(OUT_DIR / "by_model_prob_bucket_2025.csv", index=False)
    by_odds.to_csv(OUT_DIR / "by_odds_bucket_2025.csv", index=False)
    by_model_prob_odds.to_csv(OUT_DIR / "by_model_prob_bucket_and_odds_2025.csv", index=False)

    print("\nOVERALL 2025")
    print(overall.to_string(index=False))

    print("\nBY MODEL PROB BUCKET")
    print(by_model_prob.to_string(index=False))

    print("\nBY ODDS BUCKET")
    print(by_odds.to_string(index=False))

    print("\nBY MODEL PROB BUCKET AND ODDS")
    print(by_model_prob_odds.to_string(index=False))

    print(f"\nSaved calibration diagnostics to: {OUT_DIR}")


if __name__ == "__main__":
    main()
