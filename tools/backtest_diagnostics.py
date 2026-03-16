from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BETS_PATH = PROJECT_ROOT / "exports" / "backtest_bets_moneyline.csv"
OUT_DIR = PROJECT_ROOT / "exports" / "backtest_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_MIN = 0.09


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Summarize backtest performance by the requested grouping columns.

    Handles:
    - overall summary when group_cols = []
    - categorical groupers like pd.cut() buckets
    - stable sorting after aggregation
    """
    if not group_cols:
        out = pd.DataFrame(
            {
                "bets": [df["game_id"].count()],
                "wins": [df["actual_win"].sum()],
                "profit_units": [df["profit"].sum()],
                "avg_edge": [df["edge"].mean()],
                "avg_model_prob": [df["model_prob"].mean()],
                "avg_implied_prob": [df["implied_prob"].mean()],
                "avg_odds": [df["closing_odds"].mean()],
                "avg_books": [df["books_used"].mean()],
            }
        )
    else:
        out = (
            df.groupby(group_cols, dropna=False, observed=False)
            .agg(
                bets=("game_id", "count"),
                wins=("actual_win", "sum"),
                profit_units=("profit", "sum"),
                avg_edge=("edge", "mean"),
                avg_model_prob=("model_prob", "mean"),
                avg_implied_prob=("implied_prob", "mean"),
                avg_odds=("closing_odds", "mean"),
                avg_books=("books_used", "mean"),
            )
            .reset_index()
        )

    out["win_rate"] = out["wins"] / out["bets"]
    out["roi"] = out["profit_units"] / out["bets"]

    if group_cols:
        out = out.sort_values(group_cols, kind="stable").reset_index(drop=True)

    return out


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["edge_bucket"] = pd.cut(
        out["edge"],
        bins=[0.10, 0.12, 0.15, 0.20, 1.00],
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

    out["books_bucket"] = pd.cut(
        out["books_used"],
        bins=[-np.inf, 3, 5, 8, 12, np.inf],
        labels=["<=3", "4-5", "6-8", "9-12", "13+"],
        right=False,
    )

    out["bet_on_team"] = np.where(
        out["side"].astype(str).str.lower() == "home",
        out["home_team"],
        out["away_team"],
    )

    out["bet_type"] = np.where(
        out["side"].astype(str).str.lower() == "home",
        "bet_home",
        "bet_away",
    )

    return out


def main() -> None:
    print("Loading backtest bets...")
    df = pd.read_csv(BETS_PATH, low_memory=False)

    required = [
        "game_id",
        "Season",
        "Week",
        "game_date",
        "away_team",
        "home_team",
        "side",
        "closing_odds",
        "model_prob",
        "implied_prob",
        "edge",
        "actual_win",
        "profit",
        "books_used",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"backtest_bets_moneyline.csv missing columns: {missing}")

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    df["closing_odds"] = pd.to_numeric(df["closing_odds"], errors="coerce")
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["implied_prob"] = pd.to_numeric(df["implied_prob"], errors="coerce")
    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")
    df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["books_used"] = pd.to_numeric(df["books_used"], errors="coerce")

    df = df[df["edge"] >= EDGE_MIN].copy()
    print(f"Using bets with edge >= {EDGE_MIN:.2f}")
    print(f"Rows after filter: {len(df):,}")

    df = add_buckets(df)

    print("Building summaries...")

    overall = summarize(df, [])
    by_season = summarize(df, ["Season"])
    by_edge_bucket = summarize(df, ["edge_bucket"])
    by_odds_bucket = summarize(df, ["odds_bucket"])
    by_side = summarize(df, ["side"])
    by_bet_type = summarize(df, ["bet_type"])
    by_books_bucket = summarize(df, ["books_bucket"])
    by_season_edge = summarize(df, ["Season", "edge_bucket"])
    by_season_side = summarize(df, ["Season", "side"])

    overall.to_csv(OUT_DIR / "overall_summary.csv", index=False)
    by_season.to_csv(OUT_DIR / "by_season.csv", index=False)
    by_edge_bucket.to_csv(OUT_DIR / "by_edge_bucket.csv", index=False)
    by_odds_bucket.to_csv(OUT_DIR / "by_odds_bucket.csv", index=False)
    by_side.to_csv(OUT_DIR / "by_side.csv", index=False)
    by_bet_type.to_csv(OUT_DIR / "by_bet_type.csv", index=False)
    by_books_bucket.to_csv(OUT_DIR / "by_books_bucket.csv", index=False)
    by_season_edge.to_csv(OUT_DIR / "by_season_edge_bucket.csv", index=False)
    by_season_side.to_csv(OUT_DIR / "by_season_side.csv", index=False)

    print(f"Saved diagnostics to: {OUT_DIR}")

    print("\nOVERALL")
    print(overall.to_string(index=False))

    print("\nBY SEASON")
    print(by_season.to_string(index=False))

    print("\nBY EDGE BUCKET")
    print(by_edge_bucket.to_string(index=False))

    print("\nBY ODDS BUCKET")
    print(by_odds_bucket.to_string(index=False))

    print("\nBY SIDE")
    print(by_side.to_string(index=False))

    print("\nBY BOOKS BUCKET")
    print(by_books_bucket.to_string(index=False))


if __name__ == "__main__":
    main()


