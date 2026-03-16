from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BETS_PATH = PROJECT_ROOT / "exports" / "backtest_bets_moneyline.csv"
OUT_DIR = PROJECT_ROOT / "exports" / "backtest_2025_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_MIN = 0.09


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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
            df.groupby(group_cols, observed=True, dropna=False)
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
    return out


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["edge_bucket"] = pd.cut(
        out["edge"],
        bins=[0.09, 0.10, 0.12, 0.15, 0.20, 1.00],
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

    out["favorite_or_dog"] = np.where(
        out["closing_odds"] < 0,
        "favorite",
        "underdog",
    )

    return out


def save_and_print(name: str, df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)
    print(f"\n{name}")
    print(df.to_string(index=False))


def main() -> None:
    print("Loading bets...")
    print(f"Reading: {BETS_PATH}")

    df = pd.read_csv(BETS_PATH, low_memory=False)

    print(f"Rows loaded: {len(df):,}")

    required = [
        "game_id",
        "Season",
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

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df["closing_odds"] = pd.to_numeric(df["closing_odds"], errors="coerce")
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["implied_prob"] = pd.to_numeric(df["implied_prob"], errors="coerce")
    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")
    df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["books_used"] = pd.to_numeric(df["books_used"], errors="coerce")

    df = df[df["edge"] >= EDGE_MIN].copy()
    print(f"Rows after edge >= {EDGE_MIN:.2f}: {len(df):,}")

    df = df[df["Season"] == 2025].copy()
    print(f"Rows in 2025 sample: {len(df):,}")

    if df.empty:
        print("No 2025 rows found after filtering.")
        return

    df = add_buckets(df)

    print("Building summaries...")

    overall = summarize(df, [])
    by_favdog = summarize(df, ["favorite_or_dog"])
    by_odds = summarize(df, ["odds_bucket"])
    by_edge = summarize(df, ["edge_bucket"])
    by_side = summarize(df, ["side"])

    # Narrow middle band: -150 to +110
    df_mid_tight = df[
        (df["closing_odds"] >= -150) &
        (df["closing_odds"] < 110)
    ].copy()

    mid_tight_overall = summarize(df_mid_tight, [])
    mid_tight_by_edge = summarize(df_mid_tight, ["edge_bucket"])
    mid_tight_by_side = summarize(df_mid_tight, ["side"])

    # Wider middle band: -150 to +150
    df_mid_wide = df[
        (df["closing_odds"] >= -150) &
        (df["closing_odds"] < 150)
    ].copy()

    mid_wide_overall = summarize(df_mid_wide, [])
    mid_wide_by_edge = summarize(df_mid_wide, ["edge_bucket"])
    mid_wide_by_side = summarize(df_mid_wide, ["side"])

    save_and_print("OVERALL 2025", overall, OUT_DIR / "overall_2025.csv")
    save_and_print("BY FAVORITE / DOG", by_favdog, OUT_DIR / "by_favorite_or_dog_2025.csv")
    save_and_print("BY ODDS BUCKET", by_odds, OUT_DIR / "by_odds_2025.csv")
    save_and_print("BY EDGE BUCKET", by_edge, OUT_DIR / "by_edge_2025.csv")
    save_and_print("BY SIDE", by_side, OUT_DIR / "by_side_2025.csv")

    save_and_print(
        "MIDDLE BAND TIGHT (-150 to +110) OVERALL",
        mid_tight_overall,
        OUT_DIR / "mid_tight_overall_2025.csv",
    )
    save_and_print(
        "MIDDLE BAND TIGHT (-150 to +110) BY EDGE",
        mid_tight_by_edge,
        OUT_DIR / "mid_tight_by_edge_2025.csv",
    )
    save_and_print(
        "MIDDLE BAND TIGHT (-150 to +110) BY SIDE",
        mid_tight_by_side,
        OUT_DIR / "mid_tight_by_side_2025.csv",
    )

    save_and_print(
        "MIDDLE BAND WIDE (-150 to +150) OVERALL",
        mid_wide_overall,
        OUT_DIR / "mid_wide_overall_2025.csv",
    )
    save_and_print(
        "MIDDLE BAND WIDE (-150 to +150) BY EDGE",
        mid_wide_by_edge,
        OUT_DIR / "mid_wide_by_edge_2025.csv",
    )
    save_and_print(
        "MIDDLE BAND WIDE (-150 to +150) BY SIDE",
        mid_wide_by_side,
        OUT_DIR / "mid_wide_by_side_2025.csv",
    )

    print(f"\nSaved diagnostics to: {OUT_DIR}")


if __name__ == "__main__":
    main()
