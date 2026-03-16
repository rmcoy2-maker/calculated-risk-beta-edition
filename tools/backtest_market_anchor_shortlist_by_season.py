from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "exports" / "edges_market_anchor.csv"
OUT_SUMMARY_PATH = PROJECT_ROOT / "exports" / "market_anchor_shortlist_by_season_summary.csv"
OUT_BETS_PATH = PROJECT_ROOT / "exports" / "market_anchor_shortlist_by_season_bets.csv"

EDGE_THRESHOLD = 0.05
MAX_ODDS = 150


def summarize(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    return {
        "subset": label,
        "bets": n,
        "wins": float(df["actual_win"].sum()) if n else 0.0,
        "win_rate": float(df["actual_win"].mean()) if n else np.nan,
        "profit_units": float(df["profit"].sum()) if n else 0.0,
        "roi": float(df["profit"].sum() / n) if n else np.nan,
        "avg_edge": float(df["edge"].mean()) if n else np.nan,
        "avg_edge_raw": float(df["edge_raw"].mean()) if n else np.nan,
        "avg_model_prob_raw": float(df["model_prob_raw"].mean()) if n else np.nan,
        "avg_model_prob_adj": float(df["model_prob_adj"].mean()) if n else np.nan,
        "avg_implied_prob": float(df["implied_prob"].mean()) if n else np.nan,
        "avg_odds": float(df["closing_odds"].mean()) if n else np.nan,
        "avg_books_used": float(df["books_used"].mean()) if n else np.nan,
    }


def main() -> None:
    print("Loading market-anchored edges...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    needed = [
        "game_id", "Season", "Week", "game_date",
        "away_team", "home_team", "side", "closing_odds",
        "model_prob_raw", "model_prob_adj", "implied_prob",
        "edge_raw", "edge", "actual_win", "profit", "books_used"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"edges_market_anchor.csv missing columns: {missing}")

    num_cols = [
        "Season", "Week", "closing_odds", "model_prob_raw", "model_prob_adj",
        "implied_prob", "edge_raw", "edge", "actual_win", "profit", "books_used"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["favorite_or_dog"] = np.where(df["closing_odds"] < 0, "favorite", "underdog")
    df["bet_team"] = np.where(
        df["side"].astype(str).str.lower() == "home",
        df["home_team"],
        df["away_team"],
    )

    shortlist = df[
        df["closing_odds"].notna()
        & (df["favorite_or_dog"] == "favorite")
        & (df["edge"] >= EDGE_THRESHOLD)
        & (df["closing_odds"] <= MAX_ODDS)
    ].copy()

    shortlist = shortlist.sort_values(
        ["Season", "Week", "edge", "model_prob_adj", "books_used"],
        ascending=[True, True, False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    summary_rows = []
    summary_rows.append(summarize(shortlist, "all_shortlist"))

    seasons = sorted([int(s) for s in shortlist["Season"].dropna().unique()])
    for season in seasons:
        sub = shortlist[shortlist["Season"] == season].copy()
        summary_rows.append(summarize(sub, f"season_{season}"))

    by_week_rows = []
    for season in seasons:
        sub_season = shortlist[shortlist["Season"] == season].copy()
        weeks = sorted([int(w) for w in sub_season["Week"].dropna().unique()])
        for week in weeks:
            sub_week = sub_season[sub_season["Week"] == week].copy()
            row = summarize(sub_week, f"season_{season}_week_{week}")
            row["Season"] = season
            row["Week"] = week
            by_week_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    by_week = pd.DataFrame(by_week_rows)

    keep_cols = [
        "game_id", "Season", "Week", "game_date",
        "away_team", "home_team", "bet_team", "side",
        "closing_odds", "model_prob_raw", "model_prob_adj",
        "implied_prob", "edge_raw", "edge",
        "books_used", "actual_win", "profit"
    ]
    shortlist = shortlist[keep_cols].copy()

    OUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY_PATH, index=False)
    shortlist.to_csv(OUT_BETS_PATH, index=False)

    by_week_path = OUT_SUMMARY_PATH.with_name("market_anchor_shortlist_by_season_week_summary.csv")
    by_week.to_csv(by_week_path, index=False)

    print(f"\nSaved summary: {OUT_SUMMARY_PATH}")
    print(f"Saved bets: {OUT_BETS_PATH}")
    print(f"Saved week summary: {by_week_path}")

    print("\nOVERALL + BY SEASON")
    print(summary.to_string(index=False))

    print("\nBY WEEK (top 40 rows)")
    if len(by_week):
        print(by_week.head(40).to_string(index=False))

    print("\nTOP 30 SHORTLIST BETS")
    print(shortlist.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
