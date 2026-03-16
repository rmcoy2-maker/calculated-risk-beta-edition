from pathlib import Path
import argparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PROBS_PATH = PROJECT_ROOT / "exports" / "model_probs.csv"
DEFAULT_CLOSING_LINES_PATH = PROJECT_ROOT / "exports" / "closing_lines.csv"
DEFAULT_GAMES_PATH = PROJECT_ROOT / "exports" / "games_master.csv"
DEFAULT_SUMMARY_OUT_PATH = PROJECT_ROOT / "exports" / "backtest_summary_moneyline.csv"
DEFAULT_BETS_OUT_PATH = PROJECT_ROOT / "exports" / "backtest_bets_moneyline.csv"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-probs-path",
        type=Path,
        default=DEFAULT_MODEL_PROBS_PATH,
    )

    parser.add_argument(
        "--closing-lines-path",
        type=Path,
        default=DEFAULT_CLOSING_LINES_PATH,
    )

    parser.add_argument(
        "--games-path",
        type=Path,
        default=DEFAULT_GAMES_PATH,
    )

    parser.add_argument(
        "--summary-out-path",
        type=Path,
        default=DEFAULT_SUMMARY_OUT_PATH,
    )

    parser.add_argument(
        "--bets-out-path",
        type=Path,
        default=DEFAULT_BETS_OUT_PATH,
    )

    return parser.parse_args()


def american_to_implied(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(
        odds > 0,
        100 / (odds + 100),
        np.where(odds < 0, (-odds) / ((-odds) + 100), np.nan),
    )


def profit_per_unit(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(
        odds > 0,
        odds / 100.0,
        np.where(odds < 0, 100.0 / (-odds), np.nan),
    )


def build_consensus_closing_moneylines(closing):
    ml = closing[closing["market"].astype(str).str.lower() == "moneyline"].copy()
    ml["side"] = ml["side"].astype(str).str.lower().str.strip()
    ml["odds"] = pd.to_numeric(ml["odds"], errors="coerce")

    ml = ml[
        ml["side"].isin(["home", "away"])
        & ml["odds"].notna()
        & (ml["odds"] != 0)
        & (ml["odds"].abs() >= 100)
        & (ml["odds"].abs() <= 5000)
    ].copy()

    grouped = (
        ml.groupby(["game_id", "side"], as_index=False)
        .agg(
            closing_odds=("odds", "median"),
            books_used=("book_key", "nunique"),
        )
    )

    home = (
        grouped[grouped["side"] == "home"][["game_id", "closing_odds", "books_used"]]
        .rename(columns={"closing_odds": "home_closing_odds", "books_used": "home_books"})
        .copy()
    )

    away = (
        grouped[grouped["side"] == "away"][["game_id", "closing_odds", "books_used"]]
        .rename(columns={"closing_odds": "away_closing_odds", "books_used": "away_books"})
        .copy()
    )

    return home.merge(away, on="game_id", how="inner")


def summarize_thresholds(bets):
    thresholds = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.10, 0.12, 0.15]
    rows = []

    for th in thresholds:
        sub = bets[bets["edge"] >= th].copy()
        n = len(sub)

        rows.append(
            {
                "edge_threshold": th,
                "bets": int(n),
                "profit_units": float(sub["profit"].sum()) if n else 0.0,
                "roi": float(sub["profit"].sum() / n) if n else np.nan,
                "win_rate": float(sub["actual_win"].mean()) if n else np.nan,
                "avg_edge": float(sub["edge"].mean()) if n else np.nan,
                "avg_model_prob": float(sub["model_prob"].mean()) if n else np.nan,
                "avg_implied_prob": float(sub["implied_prob"].mean()) if n else np.nan,
                "avg_odds": float(sub["closing_odds"].mean()) if n else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    model_probs_path = args.model_probs_path
    closing_lines_path = args.closing_lines_path
    games_path = args.games_path
    summary_out_path = args.summary_out_path
    bets_out_path = args.bets_out_path

    print("Loading model probabilities...")
    probs = pd.read_csv(model_probs_path, low_memory=False)

    print("Loading closing lines...")
    closing = pd.read_csv(closing_lines_path, low_memory=False)

    print("Loading game results...")
    games = pd.read_csv(games_path, low_memory=False)

    needed_probs = [
        "game_id",
        "Season",
        "Week",
        "game_date",
        "away_team",
        "home_team",
        "p_home_win",
        "p_away_win",
    ]
    missing_probs = [c for c in needed_probs if c not in probs.columns]
    if missing_probs:
        raise ValueError(f"model probs missing columns: {missing_probs}")

    needed_games = ["game_id", "home_win", "away_win"]
    missing_games = [c for c in needed_games if c not in games.columns]
    if missing_games:
        raise ValueError(f"games file missing columns: {missing_games}")

    print("Building consensus closing moneylines...")
    consensus = build_consensus_closing_moneylines(closing)

    df = probs[needed_probs].merge(consensus, on="game_id", how="inner")
    df = df.merge(games[needed_games], on="game_id", how="left")

    home = df.copy()
    home["side"] = "home"
    home["closing_odds"] = pd.to_numeric(home["home_closing_odds"], errors="coerce")
    home["model_prob"] = pd.to_numeric(home["p_home_win"], errors="coerce")
    home["implied_prob"] = american_to_implied(home["closing_odds"])
    home["actual_win"] = pd.to_numeric(home["home_win"], errors="coerce")
    home["books_used"] = pd.to_numeric(home["home_books"], errors="coerce")

    away = df.copy()
    away["side"] = "away"
    away["closing_odds"] = pd.to_numeric(away["away_closing_odds"], errors="coerce")
    away["model_prob"] = pd.to_numeric(away["p_away_win"], errors="coerce")
    away["implied_prob"] = american_to_implied(away["closing_odds"])
    away["actual_win"] = pd.to_numeric(away["away_win"], errors="coerce")
    away["books_used"] = pd.to_numeric(away["away_books"], errors="coerce")

    bets = pd.concat([home, away], ignore_index=True)

    bets = bets[
        bets["closing_odds"].notna()
        & (bets["closing_odds"].abs() >= 100)
        & (bets["closing_odds"].abs() <= 5000)
        & bets["model_prob"].between(0.001, 0.999)
        & bets["actual_win"].isin([0, 1])
    ].copy()

    bets["edge"] = bets["model_prob"] - bets["implied_prob"]
    bets["profit_if_win"] = profit_per_unit(bets["closing_odds"])
    bets["profit"] = np.where(bets["actual_win"] == 1, bets["profit_if_win"], -1.0)

    print("\nOdds filter:")
    print("  closing_odds >= -5000")
    print("  closing_odds <  300")

    bets = bets[bets["closing_odds"] < 300].copy()

    keep_cols = [
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
    bets = bets[keep_cols].copy()
    bets = bets.sort_values(["edge", "game_date"], ascending=[False, True], kind="stable").reset_index(drop=True)

    summary = summarize_thresholds(bets)

    bets_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)

    bets.to_csv(bets_out_path, index=False)
    summary.to_csv(summary_out_path, index=False)

    print(f"\nSaved bets: {bets_out_path}")
    print(f"Rows after odds filter: {len(bets):,}")
    print(f"Saved summary: {summary_out_path}")

    print("\nBacktest summary:")
    print(summary.to_string(index=False))

    print("\nTop 20 edges after odds filter:")
    print(bets.head(20).to_string(index=False))


if __name__ == "__main__":
    main()