from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "exports" / "model_probs.csv"
LINES_PATH = PROJECT_ROOT / "exports" / "closing_lines.csv"
GAMES_PATH = PROJECT_ROOT / "exports" / "games_master.csv"

OUTPUT_PATH = PROJECT_ROOT / "exports" / "edges_market_anchor.csv"
SUMMARY_PATH = PROJECT_ROOT / "exports" / "edges_market_anchor_summary.csv"

MODEL_WEIGHT = 0.60
MARKET_WEIGHT = 0.40


def american_to_implied_prob(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(
            odds > 0,
            100 / (odds + 100),
            np.where(odds < 0, (-odds) / ((-odds) + 100), np.nan),
        ),
        index=odds.index,
    )


def american_profit_per_unit(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(
            odds > 0,
            odds / 100.0,
            np.where(odds < 0, 100.0 / (-odds), np.nan),
        ),
        index=odds.index,
    )


def build_consensus_closing_moneyline(closing: pd.DataFrame) -> pd.DataFrame:
    ml = closing.loc[closing["market"].astype(str).str.lower() == "moneyline"].copy()
    ml["side"] = ml["side"].astype(str).str.lower().str.strip()
    ml["odds"] = pd.to_numeric(ml["odds"], errors="coerce")

    ml = ml[
        ml["side"].isin(["home", "away"])
        & ml["odds"].notna()
        & (ml["odds"] != 0)
        & (ml["odds"].abs() >= 100)
        & (ml["odds"].abs() <= 5000)
    ].copy()

    consensus = (
        ml.groupby(["game_id", "side"], as_index=False)
        .agg(
            closing_odds=("odds", "median"),
            books_used=("book_key", "nunique"),
        )
    )

    home = (
        consensus.loc[consensus["side"] == "home", ["game_id", "closing_odds", "books_used"]]
        .rename(columns={"closing_odds": "closing_home_ml", "books_used": "home_books"})
        .copy()
    )

    away = (
        consensus.loc[consensus["side"] == "away", ["game_id", "closing_odds", "books_used"]]
        .rename(columns={"closing_odds": "closing_away_ml", "books_used": "away_books"})
        .copy()
    )

    return home.merge(away, on="game_id", how="inner")


def summarize_thresholds(bets: pd.DataFrame) -> pd.DataFrame:
    thresholds = [0.00, 0.02, 0.03, 0.05, 0.08, 0.10]
    rows = []

    for th in thresholds:
        sub = bets[bets["edge"] >= th].copy()
        n = len(sub)

        rows.append(
            {
                "edge_threshold": th,
                "bets": n,
                "profit_units": float(sub["profit"].sum()) if n else 0.0,
                "roi": float(sub["profit"].sum() / n) if n else np.nan,
                "win_rate": float(sub["actual_win"].mean()) if n else np.nan,
                "avg_edge": float(sub["edge"].mean()) if n else np.nan,
                "avg_model_prob_raw": float(sub["model_prob_raw"].mean()) if n else np.nan,
                "avg_model_prob_adj": float(sub["model_prob_adj"].mean()) if n else np.nan,
                "avg_implied_prob": float(sub["implied_prob"].mean()) if n else np.nan,
                "avg_odds": float(sub["closing_odds"].mean()) if n else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("Loading model probabilities...")
    model = pd.read_csv(MODEL_PATH, low_memory=False)

    print("Loading closing lines...")
    closing = pd.read_csv(LINES_PATH, low_memory=False)

    print("Loading game results...")
    games = pd.read_csv(GAMES_PATH, low_memory=False)

    needed_model = ["game_id", "Season", "Week", "game_date", "away_team", "home_team", "p_home_win", "p_away_win"]
    missing_model = [c for c in needed_model if c not in model.columns]
    if missing_model:
        raise ValueError(f"model_probs.csv missing columns: {missing_model}")

    needed_games = ["game_id", "home_win", "away_win"]
    missing_games = [c for c in needed_games if c not in games.columns]
    if missing_games:
        raise ValueError(f"games_master.csv missing columns: {missing_games}")

    print("Building consensus closing moneylines...")
    cl = build_consensus_closing_moneyline(closing)

    df = model[needed_model].merge(cl, on="game_id", how="inner")
    df = df.merge(games[needed_games], on="game_id", how="left")

    df["home_implied_prob"] = american_to_implied_prob(df["closing_home_ml"])
    df["away_implied_prob"] = american_to_implied_prob(df["closing_away_ml"])

    df["home_adj_prob"] = (MODEL_WEIGHT * pd.to_numeric(df["p_home_win"], errors="coerce")) + (
        MARKET_WEIGHT * df["home_implied_prob"]
    )
    df["away_adj_prob"] = (MODEL_WEIGHT * pd.to_numeric(df["p_away_win"], errors="coerce")) + (
        MARKET_WEIGHT * df["away_implied_prob"]
    )

    home = df.copy()
    home["side"] = "home"
    home["closing_odds"] = pd.to_numeric(home["closing_home_ml"], errors="coerce")
    home["model_prob_raw"] = pd.to_numeric(home["p_home_win"], errors="coerce")
    home["model_prob_adj"] = pd.to_numeric(home["home_adj_prob"], errors="coerce")
    home["implied_prob"] = pd.to_numeric(home["home_implied_prob"], errors="coerce")
    home["actual_win"] = pd.to_numeric(home["home_win"], errors="coerce")
    home["books_used"] = pd.to_numeric(home["home_books"], errors="coerce")

    away = df.copy()
    away["side"] = "away"
    away["closing_odds"] = pd.to_numeric(away["closing_away_ml"], errors="coerce")
    away["model_prob_raw"] = pd.to_numeric(away["p_away_win"], errors="coerce")
    away["model_prob_adj"] = pd.to_numeric(away["away_adj_prob"], errors="coerce")
    away["implied_prob"] = pd.to_numeric(away["away_implied_prob"], errors="coerce")
    away["actual_win"] = pd.to_numeric(away["away_win"], errors="coerce")
    away["books_used"] = pd.to_numeric(away["away_books"], errors="coerce")

    bets = pd.concat([home, away], ignore_index=True)

    bets = bets[
        bets["closing_odds"].notna()
        & (bets["closing_odds"].abs() >= 100)
        & (bets["closing_odds"].abs() <= 5000)
        & bets["model_prob_raw"].between(0.001, 0.999)
        & bets["model_prob_adj"].between(0.001, 0.999)
        & bets["actual_win"].isin([0, 1])
    ].copy()

    bets["edge_raw"] = bets["model_prob_raw"] - bets["implied_prob"]
    bets["edge"] = bets["model_prob_adj"] - bets["implied_prob"]
    bets["profit_per_unit_win"] = american_profit_per_unit(bets["closing_odds"])
    bets["profit"] = np.where(
        bets["actual_win"] == 1,
        bets["profit_per_unit_win"],
        -1.0,
    )

    keep_cols = [
        "game_id",
        "Season",
        "Week",
        "game_date",
        "away_team",
        "home_team",
        "side",
        "closing_odds",
        "model_prob_raw",
        "model_prob_adj",
        "implied_prob",
        "edge_raw",
        "edge",
        "actual_win",
        "profit",
        "books_used",
    ]
    bets = bets[keep_cols].copy()
    bets = bets.sort_values(["edge", "game_date"], ascending=[False, True], kind="stable").reset_index(drop=True)

    summary = summarize_thresholds(bets)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bets.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"\nSaved adjusted edges: {OUTPUT_PATH}")
    print(f"Rows: {len(bets):,}")
    print(f"Saved summary: {SUMMARY_PATH}")

    print("\nMarket-anchored summary:")
    print(summary.to_string(index=False))

    print("\nTop 20 adjusted edges:")
    print(bets.head(20).to_string(index=False))


if __name__ == "__main__":
    main()