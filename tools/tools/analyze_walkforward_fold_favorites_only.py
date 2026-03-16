from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Change these when testing another fold
FOLD_DIR = EXPORTS_DIR / "walkforward" / "train_to_2023_test_2024"
TEST_SEASON = 2024

MODEL_PROBS_PATH = FOLD_DIR / "model_probs_test_only.csv"
CLOSING_LINES_PATH = EXPORTS_DIR / "closing_lines.csv"
GAMES_MASTER_PATH = EXPORTS_DIR / "games_master.csv"

OUT_DIR = FOLD_DIR / "analysis_favorites_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


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


def american_profit_per_dollar(odds: pd.Series) -> pd.Series:
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
            books=("book_key", "nunique"),
        )
    )

    home = (
        consensus.loc[consensus["side"] == "home", ["game_id", "closing_odds", "books"]]
        .rename(columns={"closing_odds": "closing_home_ml", "books": "home_books"})
        .copy()
    )

    away = (
        consensus.loc[consensus["side"] == "away", ["game_id", "closing_odds", "books"]]
        .rename(columns={"closing_odds": "closing_away_ml", "books": "away_books"})
        .copy()
    )

    return home.merge(away, on="game_id", how="outer")


def build_bets() -> pd.DataFrame:
    mp = pd.read_csv(MODEL_PROBS_PATH, low_memory=False)
    closing = pd.read_csv(CLOSING_LINES_PATH, low_memory=False)
    gm = pd.read_csv(GAMES_MASTER_PATH, low_memory=False)

    needed_mp = [
        "game_id", "Season", "Week", "game_date",
        "away_team", "home_team", "p_home_win", "p_away_win"
    ]
    missing_mp = [c for c in needed_mp if c not in mp.columns]
    if missing_mp:
        raise ValueError(f"model_probs_test_only.csv missing columns: {missing_mp}")

    needed_gm = ["game_id", "home_win", "away_win"]
    missing_gm = [c for c in needed_gm if c not in gm.columns]
    if missing_gm:
        raise ValueError(f"games_master.csv missing columns: {missing_gm}")

    mp["Season"] = pd.to_numeric(mp["Season"], errors="coerce").astype("Int64")
    mp = mp[mp["Season"] == TEST_SEASON].copy()

    cl = build_consensus_closing_moneyline(closing)
    df = mp[needed_mp].merge(cl, on="game_id", how="inner")
    df = df.merge(gm[needed_gm], on="game_id", how="left")

    home = df.copy()
    home["side"] = "home"
    home["closing_odds"] = pd.to_numeric(home["closing_home_ml"], errors="coerce")
    home["model_prob"] = pd.to_numeric(home["p_home_win"], errors="coerce")
    home["actual_win"] = pd.to_numeric(home["home_win"], errors="coerce")
    home["books_used"] = pd.to_numeric(home["home_books"], errors="coerce")

    away = df.copy()
    away["side"] = "away"
    away["closing_odds"] = pd.to_numeric(away["closing_away_ml"], errors="coerce")
    away["model_prob"] = pd.to_numeric(away["p_away_win"], errors="coerce")
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

    bets["implied_prob"] = american_to_implied_prob(bets["closing_odds"])
    bets["edge"] = bets["model_prob"] - bets["implied_prob"]
    bets["profit_per_dollar_win"] = american_profit_per_dollar(bets["closing_odds"])
    bets["profit"] = np.where(bets["actual_win"] == 1, bets["profit_per_dollar_win"], -1.0)
    bets["favorite_or_dog"] = np.where(bets["closing_odds"] < 0, "favorite", "underdog")

    return bets


def summarize_thresholds(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for th in THRESHOLDS:
        sub = df[df["edge"] >= th].copy()
        n = len(sub)
        profit = float(sub["profit"].sum()) if n else 0.0

        rows.append(
            {
                "subset": label,
                "edge_threshold": th,
                "bets": n,
                "profit_units": profit,
                "roi": (profit / n) if n else np.nan,
                "win_rate": float(sub["actual_win"].mean()) if n else np.nan,
                "avg_edge": float(sub["edge"].mean()) if n else np.nan,
                "avg_model_prob": float(sub["model_prob"].mean()) if n else np.nan,
                "avg_implied_prob": float(sub["implied_prob"].mean()) if n else np.nan,
                "avg_odds": float(sub["closing_odds"].mean()) if n else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    bets = build_bets()

    favorites = bets[bets["favorite_or_dog"] == "favorite"].copy()
    underdogs = bets[bets["favorite_or_dog"] == "underdog"].copy()

    favorite_summary = summarize_thresholds(favorites, "favorites_only")
    underdog_summary = summarize_thresholds(underdogs, "underdogs_only")
    all_summary = summarize_thresholds(bets, "all_sides")

    favorite_summary.to_csv(OUT_DIR / "favorite_threshold_summary.csv", index=False)
    underdog_summary.to_csv(OUT_DIR / "underdog_threshold_summary.csv", index=False)
    all_summary.to_csv(OUT_DIR / "all_threshold_summary.csv", index=False)

    top_favorites = favorites.sort_values(["edge", "game_date"], ascending=[False, True], kind="stable")
    top_underdogs = underdogs.sort_values(["edge", "game_date"], ascending=[False, True], kind="stable")

    top_favorites.head(50).to_csv(OUT_DIR / "top_50_favorites.csv", index=False)
    top_underdogs.head(50).to_csv(OUT_DIR / "top_50_underdogs.csv", index=False)

    print("\nALL SIDES")
    print(all_summary.to_string(index=False))

    print("\nFAVORITES ONLY")
    print(favorite_summary.to_string(index=False))

    print("\nUNDERDOGS ONLY")
    print(underdog_summary.to_string(index=False))

    print("\nTOP 20 FAVORITES BY EDGE")
    print(
        top_favorites[
            ["game_id", "side", "closing_odds", "model_prob", "implied_prob", "edge", "actual_win", "profit"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nTOP 20 UNDERDOGS BY EDGE")
    print(
        top_underdogs[
            ["game_id", "side", "closing_odds", "model_prob", "implied_prob", "edge", "actual_win", "profit"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
