from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PROBS_PATH = PROJECT_ROOT / "exports" / "model_probs.csv"
CLOSING_LINES_PATH = PROJECT_ROOT / "exports" / "closing_lines.csv"
GAMES_MASTER_PATH = PROJECT_ROOT / "exports" / "games_master.csv"

OUT_DIR = PROJECT_ROOT / "exports" / "backtest_calibration_shrinkage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.05, 0.08, 0.09, 0.10, 0.12]
SHRINK_FACTORS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]

# Optional guardrails based on what we learned so far
MIN_CLOSING_ODDS = -150
MAX_CLOSING_ODDS = 110
MIN_RAW_MODEL_PROB = 0.60
MAX_RAW_MODEL_PROB = 0.75


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


def shrink_prob(prob: pd.Series, factor: float) -> pd.Series:
    """
    Shrink model probabilities toward 0.50.

    factor = 1.00 -> no shrink
    factor = 0.50 -> cut distance from 0.50 in half
    """
    prob = pd.to_numeric(prob, errors="coerce")
    shrunk = 0.5 + (prob - 0.5) * factor
    return pd.Series(np.clip(shrunk, 0.001, 0.999), index=prob.index)


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

    if ml.empty:
        raise ValueError("No usable moneyline rows found in closing_lines.csv")

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

    out = home.merge(away, on="game_id", how="outer")

    for col in ["closing_home_ml", "closing_away_ml"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[(out[col].abs() < 100) | (out[col].abs() > 5000), col] = np.nan

    return out


def build_base_bets() -> pd.DataFrame:
    print("Loading model probabilities...")
    mp = pd.read_csv(MODEL_PROBS_PATH, low_memory=False)

    print("Loading closing lines...")
    closing = pd.read_csv(CLOSING_LINES_PATH, low_memory=False)

    print("Loading game results...")
    gm = pd.read_csv(GAMES_MASTER_PATH, low_memory=False)

    needed_mp = [
        "game_id",
        "Season",
        "Week",
        "game_date",
        "away_team",
        "home_team",
        "p_home_win",
        "p_away_win",
    ]
    missing_mp = [c for c in needed_mp if c not in mp.columns]
    if missing_mp:
        raise ValueError(f"model_probs.csv missing columns: {missing_mp}")

    needed_gm = ["game_id", "home_win", "away_win"]
    missing_gm = [c for c in needed_gm if c not in gm.columns]
    if missing_gm:
        raise ValueError(f"games_master.csv missing columns: {missing_gm}")

    print("Building consensus closing moneylines...")
    cl = build_consensus_closing_moneyline(closing)

    df = mp[needed_mp].merge(cl, on="game_id", how="inner")
    df = df.merge(gm[needed_gm], on="game_id", how="left")

    home = df.copy()
    home["side"] = "home"
    home["closing_odds"] = pd.to_numeric(home["closing_home_ml"], errors="coerce")
    home["raw_model_prob"] = pd.to_numeric(home["p_home_win"], errors="coerce")
    home["actual_win"] = pd.to_numeric(home["home_win"], errors="coerce")
    home["books_used"] = pd.to_numeric(home["home_books"], errors="coerce")

    away = df.copy()
    away["side"] = "away"
    away["closing_odds"] = pd.to_numeric(away["closing_away_ml"], errors="coerce")
    away["raw_model_prob"] = pd.to_numeric(away["p_away_win"], errors="coerce")
    away["actual_win"] = pd.to_numeric(away["away_win"], errors="coerce")
    away["books_used"] = pd.to_numeric(away["away_books"], errors="coerce")

    bets = pd.concat([home, away], ignore_index=True)

    bets = bets[
        bets["closing_odds"].notna()
        & (bets["closing_odds"].abs() >= 100)
        & (bets["closing_odds"].abs() <= 5000)
        & bets["raw_model_prob"].between(0.001, 0.999)
        & bets["actual_win"].isin([0, 1])
    ].copy()

    bets["Season"] = pd.to_numeric(bets["Season"], errors="coerce").astype("Int64")
    bets["implied_prob"] = american_to_implied_prob(bets["closing_odds"])
    bets["profit_per_dollar_win"] = american_profit_per_dollar(bets["closing_odds"])
    bets["profit"] = np.where(
        bets["actual_win"] == 1,
        bets["profit_per_dollar_win"],
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
        "raw_model_prob",
        "implied_prob",
        "actual_win",
        "profit",
        "books_used",
    ]
    bets = bets[keep_cols].copy()

    bets = bets[
        (bets["closing_odds"] >= MIN_CLOSING_ODDS)
        & (bets["closing_odds"] < MAX_CLOSING_ODDS)
        & (bets["raw_model_prob"] >= MIN_RAW_MODEL_PROB)
        & (bets["raw_model_prob"] < MAX_RAW_MODEL_PROB)
    ].copy()

    bets = bets.sort_values(
        ["game_date", "game_id", "side"],
        kind="stable",
    ).reset_index(drop=True)

    return bets


def summarize_thresholds(bets: pd.DataFrame, thresholds: list[float], shrink_factor: float, season_label: str) -> pd.DataFrame:
    rows = []

    bets = bets.copy()
    bets["model_prob"] = shrink_prob(bets["raw_model_prob"], shrink_factor)
    bets["edge"] = bets["model_prob"] - bets["implied_prob"]

    for th in thresholds:
        sub = bets[bets["edge"] >= th].copy()
        n = len(sub)
        profit = float(sub["profit"].sum()) if n else 0.0

        rows.append(
            {
                "season": season_label,
                "shrink_factor": shrink_factor,
                "edge_threshold": th,
                "bets": n,
                "profit_units": profit,
                "roi": (profit / n) if n else np.nan,
                "win_rate": float(sub["actual_win"].mean()) if n else np.nan,
                "avg_raw_model_prob": float(sub["raw_model_prob"].mean()) if n else np.nan,
                "avg_model_prob": float(sub["model_prob"].mean()) if n else np.nan,
                "avg_implied_prob": float(sub["implied_prob"].mean()) if n else np.nan,
                "avg_edge": float(sub["edge"].mean()) if n else np.nan,
                "avg_odds": float(sub["closing_odds"].mean()) if n else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    bets = build_base_bets()

    print("\nConstrained universe before shrinkage testing:")
    print(f"Rows: {len(bets):,}")
    print(f"closing_odds >= {MIN_CLOSING_ODDS}")
    print(f"closing_odds <  {MAX_CLOSING_ODDS}")
    print(f"raw_model_prob >= {MIN_RAW_MODEL_PROB:.2f}")
    print(f"raw_model_prob <  {MAX_RAW_MODEL_PROB:.2f}")

    all_results = []

    for factor in SHRINK_FACTORS:
        all_results.append(summarize_thresholds(bets, THRESHOLDS, factor, "all"))
        all_results.append(summarize_thresholds(bets[bets["Season"] == 2024].copy(), THRESHOLDS, factor, "2024"))
        all_results.append(summarize_thresholds(bets[bets["Season"] == 2025].copy(), THRESHOLDS, factor, "2025"))

    results = pd.concat(all_results, ignore_index=True)

    results.to_csv(OUT_DIR / "calibration_shrinkage_results.csv", index=False)

    best_all = (
        results[results["season"] == "all"]
        .sort_values(["roi", "profit_units", "bets"], ascending=[False, False, False], kind="stable")
        .reset_index(drop=True)
    )

    best_2024 = (
        results[results["season"] == "2024"]
        .sort_values(["roi", "profit_units", "bets"], ascending=[False, False, False], kind="stable")
        .reset_index(drop=True)
    )

    best_2025 = (
        results[results["season"] == "2025"]
        .sort_values(["roi", "profit_units", "bets"], ascending=[False, False, False], kind="stable")
        .reset_index(drop=True)
    )

    print("\nTOP 15 - ALL SEASONS")
    print(best_all.head(15).to_string(index=False))

    print("\nTOP 15 - 2024 ONLY")
    print(best_2024.head(15).to_string(index=False))

    print("\nTOP 15 - 2025 ONLY")
    print(best_2025.head(15).to_string(index=False))

    for season_label in ["all", "2024", "2025"]:
        pivot = (
            results[results["season"] == season_label]
            .pivot(index="shrink_factor", columns="edge_threshold", values="roi")
            .sort_index()
        )
        pivot.to_csv(OUT_DIR / f"roi_grid_{season_label}.csv")

    print(f"\nSaved results to: {OUT_DIR}")


if __name__ == "__main__":
    main()
