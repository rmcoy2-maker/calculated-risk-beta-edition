from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PROBS_PATH = PROJECT_ROOT / "exports" / "model_probs.csv"
CLOSING_LINES_PATH = PROJECT_ROOT / "exports" / "closing_lines.csv"
GAMES_MASTER_PATH = PROJECT_ROOT / "exports" / "games_master.csv"

OUT_BETS_PATH = PROJECT_ROOT / "exports" / "backtest_bets_moneyline_constrained.csv"
OUT_SUMMARY_PATH = PROJECT_ROOT / "exports" / "backtest_summary_moneyline_constrained.csv"

THRESHOLDS = [0.05, 0.08, 0.09, 0.10, 0.12]

MIN_CLOSING_ODDS = -150
MAX_CLOSING_ODDS = 110
MIN_MODEL_PROB = 0.60
MAX_MODEL_PROB = 0.75


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


def summarize_thresholds(bets: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []

    for th in thresholds:
        sub = bets[bets["edge"] >= th].copy()
        n = len(sub)
        profit = float(sub["profit"].sum()) if n else 0.0

        rows.append(
            {
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

    bets = bets[
        (bets["closing_odds"] >= MIN_CLOSING_ODDS)
        & (bets["closing_odds"] < MAX_CLOSING_ODDS)
        & (bets["model_prob"] >= MIN_MODEL_PROB)
        & (bets["model_prob"] < MAX_MODEL_PROB)
    ].copy()

    bets = bets.sort_values(
        ["edge", "game_date"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    summary_all = summarize_thresholds(bets, THRESHOLDS)
    summary_2024 = summarize_thresholds(bets[bets["Season"] == 2024].copy(), THRESHOLDS)
    summary_2025 = summarize_thresholds(bets[bets["Season"] == 2025].copy(), THRESHOLDS)

    OUT_BETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bets.to_csv(OUT_BETS_PATH, index=False)
    summary_all.to_csv(OUT_SUMMARY_PATH, index=False)
    summary_2024.to_csv(PROJECT_ROOT / "exports" / "backtest_summary_moneyline_constrained_2024.csv", index=False)
    summary_2025.to_csv(PROJECT_ROOT / "exports" / "backtest_summary_moneyline_constrained_2025.csv", index=False)

    print("\nConstrained rule universe:")
    print(f"  closing_odds >= {MIN_CLOSING_ODDS}")
    print(f"  closing_odds <  {MAX_CLOSING_ODDS}")
    print(f"  model_prob >= {MIN_MODEL_PROB:.2f}")
    print(f"  model_prob <  {MAX_MODEL_PROB:.2f}")

    print(f"\nSaved bets: {OUT_BETS_PATH}")
    print(f"Rows after constraints: {len(bets):,}")
    print(f"Saved summary: {OUT_SUMMARY_PATH}")

    print("\nALL SEASONS")
    print(summary_all.to_string(index=False))

    print("\n2024 ONLY")
    print(summary_2024.to_string(index=False))

    print("\n2025 ONLY")
    print(summary_2025.to_string(index=False))


if __name__ == "__main__":
    main()
