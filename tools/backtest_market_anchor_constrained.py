from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "exports" / "edges_market_anchor.csv"
OUT_SUMMARY_PATH = PROJECT_ROOT / "exports" / "backtest_market_anchor_constrained_summary.csv"
OUT_BETS_PATH = PROJECT_ROOT / "exports" / "backtest_market_anchor_constrained_bets.csv"


EDGE_THRESHOLDS = [0.03, 0.05, 0.08, 0.10]
ODDS_CAPS = [150, 175, 200]


def summarize(sub: pd.DataFrame, label: str, edge_threshold: float, odds_cap: int) -> dict:
    n = len(sub)
    return {
        "subset": label,
        "edge_threshold": edge_threshold,
        "odds_cap": odds_cap,
        "bets": n,
        "wins": float(sub["actual_win"].sum()) if n else 0.0,
        "win_rate": float(sub["actual_win"].mean()) if n else np.nan,
        "profit_units": float(sub["profit"].sum()) if n else 0.0,
        "roi": float(sub["profit"].sum() / n) if n else np.nan,
        "avg_edge": float(sub["edge"].mean()) if n else np.nan,
        "avg_edge_raw": float(sub["edge_raw"].mean()) if n else np.nan,
        "avg_model_prob_raw": float(sub["model_prob_raw"].mean()) if n else np.nan,
        "avg_model_prob_adj": float(sub["model_prob_adj"].mean()) if n else np.nan,
        "avg_implied_prob": float(sub["implied_prob"].mean()) if n else np.nan,
        "avg_odds": float(sub["closing_odds"].mean()) if n else np.nan,
        "avg_books_used": float(sub["books_used"].mean()) if n else np.nan,
    }


def main() -> None:
    print("Loading market-anchored edges...")
    bets = pd.read_csv(INPUT_PATH, low_memory=False)

    required = [
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
    missing = [c for c in required if c not in bets.columns]
    if missing:
        raise ValueError(f"edges_market_anchor.csv missing columns: {missing}")

    for col in [
        "Season",
        "Week",
        "closing_odds",
        "model_prob_raw",
        "model_prob_adj",
        "implied_prob",
        "edge_raw",
        "edge",
        "actual_win",
        "profit",
        "books_used",
    ]:
        bets[col] = pd.to_numeric(bets[col], errors="coerce")

    bets["favorite_or_dog"] = np.where(bets["closing_odds"] < 0, "favorite", "underdog")

    summary_rows = []
    kept_bets = []

    for edge_th in EDGE_THRESHOLDS:
        for odds_cap in ODDS_CAPS:
            base = bets[
                bets["closing_odds"].notna()
                & (bets["closing_odds"].abs() >= 100)
                & (bets["closing_odds"] <= odds_cap)
                & (bets["edge"] >= edge_th)
            ].copy()

            if len(base):
                tagged = base.copy()
                tagged["test_edge_threshold"] = edge_th
                tagged["test_odds_cap"] = odds_cap
                kept_bets.append(tagged)

            summary_rows.append(summarize(base, "all", edge_th, odds_cap))
            summary_rows.append(
                summarize(base[base["favorite_or_dog"] == "favorite"].copy(), "favorites_only", edge_th, odds_cap)
            )
            summary_rows.append(
                summarize(base[base["favorite_or_dog"] == "underdog"].copy(), "underdogs_only", edge_th, odds_cap)
            )
            summary_rows.append(
                summarize(
                    base[(base["favorite_or_dog"] == "underdog") & (base["closing_odds"] <= 150)].copy(),
                    "dogs_upto_150",
                    edge_th,
                    odds_cap,
                )
            )
            summary_rows.append(
                summarize(
                    base[
                        (base["favorite_or_dog"] == "underdog")
                        & (base["closing_odds"] > 150)
                        & (base["closing_odds"] <= 175)
                    ].copy(),
                    "dogs_150_to_175",
                    edge_th,
                    odds_cap,
                )
            )
            summary_rows.append(
                summarize(
                    base[
                        (base["favorite_or_dog"] == "underdog")
                        & (base["closing_odds"] > 175)
                        & (base["closing_odds"] <= 200)
                    ].copy(),
                    "dogs_175_to_200",
                    edge_th,
                    odds_cap,
                )
            )

    summary = pd.DataFrame(summary_rows)
    constrained_bets = pd.concat(kept_bets, ignore_index=True) if kept_bets else pd.DataFrame()

    OUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY_PATH, index=False)
    constrained_bets.to_csv(OUT_BETS_PATH, index=False)

    print(f"\nSaved summary: {OUT_SUMMARY_PATH}")
    print(f"Saved bets: {OUT_BETS_PATH}")

    print("\nTOP RESULTS BY ROI (min 15 bets)")
    top = (
        summary[summary["bets"] >= 15]
        .sort_values(["roi", "bets"], ascending=[False, False], kind="stable")
        .reset_index(drop=True)
    )
    print(top.head(30).to_string(index=False))

    print("\nFULL SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()