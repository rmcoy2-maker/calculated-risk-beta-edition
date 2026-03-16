from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS = PROJECT_ROOT / "exports"

MODEL_PROBS = EXPORTS / "walkforward" / "train_to_2023_test_2024" / "model_probs_test_only.csv"
CLOSING = EXPORTS / "closing_lines.csv"
GAMES = EXPORTS / "games_master.csv"
MARKET_TRACKER = EXPORTS / "market_favorite_tracker.csv"

OUT_DIR = EXPORTS / "walkforward" / "train_to_2023_test_2024" / "miss_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_CUT = 0.05


def american_to_prob(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(
        odds > 0,
        100 / (odds + 100),
        (-odds) / ((-odds) + 100)
    )


def profit_per_unit(odds, win):
    if win == 0:
        return -1.0
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def build_consensus(lines: pd.DataFrame) -> pd.DataFrame:
    ml = lines[lines["market"].astype(str).str.lower() == "moneyline"].copy()
    ml["side"] = ml["side"].astype(str).str.lower().str.strip()
    ml["odds"] = pd.to_numeric(ml["odds"], errors="coerce")

    ml = ml[
        ml["side"].isin(["home", "away"])
        & ml["odds"].notna()
        & (ml["odds"].abs() >= 100)
        & (ml["odds"].abs() <= 5000)
    ].copy()

    agg = (
        ml.groupby(["game_id", "side"], as_index=False)
        .agg(
            odds=("odds", "median"),
            books=("book_key", "nunique"),
        )
    )

    home = agg[agg["side"] == "home"][["game_id", "odds", "books"]].rename(
        columns={"odds": "home_odds", "books": "home_books"}
    )
    away = agg[agg["side"] == "away"][["game_id", "odds", "books"]].rename(
        columns={"odds": "away_odds", "books": "away_books"}
    )

    return home.merge(away, on="game_id", how="inner")


def summarize(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    return {
        "group": label,
        "bets": n,
        "wins": float(df["win"].sum()) if n else 0,
        "win_rate": float(df["win"].mean()) if n else np.nan,
        "profit_units": float(df["profit"].sum()) if n else 0.0,
        "roi": float(df["profit"].sum() / n) if n else np.nan,
        "avg_edge": float(df["edge"].mean()) if n else np.nan,
        "avg_odds": float(df["odds"].mean()) if n else np.nan,
        "avg_model_prob": float(df["model_prob"].mean()) if n else np.nan,
        "avg_implied_prob": float(df["implied_prob"].mean()) if n else np.nan,
    }


def main():
    print("Loading files...")
    mp = pd.read_csv(MODEL_PROBS, low_memory=False)
    closing = pd.read_csv(CLOSING, low_memory=False)
    games = pd.read_csv(GAMES, low_memory=False)
    market = pd.read_csv(MARKET_TRACKER, low_memory=False)

    cons = build_consensus(closing)

    need_mp = ["game_id", "Season", "Week", "game_date", "away_team", "home_team", "p_home_win", "p_away_win"]
    need_games = ["game_id", "home_win", "away_win"]

    df = mp[need_mp].merge(cons, on="game_id", how="inner")
    df = df.merge(games[need_games], on="game_id", how="left")
    df = df.merge(market, on="game_id", how="left")

    rows = []
    for _, r in df.iterrows():
        home_implied = american_to_prob(pd.Series([r["home_odds"]]))[0]
        away_implied = american_to_prob(pd.Series([r["away_odds"]]))[0]

        home_edge = r["p_home_win"] - home_implied
        away_edge = r["p_away_win"] - away_implied

        if home_edge > away_edge:
            side = "home"
            edge = home_edge
            odds = r["home_odds"]
            implied = home_implied
            model_prob = r["p_home_win"]
            win = r["home_win"]
            books = r["home_books"]
            team = r["home_team"]
        else:
            side = "away"
            edge = away_edge
            odds = r["away_odds"]
            implied = away_implied
            model_prob = r["p_away_win"]
            win = r["away_win"]
            books = r["away_books"]
            team = r["away_team"]

        rows.append(
            {
                "game_id": r["game_id"],
                "Season": r["Season"],
                "Week": r["Week"],
                "game_date": r["game_date"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "bet_team": team,
                "bet_side": side,
                "odds": odds,
                "books_used": books,
                "model_prob": model_prob,
                "implied_prob": implied,
                "edge": edge,
                "win": win,
                "profit": profit_per_unit(odds, win),
                "open_favorite_side": r.get("open_favorite_side"),
                "closing_favorite_side": r.get("closing_favorite_side"),
                "favorite_flip": r.get("favorite_flip"),
                "favorite_strength_delta": r.get("favorite_strength_delta"),
                "home_ml_move": r.get("home_ml_move"),
                "away_ml_move": r.get("away_ml_move"),
                "market_pressure_side": r.get("market_pressure_side"),
            }
        )

    bets = pd.DataFrame(rows)

    bets = bets[
        bets["odds"].notna()
        & (bets["odds"].abs() >= 100)
        & (bets["odds"].abs() <= 250)
    ].copy()

    bets["favorite_or_dog"] = np.where(bets["odds"] < 0, "favorite", "underdog")

    bets["edge_band"] = pd.cut(
        bets["edge"],
        bins=[-1.0, 0.00, 0.02, 0.05, 0.08, 0.10, 1.0],
        labels=["<0.00", "0.00-0.02", "0.02-0.05", "0.05-0.08", "0.08-0.10", "0.10+"],
        right=False,
        include_lowest=True,
    )

    bets["odds_band"] = pd.cut(
        bets["odds"],
        bins=[-250, -200, -150, -110, 110, 150, 200, 250],
        labels=["-250--200", "-200--150", "-150--110", "-110-+110", "+110-+150", "+150-+200", "+200-+250"],
        right=False,
        include_lowest=True,
    )

    misses = bets[(bets["edge"] >= EDGE_CUT) & (bets["win"] == 0)].copy()
    near_below = bets[(bets["edge"] >= EDGE_CUT - 0.02) & (bets["edge"] < EDGE_CUT)].copy()
    near_above = bets[(bets["edge"] >= EDGE_CUT) & (bets["edge"] < EDGE_CUT + 0.02)].copy()

    summary = pd.DataFrame(
        [
            summarize(bets, "all_bets"),
            summarize(misses, f"misses_edge_ge_{EDGE_CUT:.2f}"),
            summarize(near_below, f"near_below_{EDGE_CUT:.2f}"),
            summarize(near_above, f"near_above_{EDGE_CUT:.2f}"),
            summarize(bets[(bets["edge"] >= EDGE_CUT) & (bets["favorite_or_dog"] == "favorite")], "favorites_edge_ge_cut"),
            summarize(bets[(bets["edge"] >= EDGE_CUT) & (bets["favorite_or_dog"] == "underdog")], "dogs_edge_ge_cut"),
        ]
    )

    by_edge_band = (
        bets.groupby(["edge_band", "favorite_or_dog"], dropna=False, observed=False)
        .agg(
            bets=("game_id", "count"),
            wins=("win", "sum"),
            profit_units=("profit", "sum"),
            avg_odds=("odds", "mean"),
            avg_edge=("edge", "mean"),
            favorite_flips=("favorite_flip", lambda s: pd.Series(s).fillna(False).astype(bool).mean()),
        )
        .reset_index()
    )
    by_edge_band["win_rate"] = by_edge_band["wins"] / by_edge_band["bets"]
    by_edge_band["roi"] = by_edge_band["profit_units"] / by_edge_band["bets"]

    by_odds_band = (
        bets.groupby(["odds_band", "edge_band"], dropna=False, observed=False)
        .agg(
            bets=("game_id", "count"),
            wins=("win", "sum"),
            profit_units=("profit", "sum"),
            avg_edge=("edge", "mean"),
        )
        .reset_index()
    )
    by_odds_band["win_rate"] = by_odds_band["wins"] / by_odds_band["bets"]
    by_odds_band["roi"] = by_odds_band["profit_units"] / by_odds_band["bets"]

    worst_misses = misses.sort_values(["edge", "profit"], ascending=[False, True], kind="stable")
    threshold_borderline = bets[(bets["edge"] >= EDGE_CUT - 0.01) & (bets["edge"] <= EDGE_CUT + 0.01)].copy()
    threshold_borderline = threshold_borderline.sort_values("edge", ascending=False, kind="stable")

    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    by_edge_band.to_csv(OUT_DIR / "by_edge_band.csv", index=False)
    by_odds_band.to_csv(OUT_DIR / "by_odds_band.csv", index=False)
    worst_misses.to_csv(OUT_DIR / "worst_misses.csv", index=False)
    threshold_borderline.to_csv(OUT_DIR / "threshold_borderline.csv", index=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))

    print("\nBY EDGE BAND")
    print(by_edge_band.to_string(index=False))

    print("\nTOP 25 MISSES")
    cols = [
        "game_id", "bet_team", "bet_side", "favorite_or_dog", "odds",
        "model_prob", "implied_prob", "edge", "win",
        "favorite_flip", "favorite_strength_delta", "market_pressure_side"
    ]
    print(worst_misses[cols].head(25).to_string(index=False))

    print(f"\nSaved files to: {OUT_DIR}")


if __name__ == "__main__":
    main()