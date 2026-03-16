from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Projects\calculated-risk-beta-edition")
EXPORTS = PROJECT_ROOT / "exports"

MARKET_PATH = EXPORTS / "fort_knox_market_joined_moneyline_all_seasons.csv"
MODEL_PATH = EXPORTS / "model_probs.csv"
OUTPUT_PATH = EXPORTS / "fort_knox_market_joined_moneyline_scored_all_seasons.csv"


def american_to_implied_prob(price):
    if pd.isna(price):
        return np.nan
    try:
        p = float(price)
    except Exception:
        return np.nan

    if p > 0:
        return 100.0 / (p + 100.0)
    if p < 0:
        return abs(p) / (abs(p) + 100.0)
    return np.nan


def prob_to_moneyline(prob):
    if pd.isna(prob):
        return np.nan
    try:
        prob = float(prob)
    except Exception:
        return np.nan

    if prob <= 0 or prob >= 1:
        return np.nan

    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def profit_per_unit(american_odds, won):
    if pd.isna(american_odds):
        return np.nan
    try:
        american_odds = float(american_odds)
    except Exception:
        return np.nan

    if int(won) == 1:
        if american_odds > 0:
            return american_odds / 100.0
        if american_odds < 0:
            return 100.0 / abs(american_odds)
        return np.nan

    return -1.0


def main() -> None:
    print("Loading market join...")
    market = pd.read_csv(MARKET_PATH, low_memory=False)

    print("Loading model probabilities...")
    model = pd.read_csv(MODEL_PATH, low_memory=False)

    # Normalize key columns
    market["game_id"] = market["game_id"].astype(str).str.strip()
    model["game_id"] = model["game_id"].astype(str).str.strip()

    for col in ["Season", "Week"]:
        if col in market.columns:
            market[col] = pd.to_numeric(market[col], errors="coerce").astype("Int64")
        if col in model.columns:
            model[col] = pd.to_numeric(model[col], errors="coerce").astype("Int64")

    for col in [
        "away_close_price",
        "home_close_price",
        "away_close_prob",
        "home_close_prob",
        "away_score",
        "home_score",
    ]:
        if col in market.columns:
            market[col] = pd.to_numeric(market[col], errors="coerce")

    for col in ["p_home_win", "p_away_win", "pred_home_margin", "pred_total"]:
        if col in model.columns:
            model[col] = pd.to_numeric(model[col], errors="coerce")

    keep_model_cols = [
        c for c in [
            "game_id",
            "Season",
            "Week",
            "game_date",
            "p_home_win",
            "p_away_win",
            "pred_home_margin",
            "pred_total",
        ]
        if c in model.columns
    ]

    print("Merging market + model...")
    merged = market.merge(
        model[keep_model_cols],
        on="game_id",
        how="inner",
        suffixes=("", "_model"),
    )

    print(f"Merged rows: {len(merged):,}")

    rows = []

    for _, r in merged.iterrows():
        home_team = r.get("home_team", np.nan)
        away_team = r.get("away_team", np.nan)

        p_home = r.get("p_home_win", np.nan)
        p_away = r.get("p_away_win", np.nan)

        home_price = r.get("home_close_price", np.nan)
        away_price = r.get("away_close_price", np.nan)

        home_close_prob = r.get("home_close_prob", np.nan)
        away_close_prob = r.get("away_close_prob", np.nan)

        home_score = r.get("home_score", np.nan)
        away_score = r.get("away_score", np.nan)

        # Need model probs and prices to score a side
        if pd.isna(p_home) or pd.isna(p_away):
            continue
        if pd.isna(home_price) or pd.isna(away_price):
            continue
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        # Choose the side with higher model win probability
        if p_home >= p_away:
            selected_team = home_team
            selected_team_full = home_team
            model_prob = p_home
            closing_odds = home_price
            close_prob = home_close_prob
            implied_prob = american_to_implied_prob(home_price)
            actual_win = 1 if home_score > away_score else 0
            opponent = away_team
        else:
            selected_team = away_team
            selected_team_full = away_team
            model_prob = p_away
            closing_odds = away_price
            close_prob = away_close_prob
            implied_prob = american_to_implied_prob(away_price)
            actual_win = 1 if away_score > home_score else 0
            opponent = home_team

        true_moneyline = prob_to_moneyline(model_prob)
        edge = model_prob - implied_prob
        profit = profit_per_unit(closing_odds, actual_win)

        rows.append(
            {
                "game_id": r.get("game_id"),
                "Season": r.get("Season"),
                "Week": r.get("Week"),
                "game_date": r.get("game_date"),
                "home_team": home_team,
                "away_team": away_team,
                "selected_team": selected_team,
                "selected_team_full": selected_team_full,
                "opponent_team": opponent,
                "model_prob": model_prob,
                "p_home_win": p_home,
                "p_away_win": p_away,
                "true_moneyline": true_moneyline,
                "closing_odds": closing_odds,
                "implied_prob": implied_prob,
                "close_prob": close_prob,
                "edge": edge,
                "actual_win": actual_win,
                "profit": profit,
                "home_score": home_score,
                "away_score": away_score,
                "pred_home_margin": r.get("pred_home_margin", np.nan),
                "pred_total": r.get("pred_total", np.nan),
                "away_close_price": away_price,
                "home_close_price": home_price,
                "away_close_prob": away_close_prob,
                "home_close_prob": home_close_prob,
            }
        )

    out = pd.DataFrame(rows)

    print(f"Scored bet rows: {len(out):,}")
    print("\nSeason counts:")
    if "Season" in out.columns:
        print(out.groupby("Season", dropna=False).size().to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()