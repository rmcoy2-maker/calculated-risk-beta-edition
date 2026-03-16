from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:\Projects\calculated-risk-beta-edition")
EXPORTS = PROJECT_ROOT / "exports"

INPUT_FILE = EXPORTS / "fort_knox_market_joined_moneyline_all_seasons.csv"
OUTPUT_FILE = EXPORTS / "fort_knox_market_joined_moneyline_scored_all_seasons.csv"


def american_to_prob(price):
    if pd.isna(price):
        return np.nan
    price = float(price)
    if price > 0:
        return 100 / (price + 100)
    else:
        return abs(price) / (abs(price) + 100)


def prob_to_moneyline(p):
    if pd.isna(p):
        return np.nan
    if p >= 0.5:
        return -100 * p / (1 - p)
    else:
        return 100 * (1 - p) / p


def american_profit(price):
    if price > 0:
        return price / 100
    else:
        return 100 / abs(price)


def main():

    print("Loading joined market file...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    rows = []

    for _, r in df.iterrows():

        away_price = r["away_close_price"]
        home_price = r["home_close_price"]

        away_prob = r["away_close_prob"]
        home_prob = r["home_close_prob"]

        if pd.isna(away_prob) or pd.isna(home_prob):
            continue

        # choose side with higher model probability
        if away_prob > home_prob:
            selected_team = r["away_team"]
            model_prob = away_prob
            closing_odds = away_price
            implied_prob = american_to_prob(away_price)
            win = 1 if r["away_score"] > r["home_score"] else 0
        else:
            selected_team = r["home_team"]
            model_prob = home_prob
            closing_odds = home_price
            implied_prob = american_to_prob(home_price)
            win = 1 if r["home_score"] > r["away_score"] else 0

        true_moneyline = prob_to_moneyline(model_prob)

        if win:
            profit = american_profit(closing_odds)
        else:
            profit = -1

        edge = model_prob - implied_prob

        rows.append({
            "game_id": r["game_id"],
            "Season": r["Season"],
            "Week": r["Week"],
            "game_date": r["game_date"],
            "selected_team": selected_team,
            "model_prob": model_prob,
            "true_moneyline": true_moneyline,
            "closing_odds": closing_odds,
            "implied_prob": implied_prob,
            "edge": edge,
            "actual_win": win,
            "profit": profit
        })

    out = pd.DataFrame(rows)

    print("Rows created:", len(out))

    out.to_csv(OUTPUT_FILE, index=False)

    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()