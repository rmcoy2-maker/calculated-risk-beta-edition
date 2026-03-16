from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS = PROJECT_ROOT / "exports"

MODEL_PROBS = EXPORTS / "walkforward/train_to_2023_test_2024/model_probs_test_only.csv"
CLOSING = EXPORTS / "closing_lines.csv"
GAMES = EXPORTS / "games_master.csv"


def american_to_prob(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(
        odds > 0,
        100 / (odds + 100),
        (-odds) / ((-odds) + 100)
    )


def profit(odds, win):

    if win == 0:
        return -1

    if odds > 0:
        return odds / 100

    return 100 / abs(odds)


def build_closing(df):

    ml = df[df["market"].str.lower() == "moneyline"].copy()
    ml["odds"] = pd.to_numeric(ml["odds"], errors="coerce")

    agg = (
        ml.groupby(["game_id", "side"])
        .agg(odds=("odds", "median"))
        .reset_index()
    )

    home = agg[agg["side"] == "home"][["game_id", "odds"]]
    home = home.rename(columns={"odds": "home_odds"})

    away = agg[agg["side"] == "away"][["game_id", "odds"]]
    away = away.rename(columns={"odds": "away_odds"})

    return home.merge(away, on="game_id")


def main():

    mp = pd.read_csv(MODEL_PROBS)
    closing = pd.read_csv(CLOSING)
    gm = pd.read_csv(GAMES)

    closing = build_closing(closing)

    df = mp.merge(closing, on="game_id")
    df = df.merge(gm[["game_id","home_win","away_win"]], on="game_id")

    df["home_implied"] = american_to_prob(df["home_odds"])
    df["away_implied"] = american_to_prob(df["away_odds"])

    df["home_edge"] = df["p_home_win"] - df["home_implied"]
    df["away_edge"] = df["p_away_win"] - df["away_implied"]

    bets = []

    for _, r in df.iterrows():

        if r["home_edge"] > r["away_edge"]:

            side = "home"
            edge = r["home_edge"]
            odds = r["home_odds"]
            win = r["home_win"]

        else:

            side = "away"
            edge = r["away_edge"]
            odds = r["away_odds"]
            win = r["away_win"]

        bets.append({
            "edge": edge,
            "odds": odds,
            "win": win
        })

    bets = pd.DataFrame(bets)

    thresholds = [0, .02, .05, .08, .10]

    rows = []

    for t in thresholds:

        subset = bets[bets["edge"] >= t]

        n = len(subset)

        if n == 0:
            continue

        profit_units = subset.apply(
            lambda r: profit(r["odds"], r["win"]), axis=1
        ).sum()

        rows.append({
            "threshold": t,
            "bets": n,
            "win_rate": subset["win"].mean(),
            "profit_units": profit_units,
            "roi": profit_units / n
        })

    out = pd.DataFrame(rows)

    print("\nSINGLE BET STRATEGY\n")
    print(out)


if __name__ == "__main__":
    main()
