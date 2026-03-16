from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "exports"

MODEL_PROBS = EXPORTS_DIR / "walkforward" / "train_to_2023_test_2024" / "model_probs_test_only.csv"
CLOSING_LINES = EXPORTS_DIR / "closing_lines.csv"
GAMES_MASTER = EXPORTS_DIR / "games_master.csv"

OUT_PATH = EXPORTS_DIR / "calibration_table.csv"


def american_to_prob(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(
        odds > 0,
        100 / (odds + 100),
        (-odds) / ((-odds) + 100)
    )


def build_closing_lines(df):

    df = df[df["market"].str.lower() == "moneyline"].copy()
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")

    consensus = (
        df.groupby(["game_id", "side"])
        .agg(
            closing_odds=("odds", "median")
        )
        .reset_index()
    )

    home = consensus[consensus["side"] == "home"][["game_id", "closing_odds"]]
    home = home.rename(columns={"closing_odds": "home_odds"})

    away = consensus[consensus["side"] == "away"][["game_id", "closing_odds"]]
    away = away.rename(columns={"closing_odds": "away_odds"})

    merged = home.merge(away, on="game_id")

    return merged


def odds_bucket(odds):

    if odds <= -300:
        return "<=-300"
    if odds <= -200:
        return "-300 to -200"
    if odds <= -150:
        return "-200 to -150"
    if odds <= -110:
        return "-150 to -110"
    if odds < 110:
        return "-110 to +110"
    if odds < 150:
        return "+110 to +150"
    if odds < 200:
        return "+150 to +200"
    if odds < 300:
        return "+200 to +300"
    return ">=+300"


def main():

    print("Loading files...")

    mp = pd.read_csv(MODEL_PROBS)
    closing = pd.read_csv(CLOSING_LINES)
    gm = pd.read_csv(GAMES_MASTER)

    closing = build_closing_lines(closing)

    df = mp.merge(closing, on="game_id")
    df = df.merge(gm[["game_id", "home_win", "away_win"]], on="game_id")

    rows = []

    for _, r in df.iterrows():

        rows.append({
            "game_id": r["game_id"],
            "side": "home",
            "model_prob": r["p_home_win"],
            "odds": r["home_odds"],
            "actual": r["home_win"]
        })

        rows.append({
            "game_id": r["game_id"],
            "side": "away",
            "model_prob": r["p_away_win"],
            "odds": r["away_odds"],
            "actual": r["away_win"]
        })

    bets = pd.DataFrame(rows)

    bets["implied_prob"] = american_to_prob(bets["odds"])

    bets["bucket"] = bets["odds"].apply(odds_bucket)

    table = (
        bets
        .groupby("bucket")
        .agg(
            bets=("actual", "count"),
            avg_model_prob=("model_prob", "mean"),
            avg_implied_prob=("implied_prob", "mean"),
            actual_win_rate=("actual", "mean")
        )
        .reset_index()
    )

    table["model_edge"] = table["avg_model_prob"] - table["avg_implied_prob"]
    table["actual_edge"] = table["actual_win_rate"] - table["avg_implied_prob"]

    table["trust_factor"] = table["actual_edge"] / table["model_edge"]

    table["trust_factor"] = table["trust_factor"].clip(0, 1)

    table = table.sort_values("bucket")

    print("\nCALIBRATION TABLE\n")
    print(table)

    table.to_csv(OUT_PATH, index=False)

    print("\nSaved calibration table:", OUT_PATH)


if __name__ == "__main__":
    main()
