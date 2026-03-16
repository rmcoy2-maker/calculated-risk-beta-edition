from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "exports" / "edges_market_anchor.csv"
OUT_PATH = PROJECT_ROOT / "exports" / "market_anchor_shortlist.csv"

EDGE_THRESHOLD = 0.05
MAX_ODDS = 150


def main() -> None:
    print("Loading market-anchored edges...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    needed = [
        "game_id", "Season", "Week", "game_date",
        "away_team", "home_team", "side", "closing_odds",
        "model_prob_raw", "model_prob_adj", "implied_prob",
        "edge_raw", "edge", "actual_win", "profit", "books_used"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    num_cols = [
        "Season", "Week", "closing_odds", "model_prob_raw", "model_prob_adj",
        "implied_prob", "edge_raw", "edge", "actual_win", "profit", "books_used"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["favorite_or_dog"] = df["closing_odds"].apply(lambda x: "favorite" if pd.notna(x) and x < 0 else "underdog")
    df["bet_team"] = df.apply(
        lambda r: r["home_team"] if str(r["side"]).lower() == "home" else r["away_team"],
        axis=1
    )

    shortlist = df[
        df["closing_odds"].notna()
        & (df["favorite_or_dog"] == "favorite")
        & (df["edge"] >= EDGE_THRESHOLD)
        & (df["closing_odds"] <= MAX_ODDS)
    ].copy()

    shortlist = shortlist.sort_values(
        ["edge", "model_prob_adj", "books_used"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    shortlist = shortlist[
        [
            "game_id", "Season", "Week", "game_date",
            "away_team", "home_team", "bet_team", "side",
            "closing_odds", "model_prob_raw", "model_prob_adj",
            "implied_prob", "edge_raw", "edge",
            "books_used", "actual_win", "profit"
        ]
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(OUT_PATH, index=False)

    print(f"Saved shortlist: {OUT_PATH}")
    print(f"Rows: {len(shortlist)}")

    print("\nTOP 30 SHORTLIST BETS")
    print(shortlist.head(30).to_string(index=False))

    print("\nSUMMARY")
    if len(shortlist):
        print("bets:", len(shortlist))
        print("wins:", int(shortlist["actual_win"].sum()))
        print("win_rate:", round(float(shortlist["actual_win"].mean()), 6))
        print("profit_units:", round(float(shortlist["profit"].sum()), 6))
        print("roi:", round(float(shortlist["profit"].sum() / len(shortlist)), 6))
        print("avg_edge:", round(float(shortlist["edge"].mean()), 6))
        print("avg_odds:", round(float(shortlist["closing_odds"].mean()), 6))


if __name__ == "__main__":
    main()
