from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "exports" / "games_master_recent_form.csv"
MARKET_TRACKER_PATH = PROJECT_ROOT / "exports" / "market_favorite_tracker.csv"
OUTPUT_PATH = PROJECT_ROOT / "exports" / "games_master_recent_form_market.csv"


def main() -> None:
    print("Loading recent-form features...")
    feats = pd.read_csv(FEATURES_PATH, low_memory=False)

    print("Loading market favorite tracker...")
    market = pd.read_csv(MARKET_TRACKER_PATH, low_memory=False)

    keep = [
        "game_id",
        "open_favorite_side",
        "closing_favorite_side",
        "open_favorite_odds",
        "closing_favorite_odds",
        "open_favorite_strength",
        "closing_favorite_strength",
        "favorite_strength_delta",
        "favorite_flip",
        "home_ml_move",
        "away_ml_move",
        "home_implied_move",
        "away_implied_move",
        "market_pressure_side",
    ]

    missing = [c for c in keep if c not in market.columns]
    if missing:
        raise ValueError(f"market_favorite_tracker.csv missing columns: {missing}")

    out = feats.merge(market[keep], on="game_id", how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(out):,}")


if __name__ == "__main__":
    main()
