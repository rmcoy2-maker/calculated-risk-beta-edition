from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMES_PATH = PROJECT_ROOT / "exports" / "games_master_recent_form_market.csv"
OUT_PATH = PROJECT_ROOT / "exports" / "games_master_recent_form_market_regime.csv"


def main():
    print("Loading enriched features...")
    df = pd.read_csv(GAMES_PATH, low_memory=False)

    required = ["game_id", "Season", "Week", "game_date", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["game_total_points"] = df["home_score"] + df["away_score"]

    df = df.sort_values(["Season", "Week", "game_date", "game_id"], kind="stable").reset_index(drop=True)

    # crude calendar flags from weekday; adjust later if you have exact kickoff time/source
    df["weekday"] = df["game_date"].dt.day_name()
    df["is_thursday"] = (df["weekday"] == "Thursday").astype(int)
    df["is_monday"] = (df["weekday"] == "Monday").astype(int)
    df["is_sunday"] = (df["weekday"] == "Sunday").astype(int)

    # holiday proxy
    df["month"] = df["game_date"].dt.month
    df["day"] = df["game_date"].dt.day
    df["is_thanksgiving_window"] = (
        (df["month"] == 11) & (df["day"] >= 22) & (df["day"] <= 29) & (df["is_thursday"] == 1)
    ).astype(int)

    df["is_late_season"] = (df["Week"] >= 12).astype(int)
    df["is_early_season"] = (df["Week"] <= 3).astype(int)
    df["is_peak_scoring_window"] = df["Week"].between(6, 7).astype(int)

    def per_season(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["Week", "game_date", "game_id"], kind="stable").copy()
        prev_total = g["game_total_points"].shift(1)

        g["league_points_avg_last4"] = prev_total.rolling(4, min_periods=1).mean()
        g["league_points_avg_last8"] = prev_total.rolling(8, min_periods=1).mean()
        g["league_points_std_last8"] = prev_total.rolling(8, min_periods=2).std()
        g["league_points_trend_last4_vs_last8"] = g["league_points_avg_last4"] - g["league_points_avg_last8"]

        return g

    df = df.groupby("Season", group_keys=False).apply(per_season).reset_index(drop=True)

    # simple scoring regime labels
    df["league_scoring_regime"] = pd.cut(
        df["league_points_trend_last4_vs_last8"],
        bins=[-999, -3, -1, 1, 3, 999],
        labels=["strong_down", "down", "flat", "up", "strong_up"],
        include_lowest=True,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(df):,}")
    print("\nSample regime columns:")
    sample = [
        "game_id", "Season", "Week",
        "league_points_avg_last4", "league_points_avg_last8",
        "league_points_trend_last4_vs_last8", "league_scoring_regime",
        "is_thursday", "is_monday", "is_thanksgiving_window",
        "is_early_season", "is_peak_scoring_window", "is_late_season",
    ]
    print(df[sample].head(20).to_string(index=False))


if __name__ == "__main__":
    main()