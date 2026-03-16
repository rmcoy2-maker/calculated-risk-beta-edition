from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "exports" / "games_master_recent_form_market.csv"


def main() -> None:
    print("Loading enriched game features...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    candidate_cols = [
        "diff_points_for_avg_last3",
        "diff_points_for_avg_last5",
        "diff_points_against_avg_last3",
        "diff_points_against_avg_last5",
        "diff_total_yards_for_avg_last3",
        "diff_total_yards_for_avg_last5",
        "diff_win_avg_last3",
        "diff_win_avg_last5",
        "diff_points_for_trend_last3_vs_season",
        "diff_points_for_trend_last5_vs_season",
        "diff_win_trend_last3_vs_season",
        "diff_win_trend_last5_vs_season",
        "favorite_strength_delta",
        "home_implied_move",
        "away_implied_move",
    ]

    available = [c for c in candidate_cols if c in df.columns]
    print("\nAvailable columns:")
    print(available)

    if "home_win" not in df.columns:
        print("\nNo home_win column found; diagnostic limited.")
        return

    num = df[available + ["home_win"]].apply(pd.to_numeric, errors="coerce")
    corr = num.corr(numeric_only=True)["home_win"].drop("home_win").sort_values(ascending=False)

    print("\nCorrelation with home_win:")
    print(corr.to_string())

    print("\nTop positive:")
    print(corr.head(10).to_string())

    print("\nTop negative:")
    print(corr.tail(10).to_string())


if __name__ == "__main__":
    main()
