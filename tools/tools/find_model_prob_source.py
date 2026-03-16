from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Projects\calculated-risk-beta-edition")
EXPORTS = PROJECT_ROOT / "exports"

CANDIDATE_FILES = [
    EXPORTS / "fort_knox_market_joined_moneyline.csv",
    EXPORTS / "fort_knox_market_joined_moneyline_all_seasons.csv",
    EXPORTS / "games_master.csv",
    EXPORTS / "games_master_recent_form.csv",
    EXPORTS / "games_master_recent_form_market.csv",
    EXPORTS / "games_master_recent_form_market_regime.csv",
    EXPORTS / "pregame_features.csv",
    EXPORTS / "historical_odds" / "nfl_odds_full_merged.csv",
]

KEYWORDS = [
    "model_prob",
    "prob",
    "prediction",
    "pred",
    "edge",
    "true_moneyline",
    "selected_team",
    "profit",
    "actual_win",
]


def summarize_file(path: Path) -> None:
    print(f"\n=== {path.name} ===")
    if not path.exists():
        print("Missing")
        return

    try:
        df = pd.read_csv(path, nrows=5, low_memory=False)
    except Exception as e:
        print(f"Could not read file: {e}")
        return

    cols = list(df.columns)
    print(f"Column count: {len(cols)}")

    matched = [c for c in cols if any(k.lower() in c.lower() for k in KEYWORDS)]
    if matched:
        print("Matching columns:")
        print(matched)
    else:
        print("No model-related columns found.")

    season_col = None
    for cand in ["Season", "season"]:
        if cand in cols:
            season_col = cand
            break

    if season_col is not None:
        try:
            full = pd.read_csv(path, usecols=[season_col], low_memory=False)
            full[season_col] = pd.to_numeric(full[season_col], errors="coerce")
            print("Season counts:")
            print(full.groupby(season_col).size().to_string())
        except Exception as e:
            print(f"Could not summarize seasons: {e}")


def main() -> None:
    print("Scanning known exported files for model probability source...")
    for path in CANDIDATE_FILES:
        summarize_file(path)

    print("\nDone.")


if __name__ == "__main__":
    main()