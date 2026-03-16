from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Projects\calculated-risk-beta-edition")
ODDS_DIR = PROJECT_ROOT / "exports" / "historical_odds"
OUTPUT_PATH = ODDS_DIR / "nfl_odds_full_merged.csv"

# Files to skip because they are already combined/derived outputs
SKIP_FILES = {
    "nfl_historical_odds_combined.csv",
    "nfl_historical_odds_2020_2022_weekly_combined.csv",
    "nfl_historical_odds_2020_2025_master.csv",
    "nfl_odds_full_merged.csv",
    "nfl_open_close_odds.csv",
    "nfl_open_mid_close_odds.csv",
    "nfl_open_wed_close_odds.csv",
}


def normalize_team_name(name: str) -> str:
    if pd.isna(name):
        return ""
    return str(name).strip()


def main() -> None:
    csv_files = sorted(ODDS_DIR.glob("*.csv"))
    source_files = [p for p in csv_files if p.name not in SKIP_FILES]

    if not source_files:
        raise FileNotFoundError("No source odds CSV files found to merge.")

    dfs: list[pd.DataFrame] = []

    print("Merging source files:\n")
    for path in source_files:
        print(f"Loading: {path}")
        df = pd.read_csv(path, low_memory=False)

        required = {"away_team", "home_team", "commence_time", "outcome_name", "outcome_price"}
        missing = required - set(df.columns)
        if missing:
            print(f"  Skipping {path.name} because required columns are missing: {sorted(missing)}")
            continue

        df = df.copy()
        df["source_file"] = path.name

        df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)

        # NFL season convention:
        # Jan-Feb games belong to the prior season.
        df["Season"] = df["commence_time"].dt.year
        jan_feb_mask = df["commence_time"].dt.month.isin([1, 2])
        df.loc[jan_feb_mask, "Season"] = df.loc[jan_feb_mask, "Season"] - 1

        df["away_team"] = df["away_team"].map(normalize_team_name)
        df["home_team"] = df["home_team"].map(normalize_team_name)

        df["game_date"] = df["commence_time"].dt.strftime("%Y-%m-%d")
        df["game_id"] = (
            df["Season"].astype("Int64").astype(str)
            + "_"
            + df["away_team"]
            + "_"
            + df["home_team"]
            + "_"
            + df["commence_time"].dt.strftime("%Y%m%d")
        )

        dfs.append(df)

    if not dfs:
        raise ValueError("No usable source files were loaded.")

    merged = pd.concat(dfs, ignore_index=True)

    # Standardize a few optional fields if present
    if "market_key" in merged.columns:
        merged["market_key"] = merged["market_key"].astype(str).str.strip().str.lower()

    if "book_key" in merged.columns:
        merged["book_key"] = merged["book_key"].astype(str).str.strip().str.lower()

    if "outcome_name" in merged.columns:
        merged["outcome_name"] = merged["outcome_name"].astype(str).str.strip()

    # Drop exact duplicate rows if any
    dedupe_cols = [c for c in [
        "event_id",
        "book_key",
        "market_key",
        "outcome_name",
        "outcome_point",
        "outcome_price",
        "snapshot_timestamp",
        "requested_snapshot",
        "commence_time",
    ] if c in merged.columns]

    if dedupe_cols:
        before = len(merged)
        merged = merged.drop_duplicates(subset=dedupe_cols).copy()
        after = len(merged)
        print(f"\nDropped exact duplicates: {before - after:,}")

    merged = merged.sort_values(
        ["Season", "commence_time", "away_team", "home_team"],
        kind="stable"
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(f"Rows: {len(merged):,}")

    print("\nSeason counts:")
    print(merged.groupby("Season", dropna=False).size().to_string())

    print("\nColumns:")
    print(list(merged.columns))


if __name__ == "__main__":
    main()