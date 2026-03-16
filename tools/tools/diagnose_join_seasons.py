from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Projects\calculated-risk-beta-edition")
EXPORTS = PROJECT_ROOT / "exports"

FILES_TO_CHECK = [
    EXPORTS / "fort_knox_market_joined_moneyline.csv",
    EXPORTS / "games_master.csv",
    EXPORTS / "games_master_recent_form.csv",
    EXPORTS / "games_master_recent_form_market.csv",
    EXPORTS / "games_master_recent_form_market_regime.csv",
    EXPORTS / "pregame_features.csv",
    EXPORTS / "historical_odds" / "nfl_odds_full_merged.csv",
]

TEAM_ALIASES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    # legacy names if needed
    "WSH": "Washington Commanders",
    "OAK": "Las Vegas Raiders",
    "SD": "Los Angeles Chargers",
    "STL": "Los Angeles Rams",
}

TEAM_NORMALIZE = {
    "Chiefs": "Kansas City Chiefs",
    "Patriots": "New England Patriots",
    "Redskins": "Washington Commanders",
    "Football Team": "Washington Commanders",
    "49ers": "San Francisco 49ers",
    "Buccaneers": "Tampa Bay Buccaneers",
    "Seahawks": "Seattle Seahawks",
    "Rams": "Los Angeles Rams",
    "Raiders": "Las Vegas Raiders",
    "Chargers": "Los Angeles Chargers",
    "Saints": "New Orleans Saints",
    "Packers": "Green Bay Packers",
    "Steelers": "Pittsburgh Steelers",
    "Vikings": "Minnesota Vikings",
    "Titans": "Tennessee Titans",
    "Cowboys": "Dallas Cowboys",
    "Giants": "New York Giants",
    "Jets": "New York Jets",
    "Browns": "Cleveland Browns",
    "Bengals": "Cincinnati Bengals",
    "Panthers": "Carolina Panthers",
    "Falcons": "Atlanta Falcons",
    "Colts": "Indianapolis Colts",
    "Texans": "Houston Texans",
    "Lions": "Detroit Lions",
    "Bears": "Chicago Bears",
    "Cardinals": "Arizona Cardinals",
    "Broncos": "Denver Broncos",
    "Bills": "Buffalo Bills",
    "Dolphins": "Miami Dolphins",
    "Eagles": "Philadelphia Eagles",
    "Jaguars": "Jacksonville Jaguars"
}

df["home_team"] = df["home_team"].replace(TEAM_NORMALIZE)
df["away_team"] = df["away_team"].replace(TEAM_NORMALIZE)

def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def normalize_team_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return TEAM_ALIASES.get(s, s)


def add_nfl_season_from_date(df: pd.DataFrame, date_col: str) -> pd.Series:
    dt = pd.to_datetime(df[date_col], errors="coerce")
    season = dt.dt.year.copy()
    season = season.where(~dt.dt.month.isin([1, 2]), season - 1)
    return season


def print_file_seasons(path: Path):
    print(f"\n=== {path.name} ===")
    if not path.exists():
        print("Missing")
        return

    df = pd.read_csv(path, low_memory=False)
    print(f"Rows: {len(df):,}")
    print(f"Columns sample: {list(df.columns)[:20]}")

    season_col = first_existing(df, ["Season", "season"])
    date_col = first_existing(df, ["game_date", "commence_time", "date", "kickoff", "event_date"])

    if season_col is not None:
        seasons = pd.to_numeric(df[season_col], errors="coerce")
    elif date_col is not None:
        seasons = add_nfl_season_from_date(df, date_col)
    else:
        print("No season/date column found.")
        return

    counts = seasons.value_counts(dropna=False).sort_index()
    print("Season counts:")
    print(counts.to_string())


def build_games_key_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).copy()

    season_col = first_existing(df, ["Season", "season"])
    week_col = first_existing(df, ["Week", "week"])
    date_col = first_existing(df, ["game_date", "date", "commence_time"])
    home_col = first_existing(df, ["home_team", "HomeTeam", "home"])
    away_col = first_existing(df, ["away_team", "AwayTeam", "away"])

    if season_col is None and date_col is not None:
        df["Season"] = add_nfl_season_from_date(df, date_col)
        season_col = "Season"

    required = [season_col, week_col, date_col, home_col, away_col]
    if any(c is None for c in required):
        missing_names = ["season", "week", "date", "home", "away"]
        missing = [name for name, col in zip(missing_names, required) if col is None]
        raise ValueError(f"games file missing required columns: {missing}")

    out = pd.DataFrame({
        "Season": pd.to_numeric(df[season_col], errors="coerce").astype("Int64"),
        "Week": pd.to_numeric(df[week_col], errors="coerce").astype("Int64"),
        "game_date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
        "home_team": df[home_col].map(normalize_team_name),
        "away_team": df[away_col].map(normalize_team_name),
    }).drop_duplicates()

    return out


def build_odds_key_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).copy()

    season_col = first_existing(df, ["Season", "season"])
    date_col = first_existing(df, ["game_date", "commence_time", "date"])
    home_col = first_existing(df, ["home_team", "HomeTeam", "home"])
    away_col = first_existing(df, ["away_team", "AwayTeam", "away"])

    if season_col is None and date_col is not None:
        df["Season"] = add_nfl_season_from_date(df, date_col)
        season_col = "Season"

    required = [season_col, date_col, home_col, away_col]
    if any(c is None for c in required):
        missing_names = ["season", "date", "home", "away"]
        missing = [name for name, col in zip(missing_names, required) if col is None]
        raise ValueError(f"odds file missing required columns: {missing}")

    out = pd.DataFrame({
        "Season": pd.to_numeric(df[season_col], errors="coerce").astype("Int64"),
        "game_date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
        "home_team": df[home_col].map(normalize_team_name),
        "away_team": df[away_col].map(normalize_team_name),
    }).drop_duplicates()

    return out


def compare_games_vs_odds():
    games_path = EXPORTS / "games_master.csv"
    odds_path = EXPORTS / "historical_odds" / "nfl_odds_full_merged.csv"

    if not games_path.exists() or not odds_path.exists():
        print("\nCannot run join-key comparison because a required file is missing.")
        return

    games = build_games_key_df(games_path)
    odds = build_odds_key_df(odds_path)

    # exact join on season/date/home/away
    merged = games.merge(
        odds.assign(has_odds=1),
        on=["Season", "game_date", "home_team", "away_team"],
        how="left",
    )

    coverage = (
        merged.groupby("Season", dropna=False)
        .agg(
            games=("home_team", "size"),
            matched=("has_odds", lambda s: int(s.fillna(0).sum())),
        )
        .reset_index()
    )
    coverage["match_rate"] = coverage["matched"] / coverage["games"]

    print("\n=== GAMES vs ODDS exact key coverage ===")
    print(coverage.to_string(index=False))

    misses = merged[merged["has_odds"].isna()].copy()
    miss_path = EXPORTS / "diagnostic_join_misses.csv"
    misses.to_csv(miss_path, index=False)
    print(f"\nSaved join misses: {miss_path}")

    print("\nSample misses:")
    sample_cols = ["Season", "game_date", "away_team", "home_team"]
    print(misses[sample_cols].head(25).to_string(index=False))


def main():
    print("Season coverage by file:")
    for path in FILES_TO_CHECK:
        print_file_seasons(path)

    compare_games_vs_odds()


if __name__ == "__main__":
    main()