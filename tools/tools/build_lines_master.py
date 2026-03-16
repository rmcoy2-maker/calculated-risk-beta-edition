from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ODDS_PATH = PROJECT_ROOT / "exports" / "historical_odds" / "nfl_historical_odds_combined.csv"
GAMES_PATH = PROJECT_ROOT / "exports" / "games_master.csv"

OUT_PATH = PROJECT_ROOT / "exports" / "lines_master.csv"
UNMATCHED_PATH = PROJECT_ROOT / "exports" / "lines_master_unmatched.csv"

ET = ZoneInfo("America/New_York")


TEAM_ALIASES = {
    "Arizona Cardinals": "Cardinals",
    "Atlanta Falcons": "Falcons",
    "Baltimore Ravens": "Ravens",
    "Buffalo Bills": "Bills",
    "Carolina Panthers": "Panthers",
    "Chicago Bears": "Bears",
    "Cincinnati Bengals": "Bengals",
    "Cleveland Browns": "Browns",
    "Dallas Cowboys": "Cowboys",
    "Denver Broncos": "Broncos",
    "Detroit Lions": "Lions",
    "Green Bay Packers": "Packers",
    "Houston Texans": "Texans",
    "Indianapolis Colts": "Colts",
    "Jacksonville Jaguars": "Jaguars",
    "Kansas City Chiefs": "Chiefs",
    "Las Vegas Raiders": "Raiders",
    "Los Angeles Chargers": "Chargers",
    "Los Angeles Rams": "Rams",
    "Miami Dolphins": "Dolphins",
    "Minnesota Vikings": "Vikings",
    "New England Patriots": "Patriots",
    "New Orleans Saints": "Saints",
    "New York Giants": "Giants",
    "New York Jets": "Jets",
    "Philadelphia Eagles": "Eagles",
    "Pittsburgh Steelers": "Steelers",
    "San Francisco 49ers": "49ers",
    "Seattle Seahawks": "Seahawks",
    "Tampa Bay Buccaneers": "Buccaneers",
    "Tennessee Titans": "Titans",
    "Washington Commanders": "Commanders",
    "Washington Football Team": "Commanders",
}


def normalize_team(name: str) -> str:
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def infer_season_from_et_date(dt_et: pd.Timestamp) -> int:
    year = dt_et.year
    if dt_et.month <= 2:
        return year - 1
    return year


def load_games_master() -> pd.DataFrame:
    gm = pd.read_csv(GAMES_PATH, low_memory=False)

    needed = ["game_id", "Season", "Week", "game_date", "away_team", "home_team"]
    missing = [c for c in needed if c not in gm.columns]
    if missing:
        raise ValueError(f"games_master.csv missing required columns: {missing}")

    gm = gm[needed].copy()
    gm["away_team"] = gm["away_team"].astype(str).str.strip().map(normalize_team)
    gm["home_team"] = gm["home_team"].astype(str).str.strip().map(normalize_team)
    gm["Season"] = pd.to_numeric(gm["Season"], errors="coerce").astype("Int64")
    gm["game_date"] = gm["game_date"].astype(str).str.strip()

    return gm


def load_odds() -> pd.DataFrame:
    df = pd.read_csv(ODDS_PATH, low_memory=False)

    required = [
        "requested_snapshot",
        "snapshot_timestamp",
        "event_id",
        "commence_time",
        "home_team",
        "away_team",
        "book_key",
        "book_title",
        "market_key",
        "outcome_name",
        "outcome_price",
        "outcome_point",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"historical odds file missing required columns: {missing}")

    df["home_team"] = df["home_team"].astype(str).str.strip().map(normalize_team)
    df["away_team"] = df["away_team"].astype(str).str.strip().map(normalize_team)

    dt_utc = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    dt_et = dt_utc.dt.tz_convert(ET)

    df["commence_time_et"] = dt_et
    df["game_date"] = dt_et.dt.strftime("%Y-%m-%d")
    df["season"] = dt_et.apply(infer_season_from_et_date).astype("Int64")

    return df


def map_market(market_key: str) -> str:
    market_key = str(market_key).strip().lower()
    if market_key == "h2h":
        return "moneyline"
    if market_key == "spreads":
        return "spread"
    if market_key == "totals":
        return "total"
    return market_key


def derive_side(row: pd.Series) -> str | None:
    market_key = str(row["market_key"]).strip().lower()
    outcome_name = str(row["outcome_name"]).strip()
    home_team = str(row["home_team"]).strip()
    away_team = str(row["away_team"]).strip()

    if market_key in {"h2h", "spreads"}:
        if normalize_team(outcome_name) == home_team:
            return "home"
        if normalize_team(outcome_name) == away_team:
            return "away"
        return None

    if market_key == "totals":
        low = outcome_name.lower()
        if low == "over":
            return "over"
        if low == "under":
            return "under"
        return None

    return None


def main() -> None:
    print("Loading games_master...")
    gm = load_games_master()

    print("Loading historical odds...")
    df = load_odds()

    print("Mapping sides and markets...")
    df["market"] = df["market_key"].map(map_market)
    df["side"] = df.apply(derive_side, axis=1)

    gm_join = gm[["game_id", "Season", "Week", "game_date", "away_team", "home_team"]].copy()

    merged = df.merge(
        gm_join,
        left_on=["season", "game_date", "away_team", "home_team"],
        right_on=["Season", "game_date", "away_team", "home_team"],
        how="left",
    )

    matched = merged["game_id"].notna().sum()
    total = len(merged)

    print(f"Matched rows: {matched:,} / {total:,}")
    print(f"Unmatched rows: {total - matched:,}")

    keep_cols = [
        "game_id",
        "event_id",
        "requested_snapshot",
        "snapshot_timestamp",
        "commence_time",
        "game_date",
        "Season",
        "Week",
        "away_team",
        "home_team",
        "book_key",
        "book_title",
        "market",
        "side",
        "outcome_price",
        "outcome_point",
    ]

    out = merged[keep_cols].copy()
    out = out.rename(
        columns={
            "Season": "season",
            "Week": "week",
            "outcome_price": "odds",
            "outcome_point": "point",
        }
    )

    out = out.sort_values(
        ["game_date", "game_id", "requested_snapshot", "book_key", "market", "side"],
        kind="stable",
    ).reset_index(drop=True)

    unmatched = merged.loc[merged["game_id"].isna(), [
        "event_id",
        "commence_time",
        "game_date",
        "season",
        "away_team",
        "home_team",
        "book_key",
        "market_key",
        "outcome_name",
        "outcome_price",
        "outcome_point",
    ]].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    unmatched.to_csv(UNMATCHED_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(out):,}")
    print(f"Saved unmatched: {UNMATCHED_PATH}")
    print(f"Unmatched rows: {len(unmatched):,}")

    print("\nSample:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()