from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:\Projects\calculated-risk-beta-edition")
EXPORTS = PROJECT_ROOT / "exports"

GAMES_PATH = EXPORTS / "games_master.csv"
ODDS_PATH = EXPORTS / "historical_odds" / "nfl_odds_full_merged.csv"
OUT_PATH = EXPORTS / "fort_knox_market_joined_moneyline_all_seasons.csv"
MISS_PATH = EXPORTS / "fort_knox_market_joined_moneyline_all_seasons_misses.csv"

TEAM_NORMALIZE = {
    "49ers": "San Francisco 49ers",
    "Bears": "Chicago Bears",
    "Bengals": "Cincinnati Bengals",
    "Bills": "Buffalo Bills",
    "Broncos": "Denver Broncos",
    "Browns": "Cleveland Browns",
    "Buccaneers": "Tampa Bay Buccaneers",
    "Cardinals": "Arizona Cardinals",
    "Chargers": "Los Angeles Chargers",
    "Chiefs": "Kansas City Chiefs",
    "Colts": "Indianapolis Colts",
    "Commanders": "Washington Commanders",
    "Cowboys": "Dallas Cowboys",
    "Dolphins": "Miami Dolphins",
    "Eagles": "Philadelphia Eagles",
    "Falcons": "Atlanta Falcons",
    "Football Team": "Washington Commanders",
    "Giants": "New York Giants",
    "Jaguars": "Jacksonville Jaguars",
    "Jets": "New York Jets",
    "Lions": "Detroit Lions",
    "Packers": "Green Bay Packers",
    "Panthers": "Carolina Panthers",
    "Patriots": "New England Patriots",
    "Raiders": "Las Vegas Raiders",
    "Rams": "Los Angeles Rams",
    "Ravens": "Baltimore Ravens",
    "Redskins": "Washington Commanders",
    "Saints": "New Orleans Saints",
    "Seahawks": "Seattle Seahawks",
    "Steelers": "Pittsburgh Steelers",
    "Texans": "Houston Texans",
    "Titans": "Tennessee Titans",
    "Vikings": "Minnesota Vikings",
}


def normalize_team(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return TEAM_NORMALIZE.get(s, s)


def nfl_season_from_datetime(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce", utc=True)
    season = dt.dt.year.astype("Int64")
    jan_feb = dt.dt.month.isin([1, 2])
    season = season.where(~jan_feb, season - 1)
    return season


def build_moneyline_snapshot_table(odds: pd.DataFrame) -> pd.DataFrame:
    ml = odds.copy()

    ml["commence_time"] = pd.to_datetime(ml["commence_time"], errors="coerce", utc=True)
    ml["Season"] = nfl_season_from_datetime(ml["commence_time"])
    ml["game_date"] = ml["commence_time"].dt.strftime("%Y-%m-%d")

    ml["home_team"] = ml["home_team"].map(normalize_team)
    ml["away_team"] = ml["away_team"].map(normalize_team)

    ml = ml[ml["market_key"].astype(str).str.lower() == "h2h"].copy()

    # Remove rows without a usable price/outcome
    ml["outcome_price"] = pd.to_numeric(ml["outcome_price"], errors="coerce")
    ml = ml.dropna(subset=["outcome_price", "outcome_name", "book_key", "commence_time"]).copy()

    # Normalize outcome side
    ml["outcome_name_norm"] = ml["outcome_name"].astype(str).str.strip()
    ml["snapshot_timestamp"] = pd.to_datetime(ml["snapshot_timestamp"], errors="coerce", utc=True)

    # Keep earliest and latest price per book / game / side
    key_cols = ["Season", "game_date", "away_team", "home_team", "book_key", "outcome_name_norm"]

    open_rows = (
        ml.sort_values("snapshot_timestamp", ascending=True)
        .drop_duplicates(subset=key_cols, keep="first")
        .copy()
    )
    close_rows = (
        ml.sort_values("snapshot_timestamp", ascending=False)
        .drop_duplicates(subset=key_cols, keep="first")
        .copy()
    )

    open_rows = open_rows.rename(columns={"outcome_price": "open_price"})
    close_rows = close_rows.rename(columns={"outcome_price": "close_price"})

    merged = open_rows[key_cols + ["open_price"]].merge(
        close_rows[key_cols + ["close_price"]],
        on=key_cols,
        how="outer",
    )

    # collapse across books to median market price
    market = (
        merged.groupby(["Season", "game_date", "away_team", "home_team", "outcome_name_norm"], dropna=False)
        .agg(
            open_price=("open_price", "median"),
            close_price=("close_price", "median"),
            books_used=("outcome_name_norm", "size"),
        )
        .reset_index()
    )

    # Pivot to game level prices for both teams
    away_side = market.rename(
        columns={
            "outcome_name_norm": "selected_team",
            "open_price": "away_open_price_raw",
            "close_price": "away_close_price_raw",
            "books_used": "away_books_used_raw",
        }
    )
    home_side = market.rename(
        columns={
            "outcome_name_norm": "selected_team",
            "open_price": "home_open_price_raw",
            "close_price": "home_close_price_raw",
            "books_used": "home_books_used_raw",
        }
    )

    # We keep one row per game after mapping selected_team to home/away
    rows = []

    game_groups = market.groupby(["Season", "game_date", "away_team", "home_team"], dropna=False)
    for (season, game_date, away_team, home_team), g in game_groups:
        out = {
            "Season": season,
            "game_date": game_date,
            "away_team": away_team,
            "home_team": home_team,
        }

        away_match = g[g["outcome_name_norm"] == away_team]
        home_match = g[g["outcome_name_norm"] == home_team]

        if not away_match.empty:
            out["away_open_price"] = away_match["open_price"].iloc[0]
            out["away_close_price"] = away_match["close_price"].iloc[0]
            out["away_books_used"] = away_match["books_used"].iloc[0]
        else:
            out["away_open_price"] = np.nan
            out["away_close_price"] = np.nan
            out["away_books_used"] = np.nan

        if not home_match.empty:
            out["home_open_price"] = home_match["open_price"].iloc[0]
            out["home_close_price"] = home_match["close_price"].iloc[0]
            out["home_books_used"] = home_match["books_used"].iloc[0]
        else:
            out["home_open_price"] = np.nan
            out["home_close_price"] = np.nan
            out["home_books_used"] = np.nan

        rows.append(out)

    return pd.DataFrame(rows)


def american_to_implied_prob(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)

    pos = s > 0
    neg = s < 0

    out.loc[pos] = 100.0 / (s.loc[pos] + 100.0)
    out.loc[neg] = s.loc[neg].abs() / (s.loc[neg].abs() + 100.0)
    return out


def main() -> None:
    print("Loading games...")
    games = pd.read_csv(GAMES_PATH, low_memory=False)

    print("Loading merged odds...")
    odds = pd.read_csv(ODDS_PATH, low_memory=False)

    games["Season"] = pd.to_numeric(games["Season"], errors="coerce").astype("Int64")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    games["home_team"] = games["home_team"].map(normalize_team)
    games["away_team"] = games["away_team"].map(normalize_team)

    print("Building game-level moneyline table...")
    market = build_moneyline_snapshot_table(odds)

    print("Joining games to market...")
    joined = games.merge(
        market,
        on=["Season", "game_date", "away_team", "home_team"],
        how="left",
    )

    # Coverage diagnostics
    joined["has_market"] = (
        joined["away_open_price"].notna()
        | joined["away_close_price"].notna()
        | joined["home_open_price"].notna()
        | joined["home_close_price"].notna()
    ).astype(int)

    coverage = (
        joined.groupby("Season", dropna=False)
        .agg(
            games=("game_id", "size"),
            matched=("has_market", "sum"),
        )
        .reset_index()
    )
    coverage["match_rate"] = coverage["matched"] / coverage["games"]

    print("\nCoverage by season:")
    print(coverage.to_string(index=False))

    misses = joined[joined["has_market"] == 0].copy()
    misses.to_csv(MISS_PATH, index=False)

    # Optional convenience implied probs
    joined["away_open_prob"] = american_to_implied_prob(joined["away_open_price"])
    joined["away_close_prob"] = american_to_implied_prob(joined["away_close_price"])
    joined["home_open_prob"] = american_to_implied_prob(joined["home_open_price"])
    joined["home_close_prob"] = american_to_implied_prob(joined["home_close_price"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(OUT_PATH, index=False)

    print("\nSaved:")
    print(OUT_PATH)
    print(MISS_PATH)
    print(f"Rows: {len(joined):,}")


if __name__ == "__main__":
    main()