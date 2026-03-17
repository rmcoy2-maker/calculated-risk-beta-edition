from pathlib import Path
import argparse
import pandas as pd


CANONICAL = Path(__file__).resolve().parent
EXPORTS = CANONICAL.parent
DATA_DIR = EXPORTS / "data"

GAMES_MASTER = EXPORTS / "games_master.csv"
EDGES_MASTER = EXPORTS / "edges_master.csv"
LINES_MASTER = EXPORTS / "lines_master.csv"
SCORES_MASTER = EXPORTS / "scores_master.csv"
PARLAY_SCORES = DATA_DIR / "parlay_scores.csv"


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed reading {path}: {e}")
        return pd.DataFrame()


def _first_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None

    value = df[col]

    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            return pd.Series(index=df.index, dtype="object")
        value = value.iloc[:, 0]

    return value


def _ensure_matchup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "matchup" in df.columns:
        return df

    away = _first_series(df, "away_team")
    home = _first_series(df, "home_team")

    if away is None or home is None:
        return df

    df = df.copy()
    df["matchup"] = (
        away.fillna("").astype(str).str.strip()
        + " @ "
        + home.fillna("").astype(str).str.strip()
    )

    return df


def _filter_season_week(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce")
        out = out[out["season"] == season]

    if "week" in out.columns:
        out["week"] = pd.to_numeric(out["week"], errors="coerce")
        out = out[out["week"] == week]

    return out


def build_games(season: int, week: int) -> pd.DataFrame:
    df = read_csv_safe(GAMES_MASTER)

    if df.empty:
        return df

    df = _filter_season_week(df, season, week)

    rename = {
        "away": "away_team",
        "home": "home_team",
        "AwayTeam": "away_team",
        "HomeTeam": "home_team",
    }
    df = df.rename(columns=rename)

    df = _ensure_matchup(df)
    return df


def build_edges(season: int, week: int) -> pd.DataFrame:
    df = read_csv_safe(EDGES_MASTER)

    if df.empty:
        return df

    df = _filter_season_week(df, season, week)

    rename = {
        "away": "away_team",
        "home": "home_team",
        "AwayTeam": "away_team",
        "HomeTeam": "home_team",
        "edge": "edge_score",
        "ev": "edge_score",
        "conf": "confidence",
        "bet": "label",
        "play": "label",
    }
    df = df.rename(columns=rename)

    if "confidence" not in df.columns:
        df["confidence"] = 50

    if "edge_score" not in df.columns:
        df["edge_score"] = 0

    df = _ensure_matchup(df)
    return df


def build_markets(season: int, week: int) -> pd.DataFrame:
    df = read_csv_safe(LINES_MASTER)

    if df.empty:
        return df

    df = _filter_season_week(df, season, week)

    rename = {
        "away": "away_team",
        "home": "home_team",
        "AwayTeam": "away_team",
        "HomeTeam": "home_team",
        "spread": "market_spread",
        "total": "market_total",
    }
    df = df.rename(columns=rename)

    df = _ensure_matchup(df)
    return df


def build_scores(season: int, week: int) -> pd.DataFrame:
    df = read_csv_safe(SCORES_MASTER)

    if df.empty:
        return df

    df = _filter_season_week(df, season, week)

    rename = {
        "away": "away_team",
        "home": "home_team",
        "AwayTeam": "away_team",
        "HomeTeam": "home_team",
    }
    df = df.rename(columns=rename)

    df = _ensure_matchup(df)
    return df


def build_parlays(season: int, week: int) -> pd.DataFrame:
    df = read_csv_safe(PARLAY_SCORES)

    if df.empty:
        return df

    df = _filter_season_week(df, season, week)

    rename = {
        "score": "parlay_score",
        "play": "label",
        "leg": "label",
    }
    df = df.rename(columns=rename)

    if "parlay_score" not in df.columns:
        df["parlay_score"] = 0

    return df


def save(df: pd.DataFrame, name: str, season: int, week: int) -> None:
    CANONICAL.mkdir(parents=True, exist_ok=True)

    path = CANONICAL / f"{name}_{season}_w{week}.csv"
    df.to_csv(path, index=False)

    print(f"[OK] {path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    season = args.season
    week = args.week

    print("Building canonical report inputs")

    games = build_games(season, week)
    edges = build_edges(season, week)
    markets = build_markets(season, week)
    scores = build_scores(season, week)
    parlays = build_parlays(season, week)

    save(games, "cr_games", season, week)
    save(edges, "cr_edges", season, week)
    save(markets, "cr_markets", season, week)
    save(scores, "cr_scores", season, week)
    save(parlays, "cr_parlay_scores", season, week)


if __name__ == "__main__":
    main()