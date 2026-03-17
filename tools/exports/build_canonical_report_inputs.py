from pathlib import Path
import argparse
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EXPORTS = ROOT / "exports"
CANONICAL = EXPORTS / "canonical"


def read_csv_safe(path):
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()


def build_games(season, week):
    src = EXPORTS / "games_master.csv"
    df = read_csv_safe(src)

    if df.empty:
        return df

    if "season" in df.columns:
        df = df[df["season"] == season]

    if "week" in df.columns:
        df = df[df["week"] == week]

    rename = {
        "away": "away_team",
        "home": "home_team",
        "AwayTeam": "away_team",
        "HomeTeam": "home_team",
    }

    df = df.rename(columns=rename)

    if "matchup" not in df.columns:
        if "away_team" in df.columns and "home_team" in df.columns:
            df["matchup"] = df["away_team"] + " @ " + df["home_team"]

    return df


def build_edges(season, week):
    src = EXPORTS / "edges_master.csv"
    df = read_csv_safe(src)

    if df.empty:
        return df

    if "season" in df.columns:
        df = df[df["season"] == season]

    if "week" in df.columns:
        df = df[df["week"] == week]

    rename = {
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

    if "matchup" not in df.columns:
        if "away_team" in df.columns and "home_team" in df.columns:
            df["matchup"] = df["away_team"] + " @ " + df["home_team"]

    return df


def build_markets(season, week):
    src = EXPORTS / "lines_master.csv"
    df = read_csv_safe(src)

    if df.empty:
        return df

    if "season" in df.columns:
        df = df[df["season"] == season]

    if "week" in df.columns:
        df = df[df["week"] == week]

    rename = {
        "spread": "market_spread",
        "total": "market_total",
    }

    df = df.rename(columns=rename)

    if "matchup" not in df.columns:
        if "away_team" in df.columns and "home_team" in df.columns:
            df["matchup"] = df["away_team"] + " @ " + df["home_team"]

    return df


def build_scores(season, week):
    src = EXPORTS / "scores_master.csv"
    df = read_csv_safe(src)

    if df.empty:
        return df

    if "season" in df.columns:
        df = df[df["season"] == season]

    if "week" in df.columns:
        df = df[df["week"] == week]

    rename = {
        "away": "away_team",
        "home": "home_team",
    }

    df = df.rename(columns=rename)

    if "matchup" not in df.columns:
        if "away_team" in df.columns and "home_team" in df.columns:
            df["matchup"] = df["away_team"] + " @ " + df["home_team"]

    return df


def build_parlays(season, week):
    src = EXPORTS / "data" / "parlay_scores.csv"
    df = read_csv_safe(src)

    if df.empty:
        return df

    if "season" in df.columns:
        df = df[df["season"] == season]

    if "week" in df.columns:
        df = df[df["week"] == week]

    rename = {
        "score": "parlay_score",
        "play": "label",
        "leg": "label",
    }

    df = df.rename(columns=rename)

    if "parlay_score" not in df.columns:
        df["parlay_score"] = 0

    return df


def save(df, name, season, week):
    CANONICAL.mkdir(exist_ok=True)

    path = CANONICAL / f"{name}_{season}_w{week}.csv"

    df.to_csv(path, index=False)

    print(f"[OK] {path} ({len(df)} rows)")


def main():
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