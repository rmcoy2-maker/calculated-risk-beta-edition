from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "exports" / "games_master.csv"
OUTPUT_PATH = PROJECT_ROOT / "exports" / "games_master_recent_form.csv"


TEAM_STATS = [
    "home_score",
    "away_score",
    "home_total_yards",
    "away_total_yards",
    "home_turnovers",
    "away_turnovers",
    "home_sacks",
    "away_sacks",
    "home_first_downs",
    "away_first_downs",
]


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_team_game_rows(games: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in games.iterrows():
        game_id = r["game_id"]
        season = r["Season"]
        week = r["Week"]
        game_date = r["game_date"]

        home_team = r["home_team"]
        away_team = r["away_team"]

        home_score = r.get("home_score", np.nan)
        away_score = r.get("away_score", np.nan)

        home_total_yards = r.get("home_total_yards", np.nan)
        away_total_yards = r.get("away_total_yards", np.nan)

        home_turnovers = r.get("home_turnovers", np.nan)
        away_turnovers = r.get("away_turnovers", np.nan)

        home_sacks = r.get("home_sacks", np.nan)
        away_sacks = r.get("away_sacks", np.nan)

        home_first_downs = r.get("home_first_downs", np.nan)
        away_first_downs = r.get("away_first_downs", np.nan)

        rows.append(
            {
                "game_id": game_id,
                "Season": season,
                "Week": week,
                "game_date": game_date,
                "team": home_team,
                "opponent": away_team,
                "is_home": 1,
                "points_for": home_score,
                "points_against": away_score,
                "total_yards_for": home_total_yards,
                "total_yards_against": away_total_yards,
                "turnovers_for": home_turnovers,
                "turnovers_against": away_turnovers,
                "sacks_for": home_sacks,
                "sacks_against": away_sacks,
                "first_downs_for": home_first_downs,
                "first_downs_against": away_first_downs,
                "win": 1 if pd.notna(home_score) and pd.notna(away_score) and home_score > away_score else 0,
            }
        )

        rows.append(
            {
                "game_id": game_id,
                "Season": season,
                "Week": week,
                "game_date": game_date,
                "team": away_team,
                "opponent": home_team,
                "is_home": 0,
                "points_for": away_score,
                "points_against": home_score,
                "total_yards_for": away_total_yards,
                "total_yards_against": home_total_yards,
                "turnovers_for": away_turnovers,
                "turnovers_against": home_turnovers,
                "sacks_for": away_sacks,
                "sacks_against": home_sacks,
                "first_downs_for": away_first_downs,
                "first_downs_against": home_first_downs,
                "win": 1 if pd.notna(home_score) and pd.notna(away_score) and away_score > home_score else 0,
            }
        )

    out = pd.DataFrame(rows)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.sort_values(["team", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    return out


def add_rolling_features(team_games: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "points_for",
        "points_against",
        "total_yards_for",
        "total_yards_against",
        "turnovers_for",
        "turnovers_against",
        "sacks_for",
        "sacks_against",
        "first_downs_for",
        "first_downs_against",
        "win",
    ]

    def per_team(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["game_date", "game_id"], kind="stable").copy()

        for m in metrics:
            prev = g[m].shift(1)

            g[f"{m}_avg_season_prior"] = prev.expanding().mean()
            g[f"{m}_avg_last3"] = prev.rolling(3, min_periods=1).mean()
            g[f"{m}_avg_last5"] = prev.rolling(5, min_periods=1).mean()
            g[f"{m}_std_last5"] = prev.rolling(5, min_periods=2).std()

            g[f"{m}_trend_last3_vs_season"] = g[f"{m}_avg_last3"] - g[f"{m}_avg_season_prior"]
            g[f"{m}_trend_last5_vs_season"] = g[f"{m}_avg_last5"] - g[f"{m}_avg_season_prior"]

        g["games_played_prior"] = np.arange(len(g))
        return g

    return (
        team_games.groupby("team", group_keys=False)
        .apply(per_team)
        .reset_index(drop=True)
    )


def merge_back_to_games(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    home = team_games.add_prefix("home_").rename(columns={"home_game_id": "game_id"})
    away = team_games.add_prefix("away_").rename(columns={"away_game_id": "game_id"})

    home_keep = [c for c in home.columns if c == "game_id" or c.startswith("home_")]
    away_keep = [c for c in away.columns if c == "game_id" or c.startswith("away_")]

    out = games.merge(home[home_keep], on="game_id", how="left")
    out = out.merge(away[away_keep], on="game_id", how="left")

    diff_pairs = [
        "points_for_avg_last3",
        "points_against_avg_last3",
        "points_for_avg_last5",
        "points_against_avg_last5",
        "total_yards_for_avg_last3",
        "total_yards_against_avg_last3",
        "total_yards_for_avg_last5",
        "total_yards_against_avg_last5",
        "turnovers_for_avg_last3",
        "turnovers_against_avg_last3",
        "turnovers_for_avg_last5",
        "turnovers_against_avg_last5",
        "sacks_for_avg_last3",
        "sacks_against_avg_last3",
        "sacks_for_avg_last5",
        "sacks_against_avg_last5",
        "first_downs_for_avg_last3",
        "first_downs_against_avg_last3",
        "first_downs_for_avg_last5",
        "first_downs_against_avg_last5",
        "win_avg_last3",
        "win_avg_last5",
        "points_for_trend_last3_vs_season",
        "points_against_trend_last3_vs_season",
        "points_for_trend_last5_vs_season",
        "points_against_trend_last5_vs_season",
        "total_yards_for_trend_last3_vs_season",
        "total_yards_against_trend_last3_vs_season",
        "total_yards_for_trend_last5_vs_season",
        "total_yards_against_trend_last5_vs_season",
        "win_trend_last3_vs_season",
        "win_trend_last5_vs_season",
        "games_played_prior",
    ]

    for base in diff_pairs:
        hc = f"home_{base}"
        ac = f"away_{base}"
        if hc in out.columns and ac in out.columns:
            out[f"diff_{base}"] = pd.to_numeric(out[home_keep and hc], errors="coerce") - pd.to_numeric(out[ac], errors="coerce")

    return out


def main() -> None:
    print("Loading games_master...")
    games = pd.read_csv(INPUT_PATH, low_memory=False)

    needed = ["game_id", "Season", "Week", "game_date", "home_team", "away_team"]
    missing = [c for c in needed if c not in games.columns]
    if missing:
        raise ValueError(f"games_master.csv missing required columns: {missing}")

    games = safe_numeric(games, TEAM_STATS)
    games["Season"] = pd.to_numeric(games["Season"], errors="coerce").astype("Int64")
    games["Week"] = pd.to_numeric(games["Week"], errors="coerce").astype("Int64")

    print("Building team-game rows...")
    team_games = build_team_game_rows(games)

    print("Adding rolling recent-form features...")
    team_games = add_rolling_features(team_games)

    print("Merging back to game level...")
    out = merge_back_to_games(games, team_games)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(out):,}")
    print("\nSample recent-form columns:")
    sample_cols = [c for c in out.columns if "last3" in c or "last5" in c][:25]
    print(sample_cols)


if __name__ == "__main__":
    main()
