# tools/build_play_features.py

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "exports" / "plays_master.csv"
OUTPUT_DIR = PROJECT_ROOT / "exports"
GAME_OUTPUT = OUTPUT_DIR / "game_features_from_plays.csv"
TEAM_OUTPUT = OUTPUT_DIR / "team_features_from_plays.csv"


def ensure_input() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}\n"
            "Run tools/build_plays_master.py first."
        )


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def contains_any(series: pd.Series, patterns: list[str]) -> pd.Series:
    pattern = "|".join(f"(?:{p})" for p in patterns)
    return (
        series.fillna("")
        .astype(str)
        .str.contains(pattern, case=False, regex=True, na=False)
    )


def build_flags(df: pd.DataFrame) -> pd.DataFrame:
    desc = df["PlayDescription"].fillna("").astype(str)
    outcome = df["PlayOutcome"].fillna("").astype(str)
    start = df["PlayStart"].fillna("").astype(str)

    play_text = desc + " " + outcome

    df["is_pass"] = contains_any(
        play_text,
        [
            r"\bpass\b",
            r"\bpasses\b",
            r"\bsacked\b",
            r"\bsack\b",
            r"\bincomplete\b",
            r"\bintercept",
        ],
    ).astype(int)

    df["is_rush"] = contains_any(
        play_text,
        [
            r"\bleft tackle\b",
            r"\bright tackle\b",
            r"\bleft guard\b",
            r"\bright guard\b",
            r"\bup the middle\b",
            r"\brushed\b",
            r"\brun\b",
            r"\bruns\b",
            r"\bscramble\b",
        ],
    ).astype(int)

    df["is_punt"] = contains_any(play_text, [r"\bpunt\b"]).astype(int)
    df["is_field_goal"] = contains_any(play_text, [r"\bfield goal\b"]).astype(int)
    df["is_kickoff"] = contains_any(play_text, [r"\bkickoff\b"]).astype(int)
    df["is_extra_point"] = contains_any(play_text, [r"\bextra point\b", r"\bpat\b"]).astype(int)

    df["is_turnover"] = contains_any(
        play_text,
        [r"\bintercept", r"\bfumble\b", r"\bfumbled\b", r"\blost fumble\b"],
    ).astype(int)

    df["is_sack"] = contains_any(play_text, [r"\bsacked\b", r"\bsack\b"]).astype(int)
    df["is_penalty"] = contains_any(play_text, [r"\bpenalty\b"]).astype(int)
    df["is_first_down"] = contains_any(play_text, [r"\bfirst down\b"]).astype(int)
    df["is_touchdown"] = contains_any(play_text, [r"\btouchdown\b"]).astype(int)
    df["is_scoring_play_flag"] = safe_num(df["IsScoringPlay"]).fillna(0).astype(int)
    df["is_scoring_drive_flag"] = safe_num(df["IsScoringDrive"]).fillna(0).astype(int)

    yardline_num = start.str.extract(r"(\d{1,2})")[0]
    yardline_num = pd.to_numeric(yardline_num, errors="coerce")

    possession = df["TeamWithPossession"].fillna("").astype(str).str.strip().str.lower()
    start_norm = start.str.lower().str.strip()

    own_side = [
        s.startswith(p) if p else False
        for s, p in zip(start_norm, possession)
    ]

    df["yardline_num"] = yardline_num
    df["is_red_zone"] = np.where(
        yardline_num.notna(),
        np.where(own_side, yardline_num <= 20, (100 - yardline_num) <= 20),
        0,
    ).astype(int)

    df["is_offensive_play"] = (
        (df["is_pass"] == 1) | (df["is_rush"] == 1) | (df["is_sack"] == 1)
    ).astype(int)

    df["is_explosive_pass"] = (
        contains_any(play_text, [r"\bpass\b", r"\bpasses\b", r"\bincomplete\b"]) &
        contains_any(play_text, [r"\b2\d yard\b", r"\b3\d yard\b", r"\b4\d yard\b", r"\b5\d yard\b", r"\b6\d yard\b", r"\b7\d yard\b", r"\b8\d yard\b", r"\b9\d yard\b"])
    ).astype(int)

    df["is_explosive_rush"] = (
        contains_any(play_text, [r"\brun\b", r"\bruns\b", r"\brushed\b", r"\bscramble\b"]) &
        contains_any(play_text, [r"\b1[0-9] yard\b", r"\b2\d yard\b", r"\b3\d yard\b", r"\b4\d yard\b", r"\b5\d yard\b", r"\b6\d yard\b", r"\b7\d yard\b", r"\b8\d yard\b", r"\b9\d yard\b"])
    ).astype(int)

    return df


def build_team_game_features(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "game_id",
        "Season",
        "Week",
        "game_date",
        "game_date_str",
        "AwayTeam",
        "HomeTeam",
        "season_type",
        "week_sort",
    ]

    team_game = (
        df.groupby(keys + ["TeamWithPossession"], dropna=False)
        .agg(
            total_plays=("play_sequence", "size"),
            offensive_plays=("is_offensive_play", "sum"),
            pass_plays=("is_pass", "sum"),
            rush_plays=("is_rush", "sum"),
            sacks=("is_sack", "sum"),
            punts=("is_punt", "sum"),
            field_goal_plays=("is_field_goal", "sum"),
            turnovers=("is_turnover", "sum"),
            penalties=("is_penalty", "sum"),
            first_downs=("is_first_down", "sum"),
            touchdowns=("is_touchdown", "sum"),
            scoring_plays=("is_scoring_play_flag", "sum"),
            scoring_drives=("is_scoring_drive_flag", "max"),
            red_zone_plays=("is_red_zone", "sum"),
            explosive_passes=("is_explosive_pass", "sum"),
            explosive_runs=("is_explosive_rush", "sum"),
            drive_count=("DriveNumber", pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"TeamWithPossession": "team"})
    )

    team_game["pass_rate"] = np.where(
        team_game["offensive_plays"] > 0,
        team_game["pass_plays"] / team_game["offensive_plays"],
        0.0,
    )
    team_game["rush_rate"] = np.where(
        team_game["offensive_plays"] > 0,
        team_game["rush_plays"] / team_game["offensive_plays"],
        0.0,
    )
    team_game["explosive_play_rate"] = np.where(
        team_game["offensive_plays"] > 0,
        (team_game["explosive_passes"] + team_game["explosive_runs"]) / team_game["offensive_plays"],
        0.0,
    )
    team_game["scoring_drive_rate"] = np.where(
        team_game["drive_count"] > 0,
        team_game["scoring_drives"] / team_game["drive_count"],
        0.0,
    )

    team_game["home_or_away"] = np.where(
        team_game["team"] == team_game["HomeTeam"],
        "home",
        np.where(team_game["team"] == team_game["AwayTeam"], "away", "unknown"),
    )

    return team_game


def pivot_to_game_features(team_game: pd.DataFrame) -> pd.DataFrame:
    home = (
        team_game[team_game["home_or_away"] == "home"]
        .copy()
        .add_prefix("home_")
        .rename(
            columns={
                "home_game_id": "game_id",
                "home_Season": "Season",
                "home_Week": "Week",
                "home_game_date": "game_date",
                "home_game_date_str": "game_date_str",
                "home_AwayTeam": "AwayTeam",
                "home_HomeTeam": "HomeTeam",
                "home_season_type": "season_type",
                "home_week_sort": "week_sort",
            }
        )
    )

    away = (
        team_game[team_game["home_or_away"] == "away"]
        .copy()
        .add_prefix("away_")
        .rename(
            columns={
                "away_game_id": "game_id",
                "away_Season": "Season",
                "away_Week": "Week",
                "away_game_date": "game_date",
                "away_game_date_str": "game_date_str",
                "away_AwayTeam": "AwayTeam",
                "away_HomeTeam": "HomeTeam",
                "away_season_type": "season_type",
                "away_week_sort": "week_sort",
            }
        )
    )

    game = home.merge(
        away,
        on=[
            "game_id",
            "Season",
            "Week",
            "game_date",
            "game_date_str",
            "AwayTeam",
            "HomeTeam",
            "season_type",
            "week_sort",
        ],
        how="outer",
    )

    game["combined_offensive_plays"] = game["home_offensive_plays"].fillna(0) + game["away_offensive_plays"].fillna(0)
    game["combined_total_plays"] = game["home_total_plays"].fillna(0) + game["away_total_plays"].fillna(0)
    game["combined_drive_count"] = game["home_drive_count"].fillna(0) + game["away_drive_count"].fillna(0)
    game["combined_turnovers"] = game["home_turnovers"].fillna(0) + game["away_turnovers"].fillna(0)
    game["combined_penalties"] = game["home_penalties"].fillna(0) + game["away_penalties"].fillna(0)
    game["combined_touchdowns"] = game["home_touchdowns"].fillna(0) + game["away_touchdowns"].fillna(0)
    game["combined_scoring_plays"] = game["home_scoring_plays"].fillna(0) + game["away_scoring_plays"].fillna(0)
    game["combined_red_zone_plays"] = game["home_red_zone_plays"].fillna(0) + game["away_red_zone_plays"].fillna(0)
    game["combined_explosive_passes"] = game["home_explosive_passes"].fillna(0) + game["away_explosive_passes"].fillna(0)
    game["combined_explosive_runs"] = game["home_explosive_runs"].fillna(0) + game["away_explosive_runs"].fillna(0)

    game["combined_explosive_play_rate"] = np.where(
        game["combined_offensive_plays"] > 0,
        (game["combined_explosive_passes"] + game["combined_explosive_runs"]) / game["combined_offensive_plays"],
        0.0,
    )

    game["combined_scoring_drive_rate"] = np.where(
        game["combined_drive_count"] > 0,
        (game["home_scoring_drives"].fillna(0) + game["away_scoring_drives"].fillna(0)) / game["combined_drive_count"],
        0.0,
    )

    game["play_volume_diff"] = game["home_offensive_plays"].fillna(0) - game["away_offensive_plays"].fillna(0)
    game["pass_rate_diff"] = game["home_pass_rate"].fillna(0) - game["away_pass_rate"].fillna(0)
    game["rush_rate_diff"] = game["home_rush_rate"].fillna(0) - game["away_rush_rate"].fillna(0)
    game["turnover_diff"] = game["home_turnovers"].fillna(0) - game["away_turnovers"].fillna(0)

    ordered_cols = [
        "game_id",
        "Season",
        "Week",
        "game_date",
        "game_date_str",
        "season_type",
        "week_sort",
        "AwayTeam",
        "HomeTeam",
    ]
    remaining = [c for c in game.columns if c not in ordered_cols]
    return game[ordered_cols + remaining].sort_values(
        ["Season", "week_sort", "game_date", "AwayTeam", "HomeTeam"],
        kind="stable",
    )


def main() -> int:
    ensure_input()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    df = build_flags(df)

    team_game = build_team_game_features(df)
    game = pivot_to_game_features(team_game)

    team_game.to_csv(TEAM_OUTPUT, index=False)
    game.to_csv(GAME_OUTPUT, index=False)

    print(f"Wrote team features: {TEAM_OUTPUT}")
    print(f"Wrote game features: {GAME_OUTPUT}")
    print(f"Games: {game['game_id'].nunique():,}")
    print(f"Team-game rows: {len(team_game):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())