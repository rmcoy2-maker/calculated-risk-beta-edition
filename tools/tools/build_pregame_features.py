from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "exports"

DEFAULT_INPUT = EXPORTS_DIR / "games_master.csv"
DEFAULT_OUTPUT = EXPORTS_DIR / "pregame_features.csv"

ROLL_WINDOWS = [3, 5]


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def pick_best_prefixed_column(df: pd.DataFrame, base_name: str, side_prefix: str) -> str | None:
    candidates = [
        f"{side_prefix}{base_name}",
        f"{side_prefix}{base_name}_x",
        f"{side_prefix}{base_name}_y",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_team_stat_bases(df: pd.DataFrame) -> list[str]:
    preferred_bases = [
        "total_plays",
        "offensive_plays",
        "pass_plays",
        "rush_plays",
        "sacks",
        "punts",
        "field_goal_plays",
        "turnovers",
        "penalties",
        "first_downs",
        "scoring_drives",
        "red_zone_plays",
        "explosive_passes",
        "explosive_runs",
        "drive_count",
        "pass_rate",
        "rush_rate",
        "explosive_play_rate",
        "scoring_drive_rate",
    ]

    keep = []
    for base in preferred_bases:
        h = pick_best_prefixed_column(df, base, "home_")
        a = pick_best_prefixed_column(df, base, "away_")
        if h is not None and a is not None:
            keep.append(base)

    return keep


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def load_games(path: Path, include_preseason: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).copy()

    required = [
        "game_id",
        "Season",
        "Week",
        "game_date",
        "away_team",
        "home_team",
        "home_score",
        "away_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"games_master.csv missing required columns: {missing}")

    df["game_date"] = safe_to_datetime(df["game_date"])
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype("Int64")

    if not include_preseason:
        df = df[
            ~df["Week"].astype(str).str.contains("Preseason|Hall Of Fame", case=False, na=False)
        ].copy()

    # ensure target/result columns exist if possible
    if "margin" not in df.columns:
        df["margin"] = pd.to_numeric(df["home_score"], errors="coerce") - pd.to_numeric(df["away_score"], errors="coerce")
    if "total_points" not in df.columns:
        df["total_points"] = pd.to_numeric(df["home_score"], errors="coerce") + pd.to_numeric(df["away_score"], errors="coerce")
    if "home_win" not in df.columns:
        df["home_win"] = (pd.to_numeric(df["home_score"], errors="coerce") > pd.to_numeric(df["away_score"], errors="coerce")).astype("Int64")
    if "away_win" not in df.columns:
        df["away_win"] = (pd.to_numeric(df["away_score"], errors="coerce") > pd.to_numeric(df["home_score"], errors="coerce")).astype("Int64")

    df = df.sort_values(["Season", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    return df


def build_team_long(gm: pd.DataFrame) -> pd.DataFrame:
    stat_bases = choose_team_stat_bases(gm)

    home = gm[["game_id", "Season", "Week", "game_date", "home_team", "away_team", "home_score", "away_score"]].copy()
    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "points_for",
            "away_score": "points_against",
        }
    )
    home["is_home"] = 1

    away = gm[["game_id", "Season", "Week", "game_date", "away_team", "home_team", "away_score", "home_score"]].copy()
    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "points_for",
            "home_score": "points_against",
        }
    )
    away["is_home"] = 0

    for base in stat_bases:
        hcol = pick_best_prefixed_column(gm, base, "home_")
        acol = pick_best_prefixed_column(gm, base, "away_")
        home[base] = pd.to_numeric(gm[hcol], errors="coerce")
        away[base] = pd.to_numeric(gm[acol], errors="coerce")

    team_long = pd.concat([home, away], ignore_index=True)

    # result columns for rolling priors
    team_long["margin"] = pd.to_numeric(team_long["points_for"], errors="coerce") - pd.to_numeric(team_long["points_against"], errors="coerce")
    team_long["win"] = (team_long["margin"] > 0).astype("Int64")
    team_long["total_points_game"] = pd.to_numeric(team_long["points_for"], errors="coerce") + pd.to_numeric(
        team_long["points_against"], errors="coerce"
    )

    team_long = team_long.sort_values(["team", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    return team_long


def add_rest_days(team_long: pd.DataFrame) -> pd.DataFrame:
    out = team_long.copy()
    out["prev_game_date"] = out.groupby("team", dropna=False)["game_date"].shift(1)
    out["rest_days"] = (out["game_date"] - out["prev_game_date"]).dt.days
    return out


def add_rolling_features(team_long: pd.DataFrame) -> pd.DataFrame:
    out = team_long.copy()

    stat_cols = [
        c
        for c in out.columns
        if c not in {"game_id", "Season", "Week", "game_date", "team", "opponent", "prev_game_date"}
        and pd.api.types.is_numeric_dtype(out[c])
    ]

    grouped = out.groupby("team", dropna=False, sort=False)

    out["games_played_prior"] = grouped.cumcount()

    # expanding mean using prior games only
    for col in stat_cols:
        shifted = grouped[col].shift(1)
        out[f"{col}_avg_season"] = (
            shifted.groupby(out["team"], dropna=False)
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )

    # rolling windows using prior games only
    for col in stat_cols:
        shifted = grouped[col].shift(1)
        for w in ROLL_WINDOWS:
            out[f"{col}_avg_last_{w}"] = (
                shifted.groupby(out["team"], dropna=False)
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # home/away split priors
    for col in stat_cols:
        home_prior = out[col].where(out["is_home"] == 1)
        away_prior = out[col].where(out["is_home"] == 0)

        out[f"{col}_avg_home_prior"] = (
            home_prior.groupby(out["team"], dropna=False)
            .shift(1)
            .groupby(out["team"], dropna=False)
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )

        out[f"{col}_avg_away_prior"] = (
            away_prior.groupby(out["team"], dropna=False)
            .shift(1)
            .groupby(out["team"], dropna=False)
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )

    return out


def build_pregame_table(team_long: pd.DataFrame, gm: pd.DataFrame) -> pd.DataFrame:
    keep_meta = ["game_id", "Season", "Week", "game_date", "team", "opponent"]

    rolling_cols = [
        c
        for c in team_long.columns
        if c in ["games_played_prior", "rest_days", "is_home"]
        or c.endswith("_avg_season")
        or "_avg_last_" in c
        or c.endswith("_avg_home_prior")
        or c.endswith("_avg_away_prior")
    ]

    team_pre = team_long[keep_meta + rolling_cols].copy()

    # Build home-team perspective rows
    home_pre = team_pre.rename(columns={"team": "home_team", "opponent": "away_team"}).copy()
    home_feature_cols = [
        c for c in home_pre.columns
        if c not in ["game_id", "Season", "Week", "game_date", "home_team", "away_team"]
    ]
    home_pre = home_pre.rename(columns={c: f"home_{c}" for c in home_feature_cols})

    # Build away-team perspective rows
    away_pre = team_pre.rename(columns={"team": "away_team", "opponent": "home_team"}).copy()
    away_feature_cols = [
        c for c in away_pre.columns
        if c not in ["game_id", "Season", "Week", "game_date", "away_team", "home_team"]
    ]
    away_pre = away_pre.rename(columns={c: f"away_{c}" for c in away_feature_cols})

    out = gm[
        [
            "game_id",
            "Season",
            "Week",
            "game_date",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "margin",
            "total_points",
            "away_win",
            "home_win",
        ]
    ].copy()

    out = out.merge(
        home_pre,
        on=["game_id", "Season", "Week", "game_date", "home_team", "away_team"],
        how="left",
    )

    out = out.merge(
        away_pre,
        on=["game_id", "Season", "Week", "game_date", "away_team", "home_team"],
        how="left",
    )

    # Build matchup diff features automatically
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    home_numeric = [c for c in numeric_cols if c.startswith("home_")]

    diff_count = 0
    for hcol in home_numeric:
        base = hcol[len("home_"):]
        acol = f"away_{base}"
        if acol in out.columns and pd.api.types.is_numeric_dtype(out[acol]):
            out[f"diff_{base}"] = pd.to_numeric(out[hcol], errors="coerce") - pd.to_numeric(out[acol], errors="coerce")
            diff_count += 1

    print(f"Built diff features: {diff_count}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pregame_features.csv from games_master.csv")
    parser.add_argument("--infile", default=str(DEFAULT_INPUT), help="Path to games_master.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Path to output pregame_features.csv")
    parser.add_argument(
        "--include-preseason",
        action="store_true",
        help="Include preseason / Hall Of Fame games",
    )
    args = parser.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.out)

    print("Loading games_master...")
    gm = load_games(infile, include_preseason=args.include_preseason)

    print("Building team-game long table...")
    team_long = build_team_long(gm)

    print("Adding rest days...")
    team_long = add_rest_days(team_long)

    print("Adding rolling prior-only features...")
    team_long = add_rolling_features(team_long)

    print("Joining back to one row per game...")
    pre = build_pregame_table(team_long, gm)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    pre.to_csv(outfile, index=False)

    print(f"Saved: {outfile}")
    print(f"Rows: {len(pre):,}  Cols: {len(pre.columns):,}")
    print("Season counts:")
    print(pre["Season"].value_counts().sort_index().to_string())
    print("\nSample columns:")
    print(pre.columns[:60].tolist())


if __name__ == "__main__":
    main()