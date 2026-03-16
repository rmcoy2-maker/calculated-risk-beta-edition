from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = PROJECT_ROOT / "exports"


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False)


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def build_full_game_date_from_season_mmdd(
    season_series: pd.Series,
    mmdd_series: pd.Series,
) -> pd.Series:
    """
    Build YYYY-MM-DD from NFL Season + MM/DD.
    Sep-Dec -> same season year
    Jan-Feb -> following calendar year
    """
    season_num = pd.to_numeric(season_series, errors="coerce")
    mmdd = mmdd_series.astype("string").str.strip()

    parts = mmdd.str.extract(r"^(?P<month>\d{1,2})/(?P<day>\d{1,2})$")
    month = pd.to_numeric(parts["month"], errors="coerce")
    day = pd.to_numeric(parts["day"], errors="coerce")

    year = season_num.copy()
    jan_feb_mask = month.isin([1, 2]).fillna(False)
    year.loc[jan_feb_mask] = year.loc[jan_feb_mask] + 1

    year_str = year.astype("Int64").astype("string")
    month_str = month.astype("Int64").astype("string").str.zfill(2)
    day_str = day.astype("Int64").astype("string").str.zfill(2)

    date_str = year_str + "-" + month_str + "-" + day_str
    return pd.to_datetime(date_str, errors="coerce")


def clean_team_name(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip()

    alias_map = {
        "WSH": "Washington Commanders",
        "WAS": "Washington Commanders",
        "Washington Football Team": "Washington Commanders",
        "Football Team": "Washington Commanders",
        "OAK": "Las Vegas Raiders",
        "LV": "Las Vegas Raiders",
        "SD": "Los Angeles Chargers",
        "STL": "Los Angeles Rams",
        "LA Rams": "Los Angeles Rams",
        "LA Chargers": "Los Angeles Chargers",
        "SF": "San Francisco 49ers",
        "TB": "Tampa Bay Buccaneers",
        "NE": "New England Patriots",
        "NO": "New Orleans Saints",
        "GB": "Green Bay Packers",
        "KC": "Kansas City Chiefs",
    }
    return s.replace(alias_map)


def normalize_id_part(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def canonical_game_id(
    date_series: pd.Series,
    home_series: pd.Series,
    away_series: pd.Series,
) -> pd.Series:
    date_part = pd.to_datetime(date_series, errors="coerce").dt.strftime("%Y-%m-%d")
    home_part = normalize_id_part(home_series)
    away_part = normalize_id_part(away_series)
    return (date_part.fillna("nodate") + "_" + away_part + "_at_" + home_part).astype("string")


def pick_game_key_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    game_id_col = first_existing(df, ["game_id", "event_id", "id", "game_key"])
    date_col = first_existing(
        df,
        [
            "game_date",
            "Date_std",
            "date_std",
            "Date",
            "date",
            "game_date_str",
            "commence_time",
            "start_time",
            "kickoff",
            "scheduled",
        ],
    )
    home_col = first_existing(df, ["home_team", "HomeTeam", "home", "team_home", "home_name"])
    away_col = first_existing(df, ["away_team", "AwayTeam", "away", "team_away", "away_name"])
    return game_id_col, date_col, home_col, away_col


def standardize_games_base(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()

    game_id_col, date_col, home_col, away_col = pick_game_key_columns(df)

    if home_col is None or away_col is None:
        raise ValueError(f"{source_name}: could not find home/away team columns")

    df["home_team"] = clean_team_name(df[home_col])
    df["away_team"] = clean_team_name(df[away_col])

    season_col = first_existing(df, ["Season", "season"])
    week_col = first_existing(df, ["Week", "week", "week_num", "nfl_week"])

    if date_col is not None:
        raw_dates = df[date_col]

        # scores_master special case: Date is MM/DD with separate Season
        if str(date_col) == "Date" and season_col is not None:
            dt_local = build_full_game_date_from_season_mmdd(df[season_col], raw_dates)
            df["game_datetime_utc"] = pd.to_datetime(dt_local, errors="coerce", utc=True)
            df["game_date"] = pd.to_datetime(dt_local, errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            dt = pd.to_datetime(raw_dates, errors="coerce", utc=True)

            if dt.notna().sum() == 0:
                dt2 = pd.to_datetime(raw_dates, errors="coerce")
                df["game_datetime_utc"] = pd.to_datetime(dt2, errors="coerce", utc=True)
                df["game_date"] = pd.to_datetime(dt2, errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                df["game_datetime_utc"] = dt
                df["game_date"] = dt.dt.strftime("%Y-%m-%d")
    else:
        df["game_datetime_utc"] = pd.NaT
        df["game_date"] = pd.NA

    if game_id_col is not None:
        df["game_id"] = df[game_id_col].astype("string")
    else:
        df["game_id"] = canonical_game_id(df["game_date"], df["home_team"], df["away_team"])

    if season_col is not None:
        df["Season"] = pd.to_numeric(df[season_col], errors="coerce").astype("Int64")
    else:
        dt_local = pd.to_datetime(df["game_date"], errors="coerce")
        df["Season"] = (dt_local.dt.year - (dt_local.dt.month <= 2).astype("Int64")).astype("Int64")

    if week_col is not None:
        df["Week"] = df[week_col].astype("string")
    else:
        df["Week"] = pd.Series([pd.NA] * len(df), dtype="string")

    return df


def report_coverage(df: pd.DataFrame, name: str) -> None:
    season_col = first_existing(df, ["Season", "season"])
    date_col = first_existing(df, ["game_date", "Date_std", "date_std", "Date", "date", "commence_time", "start_time", "kickoff", "scheduled"])

    season_msg = "Season coverage: unavailable"
    date_msg = "Date coverage: unavailable"

    if season_col is not None:
        s = pd.to_numeric(df[season_col], errors="coerce").dropna()
        if not s.empty:
            season_msg = f"Season coverage: {int(s.min())} -> {int(s.max())}"

    if date_col is not None:
        raw_dates = df[date_col]

        if str(date_col) == "Date" and season_col is not None:
            dt_local = build_full_game_date_from_season_mmdd(df[season_col], raw_dates)
            df["game_datetime_utc"] = pd.to_datetime(dt_local, errors="coerce", utc=True)
            df["game_date"] = dt_local.dt.strftime("%Y-%m-%d")
        else:
            dt = pd.to_datetime(raw_dates, errors="coerce", utc=True)

            if dt.notna().sum() == 0:
                dt2 = pd.to_datetime(raw_dates, errors="coerce")
                df["game_datetime_utc"] = pd.to_datetime(dt2, errors="coerce", utc=True)
                df["game_date"] = pd.to_datetime(dt2, errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                df["game_datetime_utc"] = dt
                df["game_date"] = dt.dt.strftime("%Y-%m-%d")
    else:
        df["game_datetime_utc"] = pd.NaT
        df["game_date"] = pd.NA

    print(f"\n[{name}]")
    print(f"rows={len(df):,} cols={len(df.columns):,}")
    print(season_msg)
    print(date_msg)


def build_team_feature_wide(team_features: pd.DataFrame) -> pd.DataFrame:
    tf = team_features.copy()

    game_id_col, date_col, home_col, away_col = pick_game_key_columns(tf)
    team_col = first_existing(tf, ["team", "team_name", "team_abbr"])
    opp_col = first_existing(tf, ["opponent", "opp_team", "opponent_team"])
    home_away_col = first_existing(tf, ["home_away", "is_home", "venue_role", "team_side"])

    if game_id_col is None and (date_col is None or team_col is None):
        raise ValueError("team_features_from_plays.csv: insufficient keys to map features to games")

    non_feature = {
        c for c in tf.columns
        if str(c).lower() in {
            "game_id", "event_id", "id", "game_key",
            "game_date", "date", "commence_time", "start_time", "kickoff", "scheduled",
            "season", "week", "week_num", "nfl_week",
            "team", "team_name", "team_abbr",
            "opponent", "opp_team", "opponent_team",
            "home_team", "home", "team_home", "home_name",
            "away_team", "away", "team_away", "away_name",
            "home_away", "is_home", "venue_role", "team_side"
        }
    }

    feature_cols = [c for c in tf.columns if c not in non_feature]

    if home_col is not None and away_col is not None and team_col is None:
        out = standardize_games_base(tf, "team_features_from_plays.csv")
        keep = ["game_id", "game_date", "Season", "Week", "home_team", "away_team"] + feature_cols
        keep = [c for c in keep if c in out.columns]
        return out[keep].drop_duplicates(subset=["game_id"])

    if team_col is None:
        raise ValueError("team_features_from_plays.csv: could not identify team column")

    tf[team_col] = clean_team_name(tf[team_col])

    if game_id_col is not None:
        tf["game_id"] = tf[game_id_col].astype("string")
    else:
        if opp_col is None:
            raise ValueError("team_features_from_plays.csv: need opponent column when no game_id exists")
        tf[opp_col] = clean_team_name(tf[opp_col])
        tf["game_id"] = canonical_game_id(
            pd.to_datetime(tf[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            tf[team_col],
            tf[opp_col],
        )

    if home_away_col is not None:
        role_raw = tf[home_away_col].astype("string").str.lower().fillna("")
        tf["_side"] = np.where(
            role_raw.isin(["home", "h", "1", "true"]),
            "home",
            np.where(role_raw.isin(["away", "a", "0", "false"]), "away", pd.NA)
        )
    elif home_col is not None and away_col is not None:
        tf[home_col] = clean_team_name(tf[home_col])
        tf[away_col] = clean_team_name(tf[away_col])
        tf["_side"] = np.where(
            tf[team_col] == tf[home_col],
            "home",
            np.where(tf[team_col] == tf[away_col], "away", pd.NA)
        )
    else:
        tf["_side"] = pd.NA

    unresolved = tf["_side"].isna()
    unresolved_count = int(unresolved.sum())

    if unresolved_count:
        print(f"[WARN] team_features_from_plays.csv unresolved home/away rows: {unresolved_count:,}")
        tf = tf.loc[~unresolved].copy()

    if tf.empty:
        print("[WARN] No usable team feature rows after home/away resolution.")
        return pd.DataFrame(columns=["game_id"])

    home_df = tf.loc[tf["_side"] == "home", ["game_id"] + feature_cols].copy()
    away_df = tf.loc[tf["_side"] == "away", ["game_id"] + feature_cols].copy()

    home_df = home_df.rename(columns={c: f"home_{c}" for c in feature_cols})
    away_df = away_df.rename(columns={c: f"away_{c}" for c in feature_cols})

    wide = home_df.merge(away_df, on="game_id", how="outer")
    return wide.drop_duplicates(subset=["game_id"])


def build_scores_game_level(scores: pd.DataFrame) -> pd.DataFrame:
    sc = standardize_games_base(scores, "scores_master.csv")

    home_score_col = first_existing(sc, ["home_score", "homescore", "score_home"])
    away_score_col = first_existing(sc, ["away_score", "awayscore", "score_away"])

    if home_score_col is None or away_score_col is None:
        raise ValueError("scores_master.csv: missing home_score / away_score columns")

    sc["home_score"] = pd.to_numeric(sc[home_score_col], errors="coerce")
    sc["away_score"] = pd.to_numeric(sc[away_score_col], errors="coerce")
    sc["total_points"] = sc["home_score"] + sc["away_score"]
    sc["margin"] = sc["home_score"] - sc["away_score"]
    sc["home_win"] = (sc["home_score"] > sc["away_score"]).astype("Int64")
    sc["away_win"] = (sc["away_score"] > sc["home_score"]).astype("Int64")
    sc["is_tie"] = (sc["home_score"] == sc["away_score"]).astype("Int64")

    keep = [
        "game_id", "game_date", "game_datetime_utc", "Season", "Week",
        "home_team", "away_team",
        "home_score", "away_score", "total_points",
        "margin", "home_win", "away_win", "is_tie"
    ]
    return sc[keep].drop_duplicates(subset=["game_id"])


def build_games_master(
    plays_master_path: Path,
    team_features_path: Path,
    game_features_path: Path,
    scores_master_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    plays = read_csv_safe(plays_master_path)
    team_features = read_csv_safe(team_features_path)
    game_features = read_csv_safe(game_features_path)
    scores = read_csv_safe(scores_master_path)

    report_coverage(plays, "plays_master.csv")
    report_coverage(team_features, "team_features_from_plays.csv")
    report_coverage(game_features, "game_features_from_plays.csv")
    report_coverage(scores, "scores_master.csv")

    scores_game = build_scores_game_level(scores)

    gf = standardize_games_base(game_features, "game_features_from_plays.csv")
    game_feature_nonkeys = [
        c for c in gf.columns
        if c not in {
            "game_id", "game_date", "game_datetime_utc", "Season", "Week", "home_team", "away_team"
        }
    ]
    gf_keep = ["game_id"] + game_feature_nonkeys
    gf_keep = [c for c in gf_keep if c in gf.columns]
    gf = gf[gf_keep].drop_duplicates(subset=["game_id"])

    tf_wide = build_team_feature_wide(team_features)

    gm = scores_game.merge(gf, on="game_id", how="left")
    gm = gm.merge(tf_wide, on="game_id", how="left")

    p = standardize_games_base(plays, "plays_master.csv")
    play_count = (
        p.groupby("game_id", dropna=False)
        .size()
        .rename("n_plays")
        .reset_index()
    )
    gm = gm.merge(play_count, on="game_id", how="left")

    front = [
        "game_id", "Season", "Week", "game_date", "game_datetime_utc",
        "away_team", "home_team",
        "away_score", "home_score", "margin", "total_points",
        "away_win", "home_win", "is_tie",
        "n_plays",
    ]
    front = [c for c in front if c in gm.columns]
    remaining = [c for c in gm.columns if c not in front]
    gm = gm[front + remaining]

    sort_cols = [c for c in ["Season", "game_date", "game_id"] if c in gm.columns]
    gm = gm.sort_values(by=sort_cols, ascending=True, kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gm.to_csv(output_path, index=False)

    print("\n[games_master.csv]")
    print(f"rows={len(gm):,} cols={len(gm.columns):,}")
    print(f"unique game_id={gm['game_id'].nunique():,}")
    print(f"duplicate game_id={gm['game_id'].duplicated().sum():,}")

    if "Season" in gm.columns:
        print("Season counts:")
        print(gm["Season"].value_counts(dropna=False).sort_index().to_string())

    return gm


def main() -> None:
    parser = argparse.ArgumentParser(description="Build games_master.csv")
    parser.add_argument(
        "--plays-master",
        default=str(DATA_DIR / "plays_master.csv"),
        help="Path to plays_master.csv",
    )
    parser.add_argument(
        "--team-features",
        default=str(DATA_DIR / "team_features_from_plays.csv"),
        help="Path to team_features_from_plays.csv",
    )
    parser.add_argument(
        "--game-features",
        default=str(DATA_DIR / "game_features_from_plays.csv"),
        help="Path to game_features_from_plays.csv",
    )
    parser.add_argument(
        "--scores-master",
        default=str(DATA_DIR / "scores_master.csv"),
        help="Path to scores_master.csv",
    )
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "games_master.csv"),
        help="Output path",
    )

    args = parser.parse_args()

    gm = build_games_master(
        plays_master_path=Path(args.plays_master),
        team_features_path=Path(args.team_features),
        game_features_path=Path(args.game_features),
        scores_master_path=Path(args.scores_master),
        output_path=Path(args.out),
    )

    print(f"\n[OK] wrote {args.out}")
    print(f"rows={len(gm):,} cols={len(gm.columns):,}")
    print("sample columns:")
    print(gm.columns[:30].tolist())


if __name__ == "__main__":
    main()