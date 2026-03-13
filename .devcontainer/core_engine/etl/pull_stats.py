# core_engine/etl/pull_stats.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------------------
# Paths / utils
# --------------------------------------------------------------------------------------
try:
    from core_engine.utils.paths import DB_DIR, SEED_DIR, ensure_dirs  # type: ignore
except Exception:
    ROOT = Path(__file__).resolve().parents[2]
    DB_DIR = ROOT / "db"
    SEED_DIR = ROOT / "core_engine" / "seeds"

    def ensure_dirs() -> None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        SEED_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = Path(DB_DIR) / "features_raw.csv"

# --------------------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------------------
def parse_weeks(spec: str) -> list[int]:
    """
    Accepts "1-22", "1,2,7", or "1-10,12,18-20" -> [ints] (unique, ordered).
    """
    if not spec:
        return []
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            try:
                a_i, b_i = int(a), int(b)
                out.extend(range(min(a_i, b_i), max(a_i, b_i) + 1))
            except Exception:
                pass
        else:
            try:
                out.append(int(tok))
            except Exception:
                pass
    # de-dupe keep order
    seen: set[int] = set()
    uniq: list[int] = []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _auto_rename(df: pd.DataFrame, mapping: dict[str, tuple[str, ...] | list[str] | str]) -> pd.DataFrame:
    # case-insensitive aliasing
    lc_map = {str(c).lower().strip(): c for c in df.columns}
    ren: dict[str, str] = {}
    for want, aliases in mapping.items():
        if want in df.columns:
            continue
        if isinstance(aliases, str):
            aliases = (aliases,)
        for a in aliases:
            if a in df.columns:
                ren[a] = want
                break
            a_lc = a.lower()
            if a_lc in lc_map:
                ren[lc_map[a_lc]] = want
                break
    return df.rename(columns=ren) if ren else df


def _normalize_game_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = _auto_rename(
        df,
        {
            "season": ("season",),
            "week": ("week", "wk"),
            "team": ("team", "team_abbr", "abbr"),
            "game_id": ("game_id", "gameid", "gid"),
            "ts": ("ts", "kickoff", "game_time", "datetime", "date"),
            "home_team": ("home_team", "home", "home_abbr"),
            "away_team": ("away_team", "away", "away_abbr"),
        },
    )
    for col in ("season", "week", "team", "game_id"):
        if col not in df.columns:
            df[col] = None
    return df


def _cast_keys(df: pd.DataFrame, season_hint: Optional[int] = None) -> pd.DataFrame:
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce").astype("Int64")
    if season_hint is not None:
        df.loc[df["season"].isna(), "season"] = season_hint
    df["week"] = pd.to_numeric(df.get("week"), errors="coerce").astype("Int64")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for c in ("team", "game_id", "home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


def _filter_weeks(df: pd.DataFrame, weeks: list[int]) -> pd.DataFrame:
    if not weeks or "week" not in df.columns:
        return df
    return df[df["week"].isin(weeks)]


def _composite_key(df: pd.DataFrame) -> pd.Series:
    # Season|Week|Team|GameID (stable)
    return (
        df["season"].astype("Int64").astype("string").fillna("")
        + "|"
        + df["week"].astype("Int64").astype("string").fillna("")
        + "|"
        + df["team"].astype("string").fillna("").str.upper()
        + "|"
        + df["game_id"].astype("string").fillna("")
    )


def _merge_append_dedupe(base: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if base is None or base.empty:
        out = new.copy()
        out["_key"] = _composite_key(out)
        return out.drop_duplicates("_key", keep="last").drop(columns=["_key"])
    X = base.copy()
    Y = new.copy()
    X["_key"] = _composite_key(X)
    Y["_key"] = _composite_key(Y)
    Z = pd.concat([X, Y], ignore_index=True)
    Z = Z.drop_duplicates("_key", keep="last").drop(columns=["_key"])
    return Z


# --------------------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------------------
def _load_seed_games(season: int) -> pd.DataFrame:
    p = _first_existing([SEED_DIR / f"games_{season}.csv", SEED_DIR / "games.csv"])
    if not p:
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = _normalize_game_keys(df)
    return _cast_keys(df, season_hint=season)


def _load_seed_teamweeks(season: int) -> pd.DataFrame:
    p = _first_existing([SEED_DIR / f"teams_week_{season}.csv", SEED_DIR / "teams_week.csv"])
    if not p:
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = _normalize_game_keys(df)
    return _cast_keys(df, season_hint=season)


def _load_nflverse_schedule(season: int) -> pd.DataFrame:
    """
    Best-effort nflverse schedule via nflreadpy if installed.
    Returns columns: season,week,ts,game_id,home_team,away_team
    """
    try:
        import nflreadpy as nfl  # type: ignore
    except Exception:
        return pd.DataFrame()

    try:
        sch = nfl.load_schedules([season]).to_pandas()
    except Exception:
        return pd.DataFrame()

    keep = sch.rename(
        columns={
            "season": "season",
            "week": "week",
            "gameday": "ts",
            "game_id": "game_id",
            "home_team": "home_team",
            "away_team": "away_team",
        }
    )[["season", "week", "ts", "game_id", "home_team", "away_team"]]
    keep["ts"] = pd.to_datetime(keep["ts"], errors="coerce")
    keep["team"] = pd.NA  # placeholder so it can merge with team-week
    return _cast_keys(keep, season_hint=season)


def _load_kaggle_teamweeks(path: Optional[Path], season: int) -> pd.DataFrame:
    """
    Optional: user-provided team-week stats file (Kaggle or your own).
    It should include (or be inferable to) season/week/team/game_id.
    """
    if not path:
        return pd.DataFrame()
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = _normalize_game_keys(df)
    return _cast_keys(df, season_hint=season)


# --------------------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------------------
@dataclass
class Args:
    season: int
    weeks: list[int]
    append: bool
    out: Optional[Path]
    parquet: Optional[Path]
    kaggle_stats: Optional[Path]


def _parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Build features_raw.csv from seeds and/or nflverse, with optional Kaggle overlay.")
    ap.add_argument("--season", type=int, required=True, help="Season, e.g., 2024")
    ap.add_argument("--weeks", type=str, default="1-22", help='Weeks like "1-22" or "1,2,7"')
    ap.add_argument("--append", action="store_true", help="Append+dedupe into db/features_raw.csv (else overwrite)")
    ap.add_argument("--out", type=Path, default=None, help="Optional second copy of the output CSV")
    ap.add_argument("--parquet", type=Path, default=None, help="Optional Parquet sidecar path")
    ap.add_argument("--kaggle_stats", type=Path, default=None, help="Optional Kaggle (or custom) team-week stats CSV to merge")
    a = ap.parse_args()
    return Args(
        season=a.season,
        weeks=parse_weeks(a.weeks),
        append=a.append,
        out=a.out,
        parquet=a.parquet,
        kaggle_stats=a.kaggle_stats,
    )


def main() -> int:
    ensure_dirs()
    args = _parse_args()

    # 1) Load sources (prefer seeds, fallback to nflverse for schedule)
    games_seed = _load_seed_games(args.season)
    teams_seed = _load_seed_teamweeks(args.season)

    if games_seed.empty:
        games_verse = _load_nflverse_schedule(args.season)
    else:
        games_verse = pd.DataFrame()

    # 2) Optional Kaggle overlay for team-week metrics
    teams_kaggle = _load_kaggle_teamweeks(args.kaggle_stats, args.season)

    # 3) Filter weeks early
    if args.weeks:
        if not games_seed.empty:
            games_seed = _filter_weeks(games_seed, args.weeks)
        if not games_verse.empty:
            games_verse = _filter_weeks(games_verse, args.weeks)
        if not teams_seed.empty:
            teams_seed = _filter_weeks(teams_seed, args.weeks)
        if not teams_kaggle.empty:
            teams_kaggle = _filter_weeks(teams_kaggle, args.weeks)

    # 4) Choose a base "games" table: seeds first, else nflverse schedule
    if not games_seed.empty:
        games = games_seed
        src_games = "seed"
    else:
        games = games_verse
        src_games = "nflverse" if not games.empty else "none"

    # 5) Choose team-week: combine seed + kaggle if both present
    if not teams_seed.empty and not teams_kaggle.empty:
        # union (dedupe on season|week|team|game_id; Kaggle last-wins)
        TW = _merge_append_dedupe(teams_seed, teams_kaggle)
        src_tw = "seed+kaggle"
    elif not teams_seed.empty:
        TW = teams_seed
        src_tw = "seed"
    else:
        TW = teams_kaggle
        src_tw = "kaggle" if not TW.empty else "none"

    if games.empty and TW.empty:
        print(f"[pull_stats] No data found for season {args.season}. (games={src_games}, teamweeks={src_tw})")
        return 2

    # 6) Merge (left join so we keep one row per game-team, if present)
    # If games has no 'team' column (nflverse schedule), we’ll merge will be on (season,week,game_id) only
    on_cols = ["season", "week", "game_id"]
    if "team" in games.columns and "team" in TW.columns:
        on_cols = ["season", "week", "game_id", "team"]

    if games.empty:
        merged = TW.copy()
    elif TW.empty:
        merged = games.copy()
    else:
        merged = games.merge(TW, on=on_cols, how="left", suffixes=("", "_tw"))

    # Ensure the keys are present and typed
    merged = _normalize_game_keys(merged)
    merged = _cast_keys(merged, season_hint=args.season)

    # 7) Append+dedupe or overwrite
    if args.append and OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
        base = pd.read_csv(OUT_CSV)
        base = _normalize_game_keys(base)
        base = _cast_keys(base)
        out_df = _merge_append_dedupe(base, merged)
    else:
        out_df = merged

    # Stable sort: by season, week, team, game time if present
    sort_cols = [c for c in ("season", "week", "team", "ts") if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols)

    # 8) Write outputs
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[pull_stats] Wrote {len(out_df):,} rows -> {OUT_CSV} (games={src_games}, teamweeks={src_tw})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"[pull_stats] Also wrote -> {args.out}")

    if args.parquet:
        args.parquet.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(args.parquet, index=False)
        print(f"[pull_stats] Also wrote Parquet -> {args.parquet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





