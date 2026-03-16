# core_engine/etl/pull_lines.py
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# --------------------------------------------------------------------------------------
# Paths / dirs
# --------------------------------------------------------------------------------------
try:
    # local utils (adjust if your utils live elsewhere)
    from core_engine.utils.paths import DB_DIR, SEED_DIR, ensure_dirs  # type: ignore
except Exception:
    # Minimal fallback so this runs standalone
    ROOT = Path(__file__).resolve().parents[2]  # <repo root>
    DB_DIR = ROOT / "db"
    SEED_DIR = ROOT / "core_engine" / "seeds"
    def ensure_dirs() -> None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        SEED_DIR.mkdir(parents=True, exist_ok=True)

OUT_DB = Path(DB_DIR) / "lines.csv"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def parse_weeks(weeks_spec: str) -> List[int]:
    """
    Accepts "1-22", "1,2,7,18", "1-18,20-22" -> [list of ints].
    """
    weeks: list[int] = []
    for tok in (weeks_spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            try:
                a_i, b_i = int(a), int(b)
                weeks.extend(list(range(min(a_i, b_i), max(a_i, b_i) + 1)))
            except Exception:
                pass
        else:
            try:
                weeks.append(int(tok))
            except Exception:
                pass
    # de-dupe, keep order
    seen = set()
    out: list[int] = []
    for w in weeks:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out

def am_to_dec(american: float | int | str | None) -> Optional[float]:
    """
    American -> decimal odds. Returns None if unknown.
    """
    if american is None:
        return None
    try:
        a = float(american)
    except Exception:
        try:
            a = float(str(american).strip())
        except Exception:
            return None
    if a > 0:
        return 1.0 + (a / 100.0)
    if a < 0:
        return 1.0 + (100.0 / abs(a))
    return None

def normalize_schema(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Make sure required columns exist and are typed consistently.
    Expected columns (best-effort):
      season, week, ts, league, market, ref, side, odds, dec_odds, imp, game_id, book
    Optional pass-through: p_win, ev
    """
    df = df.copy()

    # lowercase standard names if present
    df.columns = [str(c).strip() for c in df.columns]

    # Season
    df["season"] = season

    # Week
    if "week" not in df.columns:
        df["week"] = None
    # Market
    if "market" not in df.columns:
        df["market"] = None
    # Game id
    if "game_id" not in df.columns:
        # Try compose from known pieces if available (home@away + ts)
        if {"home_team", "away_team", "ts"} <= set(df.columns):
            df["game_id"] = (
                df["away_team"].astype(str).str.strip()
                + "@"
                + df["home_team"].astype(str).str.strip()
                + "_"
                + df["ts"].astype(str).str.strip()
            )
        else:
            df["game_id"] = None
    # Odds (american)
    if "odds" not in df.columns and "american" in df.columns:
        df["odds"] = df["american"]

    # Decimal odds + implied prob
    df["dec_odds"] = df.get("dec_odds")
    if df["dec_odds"].isna().all():
        df["dec_odds"] = df.get("odds").apply(am_to_dec)
    df["imp"] = df.get("imp")
    if df["imp"].isna().all():
        # implied probability from decimal odds
        df["imp"] = df["dec_odds"].apply(lambda d: (1.0 / float(d)) if (d and d > 0) else None)

    # Timestamp
    if "ts" not in df.columns:
        # try build from other date/time cols
        for cand in ("timestamp", "kickoff", "game_time", "date"):
            if cand in df.columns:
                df["ts"] = df[cand]
                break
        if "ts" not in df.columns:
            df["ts"] = pd.NaT
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["sort_ts"] = df["ts"]

    # League default
    if "league" not in df.columns:
        df["league"] = "NFL"

    # Book default
    if "book" not in df.columns:
        df["book"] = None

    # Ref (your unique selection key)
    if "ref" not in df.columns:
        # Try to synthesize from market/side/game_id
        df["ref"] = (
            df["market"].astype(str).str.upper().fillna("?")
            + "|"
            + df["side"].astype(str).str.upper().fillna("?")
            + "|"
            + df["game_id"].astype(str).fillna("?")
        )

    # Optional passthroughs
    if "p_win" not in df.columns:
        df["p_win"] = None
    if "ev" not in df.columns:
        df["ev"] = None

    # Final column ordering (friendly)
    order = [
        "season", "week", "ts", "sort_ts", "league",
        "market", "ref", "side", "odds", "dec_odds", "imp",
        "game_id", "book", "p_win", "ev"
    ]
    # include any extras after
    extras = [c for c in df.columns if c not in order]
    return df[order + extras]

# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------
def load_from_seed(season: int, weeks: list[int]) -> pd.DataFrame:
    """
    Load a pre-archived file: core_engine/seeds/lines_<season>.csv
    Filter by weeks if provided.
    """
    seed = Path(SEED_DIR) / f"lines_{season}.csv"
    if not seed.exists():
        return pd.DataFrame()
    df = pd.read_csv(seed)
    # If the seed is multi-season, filter it by 'season' first (just in case)
    if "season" in df.columns:
        try:
            df = df[pd.to_numeric(df["season"], errors="coerce").fillna(season).astype(int).eq(season)]
        except Exception:
            pass
    # Normalize early to ensure 'week' is present/typed
    df = normalize_schema(df, season=season)
    if weeks:
        df = df[df["week"].isin(weeks)]
    return df

def fetch_historical_from_api(season: int, weeks: list[int]) -> pd.DataFrame:
    """
    Optional: plug in a true historical feed here (e.g. Odds API archive).
    Expected to return a dataframe that normalize_schema can massage.
    If you don't have an API yet, just return empty.
    """
    # EXAMPLE scaffold (pseudo):
    # api_key = os.environ.get("ODDS_API_KEY")
    # if not api_key:
    #     return pd.DataFrame()
    # rows = your_api_client.fetch_historical(season=season, weeks=weeks, sport="americanfootball_nfl", api_key=api_key)
    # df = pd.DataFrame(rows)
    # return normalize_schema(df, season=season)
    return pd.DataFrame()

def fetch_live_current_season(weeks: list[int]) -> pd.DataFrame:
    """
    Your existing live fetcher for the current season.
    Replace this with your real function (you likely already have it).
    Must return a dataframe compatible with normalize_schema.
    """
    # If you already have a live function (e.g., pull today’s markets), call it here.
    # Below is a placeholder that returns empty.
    return pd.DataFrame()

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
@dataclass
class Args:
    season: int
    weeks: list[int]
    out: Optional[Path]

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True, help="NFL season (e.g. 2024)")
    p.add_argument("--weeks", type=str, default="1-22", help='Weeks range "1-22" or list "1,2,3"')
    p.add_argument("--out", type=Path, default=None, help="Optional path to write a copy")
    a = p.parse_args()
    return Args(season=a.season, weeks=parse_weeks(a.weeks), out=a.out)

def main() -> int:
    ensure_dirs()
    args = parse_args()
    this_year = datetime.now().year

    df = pd.DataFrame()

    # 1) Seeds first (fast + offline)
    df = load_from_seed(args.season, args.weeks)
    if df.empty:
        # 2) Historical API (if wired)
        df = fetch_historical_from_api(args.season, args.weeks)

    # 3) Live (current season only; do NOT fabricate history)
    if df.empty and args.season == this_year:
        df = fetch_live_current_season(args.weeks)

    if df.empty:
        print(f"[pull_lines] No data for season {args.season} weeks {args.weeks}. Nothing written.")
        return 2

    # Normalize one last time and enforce season/week types
    df = normalize_schema(df, season=args.season)

    # Write to DB and optional --out
    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DB, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[pull_lines] Wrote {len(df):,} rows -> {OUT_DB}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[pull_lines] Also wrote -> {args.out}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())


