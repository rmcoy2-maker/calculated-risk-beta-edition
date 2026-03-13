from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

EDITION_ORDER = ["tnf", "sunday_morning", "sunday_afternoon", "snf", "monday", "tuesday"]

EDITION_LABELS = {
    "tnf": "Thursday Night Edition",
    "sunday_morning": "Sunday Morning Edition",
    "sunday_afternoon": "Sunday Afternoon Update",
    "snf": "Sunday Night Edition",
    "monday": "Monday Edition",
    "tuesday": "Tuesday Wrap Edition",
}

DAY_ORDER = {
    "Thursday": 0,
    "Friday": 1,
    "Saturday": 2,
    "Sunday": 3,
    "Monday": 4,
    "Tuesday": 5,
    "Wednesday": 6,
}

TEAM_ABBR = {
    "ARIZONA CARDINALS": "ARI", "ATLANTA FALCONS": "ATL", "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF", "CAROLINA PANTHERS": "CAR", "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN", "CLEVELAND BROWNS": "CLE", "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN", "DETROIT LIONS": "DET", "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU", "INDIANAPOLIS COLTS": "IND", "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC", "LAS VEGAS RAIDERS": "LV", "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR", "MIAMI DOLPHINS": "MIA", "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE", "NEW ORLEANS SAINTS": "NO", "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ", "PHILADELPHIA EAGLES": "PHI", "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF", "SEATTLE SEAHAWKS": "SEA", "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN", "WASHINGTON COMMANDERS": "WAS",
    "COMMANDERS": "WAS", "RAIDERS": "LV", "CHARGERS": "LAC", "RAMS": "LAR",
    "CARDINALS": "ARI", "FALCONS": "ATL", "RAVENS": "BAL", "BILLS": "BUF",
    "PANTHERS": "CAR", "BEARS": "CHI", "BENGALS": "CIN", "BROWNS": "CLE",
    "COWBOYS": "DAL", "BRONCOS": "DEN", "LIONS": "DET", "PACKERS": "GB",
    "TEXANS": "HOU", "COLTS": "IND", "JAGUARS": "JAX", "CHIEFS": "KC",
    "DOLPHINS": "MIA", "VIKINGS": "MIN", "PATRIOTS": "NE", "SAINTS": "NO",
    "GIANTS": "NYG", "JETS": "NYJ", "EAGLES": "PHI", "STEELERS": "PIT",
    "49ERS": "SF", "SEAHAWKS": "SEA", "BUCCANEERS": "TB", "TITANS": "TEN",
}


@dataclass
class LoadedData:
    root: Path
    exports_dir: Path
    reports_dir: Path
    edges: pd.DataFrame
    parlays: pd.DataFrame
    markets: pd.DataFrame
    games: pd.DataFrame
    scores: pd.DataFrame


# ---------------------------------------------------------------------------
# Paths and loading helpers
# ---------------------------------------------------------------------------

def project_root() -> Path:
    env = os.environ.get("EDGE_FINDER_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def exports_dir(root: Optional[Path] = None) -> Path:
    root = root or project_root()
    p = root / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir(root: Optional[Path] = None) -> Path:
    p = exports_dir(root) / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _candidate_files(base_dirs: Iterable[Path], names: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for base in base_dirs:
        for name in names:
            p = base / name
            if p.exists() and p.stat().st_size > 0:
                out.append(p)
    return out


def _latest_file(base_dirs: Iterable[Path], names: Iterable[str]) -> Optional[Path]:
    files = _candidate_files(base_dirs, names)
    return max(files, key=lambda x: x.stat().st_mtime) if files else None


def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if len(df.columns) == 1 and "\t" in str(df.columns[0]):
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", sep="\t")
    return df


def _read_excel(path: Optional[Path]) -> pd.DataFrame:
    if path is None or (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("string").fillna("").str.strip()
    return df


def _to_et(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    if getattr(ts.dt, "tz", None) is None:
        ts = pd.to_datetime(series, errors="coerce")
        try:
            ts = ts.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            pass
        return ts
    return ts.dt.tz_convert("America/New_York")


def _coalesce(df: pd.DataFrame, candidates: list[str], default: object = pd.NA) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df), index=df.index)


def _american_to_decimal(series: pd.Series) -> pd.Series:
    o = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(o > 0, 1 + o / 100.0, np.where(o < 0, 1 + 100.0 / np.abs(o), np.nan)), index=series.index)


def _american_implied_prob(series: pd.Series) -> pd.Series:
    o = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(o > 0, 100.0 / (o + 100.0), np.where(o < 0, np.abs(o) / (np.abs(o) + 100.0), np.nan)), index=series.index)


def _std_team(s: object) -> str:
    if s is None or pd.isna(s):
        return ""
    raw = str(s).strip().upper()
    if not raw:
        return ""
    raw = raw.replace("ST.", "ST").replace("  ", " ")
    return TEAM_ABBR.get(raw, raw)


def _infer_game_times(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ts = _coalesce(df, ["commence_time", "start_time", "event_date", "date", "game_date", "schedule_date", "_date_iso", "_event_ts_est", "_event_ts_utc"])
    dt = _to_et(ts)
    df = df.copy()
    df["game_dt_et"] = dt
    df["game_date_et"] = pd.to_datetime(df["game_dt_et"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["day_name"] = pd.to_datetime(df["game_dt_et"], errors="coerce").dt.day_name().fillna("")
    if df["day_name"].eq("").all() and "day" in df.columns:
        df["day_name"] = df["day"].astype("string").fillna("")
    return df


def _infer_matchup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["home"] = _coalesce(out, ["home", "home_team", "team_home", "homeName", "HOME", "_home_nick", "HomeTeam", "hometeam"]).astype("string")
    out["away"] = _coalesce(out, ["away", "away_team", "team_away", "awayName", "AWAY", "_away_nick", "AwayTeam", "awayteam"]).astype("string")
    out["home"] = out["home"].map(_std_team)
    out["away"] = out["away"].map(_std_team)
    out["game_id"] = _coalesce(out, ["game_id", "event_id", "eventId", "matchup_id", "_key"]).astype("string")
    out["matchup"] = np.where(
        (out["away"].fillna("") != "") & (out["home"].fillna("") != ""),
        out["away"].fillna("") + " @ " + out["home"].fillna(""),
        _coalesce(out, ["matchup", "game", "event_name"]).astype("string"),
    )
    return out


def _normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _normalize_text(_infer_matchup(_infer_game_times(df)))
    out["season"] = pd.to_numeric(_coalesce(out, ["season", "Season", "schedule_season"]), errors="coerce")
    out["week"] = pd.to_numeric(_coalesce(out, ["week", "Week", "week_num", "schedule_week", "week_override"]), errors="coerce")
    out["market"] = _coalesce(out, ["market", "market_type", "bet_type", "_market_norm"]).astype("string")
    out["side"] = _coalesce(out, ["side", "selection", "team_name", "player_name", "outcome", "team"]).astype("string")
    out["line"] = pd.to_numeric(_coalesce(out, ["line", "point", "Line/Point"]), errors="coerce")
    out["odds"] = pd.to_numeric(_coalesce(out, ["odds", "price", "American", "american"]), errors="coerce")
    out["p_win"] = pd.to_numeric(_coalesce(out, ["p_win", "win_prob", "model_prob", "probability", "p_combo"]), errors="coerce")
    out["book"] = _coalesce(out, ["book", "sportsbook", "book_name", "ref"]).astype("string")
    out["ev"] = pd.to_numeric(_coalesce(out, ["ev", "ev/$1", "ev_per_$1", "_ev_per_$1", "ev_est", "edge", "value"]), errors="coerce")
    out["implied_prob"] = _american_implied_prob(out["odds"])
    out["edge_pct"] = out["p_win"] - out["implied_prob"]
    if out["ev"].isna().all() and out["p_win"].notna().any() and out["odds"].notna().any():
        dec = _american_to_decimal(out["odds"])
        out["ev"] = out["p_win"] * (dec - 1) - (1 - out["p_win"])
    return out


def _normalize_parlays(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _normalize_edges(df)
    out["parlay_proba"] = pd.to_numeric(_coalesce(out, ["parlay_proba", "parlay_prob", "prob_legs_all_hit"]), errors="coerce")
    out["legs"] = pd.to_numeric(_coalesce(out, ["legs", "n_legs", "legs_short"]), errors="coerce")
    out["dec_comb"] = pd.to_numeric(_coalesce(out, ["dec_comb", "combined_decimal_odds"]), errors="coerce")
    out["parlay_score"] = pd.to_numeric(_coalesce(out, ["parlay_score", "score_cut", "lift"]), errors="coerce")
    return out


def _normalize_markets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _normalize_text(_infer_matchup(_infer_game_times(df)))
    out["season"] = pd.to_numeric(_coalesce(out, ["season", "Season", "schedule_season"]), errors="coerce")
    out["week"] = pd.to_numeric(_coalesce(out, ["week", "Week", "schedule_week"]), errors="coerce")
    out["market"] = _coalesce(out, ["market", "market_type", "_market_norm"]).astype("string")
    out["side"] = _coalesce(out, ["side", "selection", "ref", "team_name", "player_name", "outcome", "team", "player"]).astype("string")
    out["line"] = pd.to_numeric(_coalesce(out, ["line", "point", "Line/Point", "spread_close", "total_close"]), errors="coerce")
    out["odds"] = pd.to_numeric(_coalesce(out, ["odds", "price", "American", "ml_home", "ml_away"]), errors="coerce")
    out["book"] = _coalesce(out, ["book", "sportsbook", "book_name", "source"]).astype("string")
    return out


def _normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _normalize_text(_infer_matchup(_infer_game_times(df)))
    out["season"] = pd.to_numeric(_coalesce(out, ["season", "Season", "schedule_season"]), errors="coerce")
    out["week"] = pd.to_numeric(_coalesce(out, ["week", "Week", "schedule_week"]), errors="coerce")
    out["spread_close"] = pd.to_numeric(_coalesce(out, ["spread_close", "spread_favorite"]), errors="coerce")
    out["total_close"] = pd.to_numeric(_coalesce(out, ["total_close", "over_under_line"]), errors="coerce")
    out["ml_home"] = pd.to_numeric(_coalesce(out, ["ml_home"]), errors="coerce")
    out["ml_away"] = pd.to_numeric(_coalesce(out, ["ml_away"]), errors="coerce")
    out["home_score"] = pd.to_numeric(_coalesce(out, ["home_score", "score_home", "HomeScore", "home_pts"]), errors="coerce")
    out["away_score"] = pd.to_numeric(_coalesce(out, ["away_score", "score_away", "AwayScore", "away_pts"]), errors="coerce")
    out["weather_wind_mph"] = pd.to_numeric(_coalesce(out, ["weather_wind_mph"]), errors="coerce")
    out["weather_temperature"] = pd.to_numeric(_coalesce(out, ["weather_temperature"]), errors="coerce")
    out["stadium"] = _coalesce(out, ["stadium"]).astype("string")
    out["weather_detail"] = _coalesce(out, ["weather_detail"]).astype("string")
    return out


def load_all(root: Optional[Path] = None) -> LoadedData:
    root = root or project_root()
    exp = exports_dir(root)
    search_dirs = [exp, root]

    edges_path = _latest_file(search_dirs, [
        "cr_edges_latest.csv",
        "cr_edges.csv",
        "edges_standardized.csv",
        "edges_graded_full_normalized_std.csv",
        "edges_graded_full.csv",
        "edges_normalized.csv",
        "edges_master.csv",
        "edges.csv",
    ])
    parlay_path = _latest_file(search_dirs, [
        "cr_parlay_scores_latest.csv",
        "cr_parlay_scores.csv",
        "parlay_scores.csv",
        "parlay_scores_live.csv",
        "parlay_scores_master.csv",
    ])
    market_path = _latest_file(search_dirs, [
        "cr_markets_latest.csv",
        "cr_markets.csv",
        "lines_snapshots.csv",
        "odds_lines_all_long.csv",
        "markets_master.csv",
        "markets_live.csv",
        "lines_live.csv",
        "markets.csv",
    ])
    games_csv_path = _latest_file(search_dirs, [
        "cr_games_latest.csv",
        "cr_games.csv",
        "games_master_template.csv",
        "games_with_odds.csv",
    ])
    games_xlsx_path = _latest_file(search_dirs, [
        "games_master_with_ev_and_tiers.xlsx",
    ])
    scores_path = _latest_file(search_dirs, [
        "cr_scores_latest.csv",
        "cr_scores.csv",
        "scores_normalized_std_maxaligned.csv",
        "spreadspoke_scores.csv",
    ])

    games_df = _read_csv(games_csv_path)
    if games_df.empty:
        games_df = _read_excel(games_xlsx_path)

    return LoadedData(
        root=root,
        exports_dir=exp,
        reports_dir=reports_dir(root),
        edges=_normalize_edges(_read_csv(edges_path)),
        parlays=_normalize_parlays(_read_csv(parlay_path)),
        markets=_normalize_markets(_read_csv(market_path)),
        games=_normalize_games(games_df),
        scores=_normalize_games(_read_csv(scores_path)),
    )


# ---------------------------------------------------------------------------
# Game filtering and report helpers
# ---------------------------------------------------------------------------

def filter_for_week(df: pd.DataFrame, season: Optional[int], week: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if season is not None and "season" in out.columns:
        out = out[pd.to_numeric(out["season"], errors="coerce") == float(season)]
    if "week" in out.columns:
        out = out[pd.to_numeric(out["week"], errors="coerce") == float(week)]
    return out


def infer_games(data: LoadedData, season: Optional[int], week: int) -> pd.DataFrame:
    frames = [
        filter_for_week(data.games, season, week),
        filter_for_week(data.edges, season, week),
        filter_for_week(data.parlays, season, week),
        filter_for_week(data.markets, season, week),
        filter_for_week(data.scores, season, week),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["game_id", "matchup", "home", "away", "game_dt_et", "day_name", "day_order"])
    combo = pd.concat(frames, ignore_index=True, sort=False)
    combo = _normalize_games(_infer_matchup(_infer_game_times(combo)))
    keep = [c for c in [
        "game_id", "matchup", "home", "away", "game_dt_et", "day_name", "spread_close", "total_close",
        "home_score", "away_score", "weather_wind_mph", "weather_temperature", "weather_detail", "stadium"
    ] if c in combo.columns]
    games = combo[keep].copy()
    games = games[games["matchup"].fillna("") != ""].copy()
    games["day_order"] = games["day_name"].map(DAY_ORDER).fillna(99)
    games = games.sort_values(["day_order", "game_dt_et", "matchup"], na_position="last")
    dedupe_cols = [c for c in ["game_id", "matchup"] if c in games.columns]
    games = games.drop_duplicates(subset=dedupe_cols, keep="first")
    return games.reset_index(drop=True)


def edition_games(games: pd.DataFrame, edition: str) -> pd.DataFrame:
    if games.empty:
        return games
    ed = edition.lower().strip()
    work = games.copy().sort_values(["day_order", "game_dt_et", "matchup"], na_position="last")
    sunday = work[work["day_name"].eq("Sunday")].copy()
    monday = work[work["day_name"].eq("Monday")].copy()
    thursday = work[work["day_name"].eq("Thursday")].copy()
    if ed == "tnf":
        return thursday.head(1)
    if ed == "sunday_morning":
        if not sunday.empty:
            return sunday
        return work[work["day_name"].isin(["Sunday", "Monday"])].copy()
    if ed == "sunday_afternoon":
        if sunday.empty:
            return work
        late = sunday[pd.to_datetime(sunday["game_dt_et"], errors="coerce").dt.hour >= 16]
        return late if not late.empty else sunday
    if ed == "snf":
        if sunday.empty:
            return work.tail(1)
        snf = sunday[pd.to_datetime(sunday["game_dt_et"], errors="coerce").dt.hour >= 19]
        return snf.tail(1) if not snf.empty else sunday.tail(1)
    if ed == "monday":
        return monday if not monday.empty else work.tail(1)
    if ed == "tuesday":
        return work.copy()
    return work.copy()


def top_edges_for_game(edges: pd.DataFrame, game_id: str, matchup: str = "", limit: int = 5) -> pd.DataFrame:
    if edges.empty:
        return edges
    out = edges.copy()
    if game_id and "game_id" in out.columns:
        sub = out[out["game_id"].astype("string") == str(game_id)]
        if not sub.empty:
            out = sub
    elif matchup and "matchup" in out.columns:
        out = out[out["matchup"].astype("string") == str(matchup)]
    out = out.sort_values(["ev", "edge_pct", "p_win"], ascending=[False, False, False], na_position="last")
    return out.head(limit)


def top_props_for_game(markets: pd.DataFrame, edges: pd.DataFrame, game_id: str, matchup: str = "", limit: int = 4) -> pd.DataFrame:
    if markets.empty and edges.empty:
        return pd.DataFrame()
    pool = edges if not edges.empty else markets
    pool = pool.copy()
    if game_id and "game_id" in pool.columns:
        sub = pool[pool["game_id"].astype("string") == str(game_id)]
        if not sub.empty:
            pool = sub
    elif matchup and "matchup" in pool.columns:
        pool = pool[pool["matchup"].astype("string") == str(matchup)]
    if "market" in pool.columns:
        mask = ~pool["market"].astype("string").str.upper().isin(["H2H", "MONEYLINE", "SPREAD", "SPREADS", "TOTAL", "TOTALS"])
        pool = pool[mask]
    sort_cols = [c for c in ["ev", "edge_pct", "p_win", "odds"] if c in pool.columns]
    if sort_cols:
        pool = pool.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return pool.head(limit)


def best_market_snapshot(markets: pd.DataFrame, game_id: str, matchup: str = "") -> dict[str, str]:
    if markets.empty:
        return {}
    pool = markets.copy()
    if game_id and "game_id" in pool.columns:
        sub = pool[pool["game_id"].astype("string") == str(game_id)]
        if not sub.empty:
            pool = sub
    elif matchup and "matchup" in pool.columns:
        pool = pool[pool["matchup"].astype("string") == str(matchup)]
    if pool.empty:
        return {}

    def _pick(mask: pd.Series) -> Optional[pd.Series]:
        sub = pool[mask].copy()
        if sub.empty:
            return None
        sort_cols = [c for c in ["odds", "line"] if c in sub.columns]
        if sort_cols:
            sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
        return sub.iloc[0]

    out: dict[str, str] = {}
    market_up = pool["market"].astype("string").str.upper()
    side_row = _pick(market_up.isin(["H2H", "MONEYLINE", "SPREAD", "SPREADS"]))
    total_row = _pick(market_up.str.contains("TOTAL", na=False))
    if side_row is not None:
        desc = str(side_row.get("side", "")).strip()
        line = side_row.get("line")
        odds = side_row.get("odds")
        bits = [desc]
        if pd.notna(line):
            bits.append(f"{float(line):+.1f}")
        if pd.notna(odds):
            bits.append(f"({int(odds):+d})")
        out["side"] = " ".join(bits).strip()
    if total_row is not None:
        side = str(total_row.get("side", "Total")).strip()
        line = total_row.get("line")
        odds = total_row.get("odds")
        bits = [side]
        if pd.notna(line):
            bits.append(f"{float(line):.1f}")
        if pd.notna(odds):
            bits.append(f"({int(odds):+d})")
        out["total"] = " ".join(bits).strip()
    return out
