from __future__ import annotations

# ---- recovered app shims ----
try:
    from lib.compliance_gate import require_eligibility
except Exception:
    def require_eligibility(*args, **kwargs):
        return True

try:
    from lib.access import require_allowed_page, beta_banner, live_enabled, premium_enabled
except Exception:
    def require_allowed_page(*args, **kwargs):
        return None
    def beta_banner(*args, **kwargs):
        return None
    def live_enabled(*args, **kwargs):
        return False
    def premium_enabled(*args, **kwargs):
        return True

try:
    from app.lib.auth import login, show_logout
except Exception:
    def login(required: bool = False):
        class _Auth:
            ok = True
            authenticated = True
            username = None
        return _Auth()
    def show_logout():
        return None

try:
    from app.utils.diagnostics import mount_in_sidebar
except Exception:
    def mount_in_sidebar(page_name: str):
        return None

try:
    from app.utils.nudge import begin_session, touch_session, session_duration_str, bump_usage, show_nudge
except Exception:
    def begin_session():
        return None
    def touch_session():
        return None
    def session_duration_str():
        return ""
    def bump_usage(*args, **kwargs):
        return None
    def show_nudge(*args, **kwargs):
        return None
# ---- /recovered app shims ----

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

LIVE_START_YEAR = 2020
BUILD_TAG = "backtest-archive-live-v2.0"

st.set_page_config(page_title="Backtest — Scores Browser", layout="wide")
require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})

# session / nudge
begin_session()
touch_session()
if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")
bump_usage("page_visit")
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")


def _here() -> Path:
    return Path(__file__).resolve()


def _repo_root() -> Path:
    here = _here()
    for p in [here.parent] + list(here.parents):
        if (p / "streamlit_app.py").exists() or (p / "app").exists() or (p / "exports").exists():
            return p
    return Path.cwd()


@st.cache_data(show_spinner=False)
def _exports_dir() -> Path:
    try:
        env_val = st.secrets.get("EDGE_EXPORTS_DIR", "")
    except Exception:
        env_val = os.environ.get("EDGE_EXPORTS_DIR", "")

    if str(env_val).strip():
        p = Path(str(env_val).strip())
        if p.exists() and p.is_dir():
            return p

    repo = _repo_root()
    p = repo / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _exports_search_roots() -> list[Path]:
    root = _exports_dir()
    roots = [root]
    for sub in ["canonical", "data", "raw", "reports", "historical_odds"]:
        p = root / sub
        if p.exists() and p.is_dir():
            roots.append(p)
    return roots


def _recursive_csvs() -> list[Path]:
    found: list[Path] = []
    for root in _exports_search_roots():
        found.extend([p for p in root.rglob("*.csv") if p.is_file()])
    uniq = {str(p.resolve()): p for p in found}
    return sorted(uniq.values(), key=lambda p: (p.parent.as_posix(), p.name.lower()))


def _pick_latest_file(preferred_names: list[str], glob_patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for root in _exports_search_roots():
        for name in preferred_names:
            p = root / name
            if p.exists() and p.is_file():
                matches.append(p)
        for pattern in glob_patterns:
            matches.extend([p for p in root.rglob(pattern) if p.is_file()])
    if not matches:
        return None
    uniq = {str(p.resolve()): p for p in matches}
    return max(uniq.values(), key=lambda p: p.stat().st_mtime)


def _safe_read_csv(path: Path | None, label: str) -> pd.DataFrame:
    if path is None:
        st.error(f"No {label} CSV found in exports.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        df.attrs["source_path"] = str(path)
        return df
    except Exception as e:
        st.error(f"Could not load {label} file `{path}`: {e}")
        return pd.DataFrame()


def _s(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype("string").fillna("")
    return pd.Series([x], dtype="string")


def _as_str_series(df: pd.DataFrame, *cands: str, default: str = "") -> pd.Series:
    lower = {str(c).lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns:
            return _s(df[c])
        key = str(c).lower()
        if key in lower:
            return _s(df[lower[key]])
    return pd.Series([default] * len(df), index=df.index, dtype="string")


def _first_present(df: pd.DataFrame, cols: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in cols:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols, seen = [], {}
    for c in map(str, df.columns):
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    out = df.copy()
    out.columns = new_cols
    return out


def _ensure_date_iso(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "_date_iso" in df.columns:
        df["_date_iso"] = pd.to_datetime(df["_date_iso"], errors="coerce").dt.strftime("%Y-%m-%d")
        return df
    for cand in ("date", "game_date", "commence_time", "Date", "start_time", "scheduled", "event_date"):
        col = _first_present(df, [cand])
        if col:
            df["_date_iso"] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            return df
    df["_date_iso"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    return df


def _season_from_date_str(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series, errors="coerce")
    yr, mo = dt.dt.year, dt.dt.month
    return (yr - (mo <= 2).astype("int64")).astype("Int64")


def _clean_key(s: str) -> str:
    s = re.sub(r"[^A-Z0-9 ]+", "", str(s).upper()).strip()
    s = re.sub(r"\s+", "_", s)
    return s


TEAM_ALIASES = {
    "COMMANDERS": ["Washington Commanders", "Washington", "Commanders", "Washington Football Team", "Football Team", "WFT", "Redskins", "Skins", "WAS", "WSH", "WDC", "DC"],
    "COWBOYS": ["Dallas Cowboys", "Dallas", "Cowboys", "DAL", "DLS"],
    "EAGLES": ["Philadelphia Eagles", "Philadelphia", "Eagles", "PHI", "PHL"],
    "GIANTS": ["New York Giants", "NY Giants", "Giants", "NYG"],
    "PACKERS": ["Green Bay Packers", "Green Bay", "Packers", "GB", "GNB"],
    "BEARS": ["Chicago Bears", "Chicago", "Bears", "CHI"],
    "VIKINGS": ["Minnesota Vikings", "Minnesota", "Vikings", "MIN", "MINN"],
    "LIONS": ["Detroit Lions", "Detroit", "Lions", "DET"],
    "SAINTS": ["New Orleans Saints", "New Orleans", "Saints", "NO", "NOS", "NOLA"],
    "BUCCANEERS": ["Tampa Bay Buccaneers", "Tampa Bay", "Buccaneers", "Bucs", "TB", "TBB"],
    "FALCONS": ["Atlanta Falcons", "Atlanta", "Falcons", "ATL"],
    "PANTHERS": ["Carolina Panthers", "Carolina", "Panthers", "CAR"],
    "RAMS": ["Los Angeles Rams", "LA Rams", "Rams", "LAR", "St. Louis Rams", "St Louis Rams"],
    "49ERS": ["San Francisco 49ers", "49ers", "Niners", "San Francisco", "SF", "SFO"],
    "SEAHAWKS": ["Seattle Seahawks", "Seattle", "Seahawks", "SEA"],
    "CARDINALS": ["Arizona Cardinals", "Cardinals", "ARI"],
    "PATRIOTS": ["New England Patriots", "New England", "Patriots", "NE", "NWE"],
    "BILLS": ["Buffalo Bills", "Buffalo", "Bills", "BUF"],
    "JETS": ["New York Jets", "NY Jets", "Jets", "NYJ"],
    "DOLPHINS": ["Miami Dolphins", "Miami", "Dolphins", "MIA"],
    "RAVENS": ["Baltimore Ravens", "Baltimore", "Ravens", "BAL"],
    "BENGALS": ["Cincinnati Bengals", "Cincinnati", "Bengals", "CIN"],
    "BROWNS": ["Cleveland Browns", "Cleveland", "Browns", "CLE"],
    "STEELERS": ["Pittsburgh Steelers", "Pittsburgh", "Steelers", "PIT"],
    "COLTS": ["Indianapolis Colts", "Indianapolis", "Colts", "IND"],
    "TITANS": ["Tennessee Titans", "Tennessee", "Titans", "TEN"],
    "JAGUARS": ["Jacksonville Jaguars", "Jacksonville", "Jaguars", "Jags", "JAX", "JAC"],
    "TEXANS": ["Houston Texans", "Houston", "Texans", "HOU"],
    "CHIEFS": ["Kansas City Chiefs", "Kansas City", "Chiefs", "KC", "KCC"],
    "RAIDERS": ["Las Vegas Raiders", "Las Vegas", "Raiders", "LV", "LVR", "OAK"],
    "CHARGERS": ["Los Angeles Chargers", "LA Chargers", "Chargers", "LAC", "San Diego Chargers", "SD"],
    "BRONCOS": ["Denver Broncos", "Denver", "Broncos", "DEN"],
}
ALIAS: dict[str, str] = {}
for canon, names in TEAM_ALIASES.items():
    ALIAS[_clean_key(canon)] = canon
    for nm in names:
        ALIAS[_clean_key(nm)] = canon


@st.cache_data(show_spinner=False)
def _nickify(series: pd.Series) -> pd.Series:
    s = _s(series).astype("string")
    cleaned = s.str.upper().str.replace(r"[^A-Z0-9 ]+", "", regex=True).str.strip().str.replace(r"\s+", "_", regex=True)
    mapped = cleaned.replace(ALIAS)
    return mapped.str.replace(r"_+", "_", regex=True).str.strip("_")


def _ensure_nicks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "_home_nick" not in df.columns:
        col = _first_present(df, ["home", "home_team", "Home", "HOME", "home_team_bet"])
        df["_home_nick"] = _nickify(df[col]) if col else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    else:
        df["_home_nick"] = _nickify(df["_home_nick"])
    if "_away_nick" not in df.columns:
        col = _first_present(df, ["away", "away_team", "Away", "AWAY", "away_team_bet"])
        df["_away_nick"] = _nickify(df[col]) if col else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    else:
        df["_away_nick"] = _nickify(df["_away_nick"])
    return df


def _normalize_market_side(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    m = _as_str_series(x, "market_norm", "market", "bet_type").str.upper()
    m = (m.replace({"SPREADS": "SPREAD", "TOTALS": "TOTAL"})
           .mask(m.str.contains("MONEY|ML|H2H", na=False), "H2H")
           .mask(m.str.contains("SPREAD|HANDICAP|ATS", na=False), "SPREAD")
           .mask(m.str.contains("TOTAL|OU|O/U|O-U", na=False), "TOTAL"))

    if (m == "").all() or m.isna().all():
        inferred = pd.Series(["H2H"] * len(x), index=x.index, dtype="string")
        inferred = inferred.mask(pd.to_numeric(_as_str_series(x, "line", "spread", "handicap"), errors="coerce").notna(), "SPREAD")
        inferred = inferred.mask(_as_str_series(x, "selection", "side_norm", "side").str.contains("OVER|UNDER", case=False, na=False), "TOTAL")
        m = inferred
    x["market_norm"] = m.fillna("H2H")

    s = _as_str_series(x, "side_norm", "side", "selection", "selected_team", "selected_team_full").str.upper()
    s = s.mask(s.str.contains(r"\bOVER\b", na=False), "OVER")
    s = s.mask(s.str.contains(r"\bUNDER\b", na=False), "UNDER")
    home = _nickify(_as_str_series(x, "_home_nick", "home", "home_team", "home_team_bet"))
    away = _nickify(_as_str_series(x, "_away_nick", "away", "away_team", "away_team_bet"))
    s_team = _nickify(s)
    s = s.where(~s_team.eq(home), "HOME").where(~s_team.eq(away), "AWAY")
    x["side_norm"] = s

    x["line"] = pd.to_numeric(_as_str_series(x, "line", "handicap", "spread", "total", "points_total", "ou_line", "runner_line"), errors="coerce")
    x["odds"] = pd.to_numeric(_as_str_series(x, "odds", "price", "american", "american_odds", "us_odds", "closing_odds", "true_moneyline"), errors="coerce")
    return x


def _fix_side_labels(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    s = _as_str_series(x, "side_norm", "side", "selection", "selected_team", "selected_team_full").str.upper()
    home = _nickify(_as_str_series(x, "_home_nick", "home", "home_team", "home_team_bet"))
    away = _nickify(_as_str_series(x, "_away_nick", "away", "away_team", "away_team_bet"))
    s_team = _nickify(s)
    s = s.mask(s.str.contains(r"\bOVER\b", na=False), "OVER")
    s = s.mask(s.str.contains(r"\bUNDER\b", na=False), "UNDER")
    s = s.where(~s_team.eq(home), "HOME").where(~s_team.eq(away), "AWAY")
    x["side_norm"] = s
    return x


def _prep_for_settlement(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["home_score"] = pd.to_numeric(_as_str_series(x, "home_score", "HomeScore"), errors="coerce")
    x["away_score"] = pd.to_numeric(_as_str_series(x, "away_score", "AwayScore"), errors="coerce")
    x["line"] = pd.to_numeric(_as_str_series(x, "line", "spread", "total", "ou_line"), errors="coerce")
    x["odds"] = pd.to_numeric(_as_str_series(x, "odds", "price", "closing_odds", "true_moneyline"), errors="coerce")
    x["market_norm"] = _as_str_series(x, "market_norm", "market").str.upper().replace({"SPREADS": "SPREAD", "TOTALS": "TOTAL"})
    x["side_norm"] = _as_str_series(x, "side_norm", "side", "selection").str.upper()
    return x


def settle_rows(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_settled"] = False
    x["_result"] = pd.NA

    hs = pd.to_numeric(x["home_score"], errors="coerce")
    aw = pd.to_numeric(x["away_score"], errors="coerce")
    line = pd.to_numeric(x["line"], errors="coerce")
    market = _as_str_series(x, "market_norm", "market").str.upper()
    side = _as_str_series(x, "side_norm", "side", "selection").str.upper()
    total = hs + aw
    margin_home = hs - aw

    have_scores = hs.notna() & aw.notna()

    # H2H
    mask = have_scores & market.eq("H2H")
    if mask.any():
        home_win = margin_home > 0
        away_win = margin_home < 0
        x.loc[mask & side.eq("HOME") & home_win, "_result"] = "WIN"
        x.loc[mask & side.eq("HOME") & away_win, "_result"] = "LOSS"
        x.loc[mask & side.eq("AWAY") & away_win, "_result"] = "WIN"
        x.loc[mask & side.eq("AWAY") & home_win, "_result"] = "LOSS"
        x.loc[mask & (margin_home == 0), "_result"] = "PUSH"

    # SPREAD (line is from selected side perspective when available)
    mask = have_scores & market.eq("SPREAD") & line.notna()
    if mask.any():
        perf = np.where(side.eq("HOME"), margin_home + line, np.where(side.eq("AWAY"), -margin_home + line, np.nan))
        perf = pd.Series(perf, index=x.index, dtype="float")
        x.loc[mask & (perf > 0), "_result"] = "WIN"
        x.loc[mask & (perf < 0), "_result"] = "LOSS"
        x.loc[mask & (perf == 0), "_result"] = "PUSH"

    # TOTAL
    mask = have_scores & market.eq("TOTAL") & line.notna()
    if mask.any():
        x.loc[mask & side.eq("OVER") & (total > line), "_result"] = "WIN"
        x.loc[mask & side.eq("OVER") & (total < line), "_result"] = "LOSS"
        x.loc[mask & side.eq("UNDER") & (total < line), "_result"] = "WIN"
        x.loc[mask & side.eq("UNDER") & (total > line), "_result"] = "LOSS"
        x.loc[mask & (total == line), "_result"] = "PUSH"

    x["_settled"] = x["_result"].isin(["WIN", "LOSS", "PUSH"])
    x["total_points"] = total
    return x


@st.cache_data(show_spinner=True)
def discover_backtest_sources() -> dict[str, list[Path]]:
    files = _recursive_csvs()
    groups = {
        "edges": [],
        "scores": [],
        "odds": [],
    }
    for p in files:
        n = p.name.lower()
        if any(k in n for k in ["score", "scored", "games_master_template", "games_master", "fort_knox_market_joined_moneyline_scored"]):
            groups["scores"].append(p)
        if any(k in n for k in ["edge", "backtest_bets", "edges_master", "market_anchor", "moneyline_constrained_bets"]):
            groups["edges"].append(p)
        if any(k in n for k in ["odds", "lines", "closing_lines"]):
            groups["odds"].append(p)
    for k in groups:
        uniq = {str(p.resolve()): p for p in groups[k]}
        groups[k] = sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return groups


@st.cache_data(show_spinner=True)
def load_edges_scores(edges_path_str: str | None = None, scores_path_str: str | None = None):
    groups = discover_backtest_sources()
    edges_path = Path(edges_path_str) if edges_path_str else None
    scores_path = Path(scores_path_str) if scores_path_str else None

    if edges_path is None:
        edges_path = _pick_latest_file(
            preferred_names=[
                "backtest_bets_moneyline_constrained.csv",
                "backtest_bets_moneyline.csv",
                "edges_master.csv",
                "edges_market_anchor.csv",
                "edges_standardized.csv",
                "edges_graded_full_normalized_std.csv",
                "edges_graded_full.csv",
                "edges.csv",
            ],
            glob_patterns=[
                "*backtest_bets*moneyline*constrained*.csv",
                "*backtest_bets*moneyline*.csv",
                "*edges_master*.csv",
                "*edges_market_anchor*.csv",
                "*edges*.csv",
            ],
        )
        if edges_path is None and groups["edges"]:
            edges_path = groups["edges"][0]

    if scores_path is None:
        scores_path = _pick_latest_file(
            preferred_names=[
                "fort_knox_market_joined_moneyline_scored_all_seasons.csv",
                "fort_knox_market_joined_moneyline_scored.csv",
                "games_master_template.csv",
                "scores_normalized_std.csv",
                "scores_1966-2025.csv",
            ],
            glob_patterns=[
                "*fort_knox*market_joined*moneyline*scored*.csv",
                "*games_master*template*.csv",
                "*scores*normalized*.csv",
                "*scores*.csv",
                "*scored*.csv",
            ],
        )
        if scores_path is None and groups["scores"]:
            scores_path = groups["scores"][0]

    edges = _safe_read_csv(edges_path, "edges")
    scores = _safe_read_csv(scores_path, "scores")
    return edges, scores, edges_path, scores_path


ODDS_CANDIDATES = ["odds", "american", "price", "american_odds", "us_odds", "odds_american", "closing_odds", "true_moneyline"]


@st.cache_data(show_spinner=True)
def _load_odds_long(selected_path_str: str | None = None) -> pd.DataFrame | None:
    path = Path(selected_path_str) if selected_path_str else _pick_latest_file(
        preferred_names=["odds_lines_all.csv", "closing_lines.csv", "nfl_open_mid_close_odds.csv"],
        glob_patterns=["*odds_lines_all*.csv", "*closing_lines*.csv", "*open*mid*close*odds*.csv", "*odds*.csv", "*lines*.csv"],
    )
    if path is None or not path.exists():
        return None
    o = pd.read_csv(path, low_memory=False)
    o = _ensure_nicks(_ensure_date_iso(o))
    o = _normalize_market_side(o)
    home = _nickify(_as_str_series(o, "_home_nick", "home", "home_team"))
    away = _nickify(_as_str_series(o, "_away_nick", "away", "away_team"))
    o["_gk"] = (_as_str_series(o, "_date_iso") + "|" + home + "|" + away).astype("string")
    if "book" not in o.columns:
        o["book"] = "ODDS"
    return o[[c for c in ["_gk", "market_norm", "side_norm", "line", "odds", "book"] if c in o.columns]].rename(columns={"odds": "price"})


def attach_scores(edges: pd.DataFrame, scores: pd.DataFrame, year_min: int | None = None) -> tuple[pd.DataFrame, dict]:
    e = _ensure_nicks(_ensure_date_iso(edges.copy()))
    sc = _ensure_nicks(_ensure_date_iso(scores.copy()))
    e = _normalize_market_side(e)

    if year_min is not None:
        e_season = _season_from_date_str(_as_str_series(e, "_date_iso", "date", "game_date"))
        e = e.loc[e_season >= year_min].copy()

    # score columns
    sc["home_score"] = pd.to_numeric(_as_str_series(sc, "home_score", "HomeScore"), errors="coerce")
    sc["away_score"] = pd.to_numeric(_as_str_series(sc, "away_score", "AwayScore"), errors="coerce")
    sc["season"] = pd.to_numeric(_as_str_series(sc, "season", "Season"), errors="coerce")
    sc["week"] = pd.to_numeric(_as_str_series(sc, "week", "Week"), errors="coerce")

    e_key_fw = (_as_str_series(e, "_date_iso") + "|" + _nickify(_as_str_series(e, "_home_nick", "home", "home_team", "home_team_bet")) + "|" + _nickify(_as_str_series(e, "_away_nick", "away", "away_team", "away_team_bet"))).astype("string")
    e_key_sw = (_as_str_series(e, "_date_iso") + "|" + _nickify(_as_str_series(e, "_away_nick", "away", "away_team", "away_team_bet")) + "|" + _nickify(_as_str_series(e, "_home_nick", "home", "home_team", "home_team_bet"))).astype("string")
    sc_key_fw = (_as_str_series(sc, "_date_iso") + "|" + _nickify(_as_str_series(sc, "_home_nick", "home", "home_team")) + "|" + _nickify(_as_str_series(sc, "_away_nick", "away", "away_team"))).astype("string")

    sc_small = sc.copy()
    sc_small["_sc_key"] = sc_key_fw
    sc_small = sc_small.sort_values("_date_iso").drop_duplicates(subset=["_sc_key"], keep="last")

    out = e.copy()
    out["_join_key_fw"] = e_key_fw
    out["_join_key_sw"] = e_key_sw

    lookup = sc_small.set_index("_sc_key")[["home_score", "away_score", "season", "week", "_date_iso"]]
    out[["home_score", "away_score", "season", "week", "_date_iso_sc"]] = lookup.reindex(out["_join_key_fw"]).reset_index(drop=True)

    miss = out["home_score"].isna() | out["away_score"].isna()
    if miss.any():
        rev = lookup.rename(columns={"home_score": "away_score", "away_score": "home_score"})
        fill = rev.reindex(out.loc[miss, "_join_key_sw"]).reset_index(drop=True)
        out.loc[miss, ["home_score", "away_score", "season", "week", "_date_iso_sc"]] = fill[["home_score", "away_score", "season", "week", "_date_iso"]].values

    if out["season"].isna().all():
        out["season"] = _season_from_date_str(_as_str_series(out, "_date_iso", "_date_iso_sc"))

    out["total_points"] = pd.to_numeric(out["home_score"], errors="coerce") + pd.to_numeric(out["away_score"], errors="coerce")
    with_scores = out["home_score"].notna() & out["away_score"].notna()
    stats = {
        "rows": int(len(out)),
        "with_scores": int(with_scores.sum()),
        "coverage_pct": float(with_scores.mean() * 100 if len(out) else 0.0),
    }
    return out.drop(columns=["_join_key_fw", "_join_key_sw"], errors="ignore"), stats


def fill_scores_by_nearest_date(joined: pd.DataFrame, scores: pd.DataFrame, max_days: int = 3) -> pd.DataFrame:
    j = _ensure_nicks(_ensure_date_iso(joined.copy()))
    sc = _ensure_nicks(_ensure_date_iso(scores.copy()))
    if j.empty or sc.empty:
        return j

    def pair_key(h, a):
        h1 = _nickify(h)
        a1 = _nickify(a)
        return np.where(h1 <= a1, h1 + "|" + a1, a1 + "|" + h1)

    j["_pair"] = pair_key(_as_str_series(j, "_home_nick", "home", "home_team", "home_team_bet"), _as_str_series(j, "_away_nick", "away", "away_team", "away_team_bet"))
    sc["_pair"] = pair_key(_as_str_series(sc, "_home_nick", "home", "home_team"), _as_str_series(sc, "_away_nick", "away", "away_team"))
    sc["home_score"] = pd.to_numeric(_as_str_series(sc, "home_score", "HomeScore"), errors="coerce")
    sc["away_score"] = pd.to_numeric(_as_str_series(sc, "away_score", "AwayScore"), errors="coerce")
    sc["season"] = pd.to_numeric(_as_str_series(sc, "season", "Season"), errors="coerce")
    sc["week"] = pd.to_numeric(_as_str_series(sc, "week", "Week"), errors="coerce")

    j["_dt"] = pd.to_datetime(_as_str_series(j, "_date_iso"), errors="coerce")
    sc["_dt"] = pd.to_datetime(_as_str_series(sc, "_date_iso"), errors="coerce")
    miss = j["home_score"].isna() | j["away_score"].isna()
    if not miss.any():
        return j.drop(columns=["_pair", "_dt"], errors="ignore")

    cand = j.loc[miss, ["_pair", "_dt"]].merge(sc[["_pair", "_dt", "home_score", "away_score", "season", "week"]], on="_pair", how="left")
    cand["abs_diff"] = (cand["_dt_x"] - cand["_dt_y"]).abs()
    cand = cand[cand["abs_diff"] <= pd.Timedelta(days=max_days)]
    if cand.empty:
        return j.drop(columns=["_pair", "_dt"], errors="ignore")

    cand = cand.sort_values(["_dt_x", "abs_diff"]).drop_duplicates(subset=["_pair", "_dt_x"], keep="first")
    cand.index = pd.MultiIndex.from_frame(cand[["_pair", "_dt_x"]])
    idx = pd.MultiIndex.from_frame(j.loc[miss, ["_pair", "_dt"]])
    j.loc[miss, "home_score"] = cand.reindex(idx)["home_score"].values
    j.loc[miss, "away_score"] = cand.reindex(idx)["away_score"].values
    j.loc[miss, "season"] = cand.reindex(idx)["season"].values
    j.loc[miss, "week"] = cand.reindex(idx)["week"].values
    return j.drop(columns=["_pair", "_dt"], errors="ignore")


def _backfill_lines_and_odds(bets: pd.DataFrame, all_odds: pd.DataFrame | None) -> pd.DataFrame:
    x = bets.copy()
    if all_odds is None or all_odds.empty or x.empty:
        return x
    x["line"] = pd.to_numeric(x.get("line"), errors="coerce")
    x["odds"] = pd.to_numeric(x.get("odds"), errors="coerce")
    if "book" not in x.columns:
        x["book"] = pd.NA
    o = all_odds.copy()
    o["line"] = pd.to_numeric(o.get("line"), errors="coerce")
    o["price"] = pd.to_numeric(o.get("price"), errors="coerce")

    by_side = o.drop_duplicates(["_gk", "market_norm", "side_norm"], keep="last").rename(columns={"line": "line_fill", "price": "price_fill", "book": "book_fill"})
    x = x.merge(by_side, how="left", on=["_gk", "market_norm", "side_norm"])
    x["line"] = x["line"].where(x["line"].notna(), x["line_fill"])
    x["odds"] = x["odds"].where(x["odds"].notna(), x["price_fill"])
    x["book"] = x["book"].where(x["book"].notna(), x["book_fill"])
    return x.drop(columns=[c for c in ["line_fill", "price_fill", "book_fill"] if c in x.columns], errors="ignore")


# ---------- UI ----------
st.title("Backtest — Scores Browser")
st.caption(f"Build: {BUILD_TAG}")
mount_in_sidebar("Backtest")

if st.button("🔄 Refresh Scores (pull live)"):
    if not os.environ.get("THE_ODDS_API_KEY"):
        st.error("Missing THE_ODDS_API_KEY. Set it in your environment before refreshing.")
    else:
        result = subprocess.run([sys.executable, "tools/pull_scores.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Scores refreshed.")
            if result.stdout.strip():
                st.code(result.stdout)
        else:
            st.error(f"Refresh failed (exit {result.returncode}).")
            if result.stderr.strip():
                st.code(result.stderr)
            elif result.stdout.strip():
                st.code(result.stdout)

sources = discover_backtest_sources()
all_edges = sources["edges"]
all_scores = sources["scores"]
all_odds = sources["odds"]

with st.sidebar:
    st.markdown("### Backtest Sources")
    st.caption(f"Using exports dir: {_exports_dir()}")
    st.write({
        "edges_csvs": len(all_edges),
        "scores_csvs": len(all_scores),
        "odds_csvs": len(all_odds),
    })

edge_choice = st.selectbox(
    "Edges / bets CSV",
    options=[str(p) for p in all_edges] if all_edges else [""],
    index=0,
    format_func=lambda s: Path(s).name if s else "No edges file found",
)
score_choice = st.selectbox(
    "Scores CSV",
    options=[str(p) for p in all_scores] if all_scores else [""],
    index=0,
    format_func=lambda s: Path(s).name if s else "No scores file found",
)
odds_choice = st.selectbox(
    "Odds / lines CSV",
    options=[str(p) for p in all_odds] if all_odds else [""],
    index=0,
    format_func=lambda s: Path(s).name if s else "No odds file found",
)

edges, scores, edges_path, scores_path = load_edges_scores(edge_choice or None, score_choice or None)
if edges.empty or scores.empty:
    st.stop()

joined, stats_all = attach_scores(edges, scores, year_min=LIVE_START_YEAR)
joined = fill_scores_by_nearest_date(joined, scores, max_days=3)
joined = _fix_side_labels(_normalize_market_side(joined))
season_from_date = _season_from_date_str(_as_str_series(joined, "_date_iso", "_date_iso_sc"))
joined["season_ui"] = pd.to_numeric(joined.get("season"), errors="coerce").fillna(season_from_date).astype("Int64")
joined["week"] = pd.to_numeric(joined.get("week"), errors="coerce").astype("Int64")

_dt = _as_str_series(joined, "_date_iso")
_hm = _nickify(_as_str_series(joined, "_home_nick", "home", "home_team", "home_team_bet"))
_aw = _nickify(_as_str_series(joined, "_away_nick", "away", "away_team", "away_team_bet"))
joined["_gk"] = (_dt + "|" + _hm + "|" + _aw).astype("string")

odds_long = _load_odds_long(odds_choice or None)
joined = _backfill_lines_and_odds(joined, odds_long)
archive = joined.copy()

st.info(
    f"Join coverage (2020+): {stats_all['with_scores']:,}/{stats_all['rows']:,} ({stats_all['coverage_pct']:.1f}%) matched with scores."
)
st.success(f"Loaded edges={len(edges):,} from {edges_path.name if edges_path else '—'} • scores={len(scores):,} from {scores_path.name if scores_path else '—'}")

# High-value controls
st.markdown("### Filter: Backtest Range")
seasons_all = pd.to_numeric(joined.get("season_ui"), errors="coerce").dropna().astype(int).unique().tolist()
seasons_all = sorted(seasons_all)
default_seasons = [s for s in seasons_all if s >= LIVE_START_YEAR] or seasons_all[-3:]

c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
with c1:
    preset = st.selectbox("Range preset", ["2020+", "Last 3 seasons", "Single season", "Custom"], index=0)
with c2:
    market_filter = st.multiselect("Markets", options=sorted(joined["market_norm"].dropna().astype(str).unique().tolist()), default=sorted(joined["market_norm"].dropna().astype(str).unique().tolist()))
with c3:
    settled_only = st.toggle("Settled only", value=True)
with c4:
    only_positive_profit = st.toggle("Winners only", value=False)

if preset == "2020+":
    sel_seasons = [s for s in seasons_all if s >= 2020]
elif preset == "Last 3 seasons":
    sel_seasons = seasons_all[-3:] if len(seasons_all) >= 3 else seasons_all
elif preset == "Single season":
    pick = st.selectbox("Season", seasons_all, index=len(seasons_all) - 1 if seasons_all else 0)
    sel_seasons = [pick]
else:
    sel_seasons = st.multiselect("Seasons", seasons_all, default=default_seasons)

work = joined.copy()
if sel_seasons:
    work = work[pd.to_numeric(work["season_ui"], errors="coerce").isin(sel_seasons)].copy()
if market_filter:
    work = work[work["market_norm"].isin(market_filter)].copy()

weeks_all = pd.to_numeric(work.get("week"), errors="coerce").dropna().astype(int).sort_values().unique().tolist()
sel_weeks = st.multiselect("Weeks", weeks_all, default=weeks_all)
if sel_weeks:
    work = work[pd.to_numeric(work["week"], errors="coerce").isin(sel_weeks)].copy()

team_opts = sorted(set(_as_str_series(work, "_home_nick").dropna().tolist()) | set(_as_str_series(work, "_away_nick").dropna().tolist()))
team_filter = st.multiselect("Teams", team_opts, default=[])
if team_filter:
    work = work[_as_str_series(work, "_home_nick").isin(team_filter) | _as_str_series(work, "_away_nick").isin(team_filter)].copy()

work = _prep_for_settlement(work)
work = _fix_side_labels(work)
base = settle_rows(work.copy())
if settled_only:
    base = base[base["_settled"]].copy()
if only_positive_profit and "profit" in base.columns:
    base = base[pd.to_numeric(base["profit"], errors="coerce") > 0].copy()

st.info(f"Rows in memory (post-join): {len(joined):,} • Filtered view: {len(base):,}")

# P&L and model metrics
stake = st.number_input("Stake per bet ($)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
view = base.copy()
odds_col = _first_present(view, ODDS_CANDIDATES) or "odds"
view["_american"] = pd.to_numeric(view.get(odds_col), errors="coerce")
view["_stake"] = float(stake)
view["_pnl"] = pd.to_numeric(view.get("profit"), errors="coerce")

mask = view["_result"].isin(["WIN", "LOSS", "PUSH"])
amer = view.loc[mask, "_american"].fillna(0.0).astype(float)
res = view.loc[mask, "_result"]
win_payout = np.where(amer > 0, stake * (amer / 100.0), stake * (100.0 / np.maximum(1.0, np.abs(amer))))
auto_pnl = np.where(res == "PUSH", 0.0, np.where(res == "WIN", win_payout, -stake))
view.loc[mask & view["_pnl"].isna(), "_pnl"] = auto_pnl

bets = int(mask.sum())
wins = int((view["_result"] == "WIN").sum())
push = int((view["_result"] == "PUSH").sum())
net = float(np.nansum(view["_pnl"]))
roi = net / max(1.0, (bets * stake))

# backtest utility metrics
clv = pd.to_numeric(view.get("close_prob"), errors="coerce") - pd.to_numeric(view.get("implied_prob"), errors="coerce")
beat_close = (clv > 0).mean() if clv.notna().any() else np.nan
edge_vs_close = pd.to_numeric(view.get("model_edge_vs_close"), errors="coerce")
if edge_vs_close.isna().all() and {"model_prob", "close_prob"}.issubset(view.columns):
    edge_vs_close = pd.to_numeric(view["model_prob"], errors="coerce") - pd.to_numeric(view["close_prob"], errors="coerce")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Bets", f"{bets:,}")
m2.metric("Win %", f"{(wins / max(1, bets - push) * 100):.1f}%")
m3.metric("Net P&L", f"${net:,.2f}")
m4.metric("ROI / bet", f"{roi*100:.2f}%")
m5.metric("Avg CLV", f"{clv.mean()*100:.2f}%" if clv.notna().any() else "—")
m6.metric("Beat Close", f"{beat_close*100:.1f}%" if pd.notna(beat_close) else "—")

# grouped rollups
st.markdown("### Rollups")
group_by = st.multiselect("Group by", ["season_ui", "week", "market_norm", "book", "side_norm"], default=["season_ui", "market_norm"])
group_cols = [c for c in group_by if c in view.columns]
if group_cols:
    g = (
        view.assign(
            _bet=view["_result"].isin(["WIN", "LOSS", "PUSH"]).astype(int),
            _win=(view["_result"] == "WIN").astype(int),
            _push=(view["_result"] == "PUSH").astype(int),
        )
        .groupby(group_cols, dropna=False)
        .agg(
            bets=("_bet", "sum"),
            wins=("_win", "sum"),
            pushes=("_push", "sum"),
            pnl=("_pnl", "sum"),
            avg_odds=("_american", "mean"),
            avg_edge_vs_close=(edge_vs_close.name if getattr(edge_vs_close, 'name', None) in view.columns else "_american", "mean"),
        )
        .reset_index()
    )
    g["win_pct"] = g["wins"] / np.maximum(1, g["bets"] - g["pushes"]) * 100.0
    g["roi_%"] = g["pnl"] / np.maximum(1.0, g["bets"] * stake) * 100.0
    st.dataframe(g, use_container_width=True, hide_index=True)

# useful tabs
settled_df = view[view["_settled"]].copy()
na_df = view[~view["_settled"]].copy()
needs_line_df = view[view["market_norm"].isin(["SPREAD", "TOTAL"]) & pd.to_numeric(view.get("line"), errors="coerce").isna()].copy()

tab1, tab2, tab3, tab4 = st.tabs(["Results", "Diagnostics", "Downloads", "Source Files"])
with tab1:
    pref = [c for c in ["season_ui", "week", "_date_iso", "_home_nick", "_away_nick", "market_norm", "side_norm", "line", "odds", "home_score", "away_score", "total_points", "_result", "_pnl", "book"] if c in view.columns]
    show = _ensure_unique_columns(view)
    st.dataframe(show[pref + [c for c in show.columns if c not in pref]], use_container_width=True, hide_index=True)

with tab2:
    by_res = view["_result"].value_counts(dropna=False).rename_axis("result").reset_index(name="n")
    st.dataframe(by_res, use_container_width=True, hide_index=True)
    counts = pd.DataFrame({
        "diagnostic": ["unsettled_rows", "needs_line", "missing_scores"],
        "rows": [len(na_df), len(needs_line_df), int((view["home_score"].isna() | view["away_score"].isna()).sum())],
    })
    st.dataframe(counts, use_container_width=True, hide_index=True)

with tab3:
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Settled rows", settled_df.to_csv(index=False).encode("utf-8"), file_name="settled_rows.csv", mime="text/csv")
    c2.download_button("Unsettled rows", na_df.to_csv(index=False).encode("utf-8"), file_name="unsettled_rows.csv", mime="text/csv")
    c3.download_button("Needs line", needs_line_df.to_csv(index=False).encode("utf-8"), file_name="needs_line_rows.csv", mime="text/csv")
    c4.download_button("Full joined archive", archive.to_csv(index=False).encode("utf-8"), file_name="backtest_archive_joined.csv", mime="text/csv")

with tab4:
    source_tbl = pd.DataFrame({
        "type": ["edges", "scores", "odds"],
        "selected_file": [str(edges_path) if edges_path else "", str(scores_path) if scores_path else "", odds_choice or ""],
    })
    st.dataframe(source_tbl, use_container_width=True, hide_index=True)
    with st.expander("Discovered CSV files"):
        st.write({
            "edges": [str(p) for p in all_edges[:25]],
            "scores": [str(p) for p in all_scores[:25]],
            "odds": [str(p) for p in all_odds[:25]],
        })
