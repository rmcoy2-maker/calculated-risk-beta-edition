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

def do_expensive_refresh():
    return None

try:
    from app.lib.auth import login, show_logout
except Exception:
    def login(required: bool = False):
        class _Auth:
            ok = True
            authenticated = True
        return _Auth()
    def show_logout():
        return None

try:
    from app.utils.diagnostics import mount_in_sidebar
except Exception:
    def mount_in_sidebar(page_name: str):
        return None

try:
    from app.utils.parlay_ui import selectable_odds_table
except Exception:
    def selectable_odds_table(*args, **kwargs):
        return None

try:
    from app.utils.parlay_cart import read_cart, add_to_cart, clear_cart
except Exception:
    import pandas as _shim_pd
    def read_cart():
        return _shim_pd.DataFrame()
    def add_to_cart(*args, **kwargs):
        return None
    def clear_cart():
        return None

try:
    from app.utils.nudge import begin_session, touch_session, session_duration_str, bump_usage, show_nudge
except Exception:
    def begin_session(): return None
    def touch_session(): return None
    def session_duration_str(): return ""
    def bump_usage(*args, **kwargs): return None
    def show_nudge(*args, **kwargs): return None
# ---- /recovered app shims ----

import numpy as np

import re
import pandas as pd
try:
    from lib.compliance_gate import require_eligibility
except Exception:
    def require_eligibility(*args, **kwargs):
        return True

import streamlit as st
import subprocess
import sys


# ---------------- Config ----------------
LIVE_START_YEAR = 2020
BUILD_TAG = "backtest-archive-live-v1.8"

st.set_page_config(page_title="Backtest — Scores Browser", layout="wide")
from pathlib import Path
require_eligibility(min_age=18, restricted_states={"WA","ID","NV"})





# === Nudge+Session (auto-injected) ===
try:
    from app.utils.nudge import begin_session, touch_session, session_duration_str, bump_usage, show_nudge  # type: ignore
except Exception:
    def begin_session(): pass
    def touch_session(): pass
    def session_duration_str(): return ""
    bump_usage = lambda *a, **k: None
    def show_nudge(*a, **k): pass

# Initialize/refresh session and show live duration
begin_session()
touch_session()
if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")

# Count a lightweight interaction per page load
bump_usage("page_visit")

# Optional upsell banner after threshold interactions in last 24h
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")
# === /Nudge+Session (auto-injected) ===

# === Nudge (auto-injected) ===
try:
    from app.utils.nudge import bump_usage, show_nudge  # type: ignore
except Exception:
    bump_usage = lambda *a, **k: None
    def show_nudge(*a, **k): pass

# Count a lightweight interaction per page load
bump_usage("page_visit")

# Show a nudge once usage crosses threshold in the last 24h
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")
# === /Nudge (auto-injected) ===

# Optional: pull fresh scores (your script)
if st.button("🔄 Refresh Scores (pull live)"):
    import os

    if not os.environ.get("THE_ODDS_API_KEY"):
        st.error("Missing THE_ODDS_API_KEY. Set it in your environment before refreshing.")
    else:
        result = subprocess.run(
            [sys.executable, "tools/pull_scores.py"],
            capture_output=True,
            text=True,
        )
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
# ---------------- Paths ----------------
def _here() -> Path:
    return Path(__file__).resolve()

@st.cache_data(show_spinner=False)
def _exports_dir() -> Path:
    here = _here()
    for up in [here.parent] + list(here.parents):
        if up.name.lower() == "edge-finder":
            p = up / "exports"
            p.mkdir(parents=True, exist_ok=True)
            return p
    p = here.parent / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------- Helpers ----------------
def _s(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype("string").fillna("")
    return pd.Series([x], dtype="string")

def _as_str_series(df: pd.DataFrame, *cands: str, default: str = "") -> pd.Series:
    for c in cands:
        if c in df.columns:
            return _s(df[c])
    # length-aware default series
    try:
        n = len(df)
        return pd.Series([default] * n, index=getattr(df, "index", None), dtype="string")
    except Exception:
        return pd.Series([default], dtype="string")

def _ensure_date_iso(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "_date_iso" not in df.columns:
        for cand in ("date", "game_date", "commence_time", "Date", "start_time", "scheduled", "event_date"):
            if cand in df.columns:
                df["_date_iso"] = pd.to_datetime(df[cand], errors="coerce").dt.strftime("%Y-%m-%d")
                break
    else:
        df["_date_iso"] = pd.to_datetime(df["_date_iso"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

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

def _season_from_date_str(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series, errors="coerce")
    yr, mo = dt.dt.year, dt.dt.month
    return (yr - (mo <= 2).astype("int64")).astype("Int64")

# ---------- Team alias normalization (full NFL) ----------
def _clean_key(s: str) -> str:
    s = re.sub(r"[^A-Z0-9 ]+", "", str(s).upper()).strip()
    s = re.sub(r"\s+", "_", s)
    return s

TEAM_ALIASES = {
    "COMMANDERS": ["Washington Commanders","Washington","Commanders","Washington Football Team","Football Team","WFT","Redskins","Skins","WAS","WSH","WDC","DC"],
    "COWBOYS":    ["Dallas Cowboys","Dallas","Cowboys","America's Team","Americas Team","DAL","DLS"],
    "EAGLES":     ["Philadelphia Eagles","Philadelphia","Eagles","PHI","PHL"],
    "GIANTS":     ["New York Giants","NY Giants","Giants","NYG","NYGiant","New York NYG"],

    "PACKERS": ["Green Bay Packers","Green Bay","Packers","GB","GNB","G Bay"],
    "BEARS":   ["Chicago Bears","Chicago","Bears","CHI"],
    "VIKINGS": ["Minnesota Vikings","Minnesota","Vikings","MIN","MINN"],
    "LIONS":   ["Detroit Lions","Detroit","Lions","DET"],

    "SAINTS":     ["New Orleans Saints","New Orleans","Saints","NO","NOS","NOLA","N.O.","N O"],
    "BUCCANEERS": ["Tampa Bay Buccaneers","Tampa Bay","Buccaneers","Bucs","TB","TBB","TAMPA"],
    "FALCONS":    ["Atlanta Falcons","Atlanta","Falcons","ATL"],
    "PANTHERS":   ["Carolina Panthers","Carolina","Panthers","CAR"],

    "RAMS":     ["Los Angeles Rams","LA Rams","Rams","LAR","St. Louis Rams","St Louis Rams","St Louis","Los Angelos Rams","LA","L A"],
    "49ERS":    ["San Francisco 49ers","SF 49ers","Forty Niners","Forty-Niners","49ers","Niners","San Francisco","SF","SFO"],
    "SEAHAWKS": ["Seattle Seahawks","Seattle","Seahawks","SEA","SEATTLE SEAHAWKS"],
    "CARDINALS":["Arizona Cardinals","Cardinals","AZ Cardinals","ARI","Phoenix Cardinals","Phoenix","St. Louis Cardinals","St Louis Cardinals","St Louis"],

    "PATRIOTS": ["New England Patriots","New England","Patriots","NE","NWE","N.E."],
    "BILLS":    ["Buffalo Bills","Buffalo","Bills","BUF"],
    "JETS":     ["New York Jets","NY Jets","Jets","NYJ"],
    "DOLPHINS": ["Miami Dolphins","Miami","Dolphins","MIA"],

    "RAVENS":  ["Baltimore Ravens","Baltimore","Ravens","BAL"],
    "BENGALS": ["Cincinnati Bengals","Cincinnati","Bengals","CIN"],
    "BROWNS":  ["Cleveland Browns","Cleveland","Browns","CLE"],
    "STEELERS":["Pittsburgh Steelers","Pittsburgh","Steelers","PIT"],

    "COLTS":   ["Indianapolis Colts","Indianapolis","Colts","IND","Baltimore Colts"],
    "TITANS":  ["Tennessee Titans","Tennessee","Titans","TEN","TN","Tennessee Oilers","Houston Oilers","Oilers","HOU Oilers"],
    "JAGUARS": ["Jacksonville Jaguars","Jacksonville","Jaguars","Jags","JAX","JAC"],
    "TEXANS":  ["Houston Texans","Houston","Texans","HOU"],

    "CHIEFS":   ["Kansas City Chiefs","Kansas City","KC Chiefs","Chiefs","KC","KCC","KANSAS_CITY"],
    "RAIDERS":  ["Las Vegas Raiders","Las Vegas","LV","Raiders","Oakland Raiders","Oakland","Los Angeles Raiders","LA Raiders","Los Angeles","LVR","OAK","L.A."],
    "CHARGERS": ["Los Angeles Chargers","LA Chargers","Chargers","LAC","San Diego Chargers","San Diego","SD","LA","L.A."],
    "BRONCOS":  ["Denver Broncos","Denver","Broncos","DEN"],
}

# Build alias lookup (cleaned -> canonical short)
ALIAS: dict[str, str] = {}
for canon, names in TEAM_ALIASES.items():
    ALIAS[_clean_key(canon)] = canon
    for nm in names:
        ALIAS[_clean_key(nm)] = canon

@st.cache_data(show_spinner=False)
def _nickify(series: pd.Series) -> pd.Series:
    s = _s(series).astype("string")
    cleaned = (
        s.str.upper()
         .str.replace(r"[^A-Z0-9 ]+", "", regex=True)
         .str.strip()
         .str.replace(r"\s+", "_", regex=True)
    )
    mapped = cleaned.replace(ALIAS)
    mapped = mapped.str.replace(r"_+", "_", regex=True).str.strip("_")
    return mapped

def _ensure_nicks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "_home_nick" not in df.columns:
        for col in ("home", "home_team", "Home", "HOME"):
            if col in df.columns:
                df["_home_nick"] = _nickify(df[col]); break
    else:
        df["_home_nick"] = _nickify(df["_home_nick"])
    if "_away_nick" not in df.columns:
        for col in ("away", "away_team", "Away", "AWAY"):
            if col in df.columns:
                df["_away_nick"] = _nickify(df[col]); break
    else:
        df["_away_nick"] = _nickify(df["_away_nick"])
    return df

# ---------- Odds normalization + key helpers ----------
def _game_key(d_s, h_s, a_s):
    d = _as_str_series(pd.DataFrame(index=range(len(d_s))) if not isinstance(d_s, pd.Series) else pd.DataFrame(index=d_s.index), d_s.name if isinstance(d_s, pd.Series) else None)
    d = _as_str_series(pd.DataFrame({"_": d_s}) if not isinstance(d_s, pd.Series) else pd.DataFrame({d_s.name or "_": d_s}), d_s.name or "_")
    d = d.astype("string")
    h = _nickify(_as_str_series(pd.DataFrame({"_": h_s}) if not isinstance(h_s, pd.Series) else pd.DataFrame({h_s.name or "_": h_s}), h_s.name or "_"))
    a = _nickify(_as_str_series(pd.DataFrame({"_": a_s}) if not isinstance(a_s, pd.Series) else pd.DataFrame({a_s.name or "_": a_s}), a_s.name or "_"))
    return (d + "|" + h + "|" + a).astype("string")

def _normalize_market_side(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    m = _as_str_series(x, "market_norm", "market").str.upper()
    m = (m.replace({"SPREADS":"SPREAD", "TOTALS":"TOTAL"})
           .mask(m.str.contains("MONEY|ML|H2H"), "H2H")
           .mask(m.str.contains("SPREAD|HANDICAP|ATS"), "SPREAD")
           .mask(m.str.contains("TOTAL|OU|O/U|O-U"), "TOTAL"))
    x["market_norm"] = m

    s = _as_str_series(x, "side_norm", "side", "selection").str.upper()
    s = s.mask(s.str.contains(r"\bOVER\b"), "OVER")
    s = s.mask(s.str.contains(r"\bUNDER\b"), "UNDER")

    home = _nickify(_as_str_series(x, "_home_nick", "home", "home_team"))
    away = _nickify(_as_str_series(x, "_away_nick", "away", "away_team"))
    s_team = _nickify(s)
    s = s.where(~s_team.eq(home), "HOME").where(~s_team.eq(away), "AWAY")
    x["side_norm"] = s

    x["line"] = pd.to_numeric(_as_str_series(x, "line","handicap","spread","total","points_total","ou_line","runner_line"), errors="coerce")
    x["odds"] = pd.to_numeric(_as_str_series(x, "odds","price","american","american_odds","us_odds"), errors="coerce")
    return x

def _mk_gk_columns(x: pd.DataFrame, date_col="_date_iso"):
    d = _as_str_series(x, date_col)
    home = _nickify(_as_str_series(x, "_home_nick", "home", "home_team"))
    away = _nickify(_as_str_series(x, "_away_nick", "away", "away_team"))
    gk0 = (d + "|" + home + "|" + away).astype("string")
    dt = pd.to_datetime(d, errors="coerce")
    gkm1 = ((dt - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d") + "|" + home + "|" + away).astype("string")
    gkp1 = ((dt + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d") + "|" + home + "|" + away).astype("string")
    return gk0, gkm1, gkp1

# ---------------- Data Load ----------------
@st.cache_data(show_spinner=True)
def load_edges_scores(edges_path: Path | None = None, scores_path: Path | None = None):
    exp = _exports_dir()

    if scores_path is None:
        candidates = (
            sorted(exp.glob("scores_normalized_std_maxaligned*.csv"))
            or sorted(exp.glob("scores_normalized_std*.csv"))
        )
        scores_path = candidates[-1] if candidates else (exp / "scores_1966-2025.csv")

    edges_path = edges_path or (exp / "edges_graded_full_normalized_std.csv")

    if not edges_path.exists():
        edge_candidates = (
            sorted(exp.glob("edges*.csv"))
            or sorted(exp.glob("*edges*.csv"))
        )
        if edge_candidates:
            edges_path = edge_candidates[-1]
            st.warning(f"Using fallback edges file: {edges_path.name}")
        else:
            st.error("No edges CSV found in exports.")
            st.stop()

    if not scores_path.exists():
        st.error(f"Scores file not found: {scores_path}")
        st.stop()

    edges = pd.read_csv(edges_path)
    scores = pd.read_csv(scores_path)
    return edges, scores

# ---------------- Odds helpers ----------------
ODDS_CANDIDATES = ["odds", "american", "price", "american_odds", "us_odds", "odds_american"]

def _first_present(df: pd.DataFrame, cols: list[str]) -> str | None:
    for c in cols:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=True)
def _load_odds_long() -> pd.DataFrame | None:
    """Load API odds from exports/odds_lines_all.csv and normalize to (_gk, market_norm, side_norm, line, price, book)."""
    exp = _exports_dir()
    path = exp / "odds_lines_all.csv"
    if not path.exists():
        return None

    o = pd.read_csv(path)
    o = _ensure_nicks(_ensure_date_iso(o))

    date_raw = _as_str_series(o, "_date_iso", "date", "game_date", "commence_time", "Date",
                              "start_time", "start", "event_date", "EventDate", "match_date", "scheduled")
    o["_date_iso_tmp"] = pd.to_datetime(date_raw, errors="coerce").dt.strftime("%Y-%m-%d")
    if o["_date_iso_tmp"].isna().all():
        return None

    m = _as_str_series(o, "_market_norm", "market_norm", "market").str.upper()
    m = (m.mask(m.str.contains("MONEY|ML|H2H"), "H2H")
           .mask(m.str.contains("SPREAD|HANDICAP|ATS"), "SPREAD")
           .mask(m.str.contains("TOTAL|OU|O/U|O-U"), "TOTAL")
           .fillna("")
           .astype("string"))
    o["market_norm"] = m.replace({"SPREADS": "SPREAD", "TOTALS": "TOTAL"})

    s = _as_str_series(o, "side_norm", "side").str.upper()
    sh = _nickify(s)
    home = _nickify(_as_str_series(o, "_home_nick", "home", "home_team"))
    away = _nickify(_as_str_series(o, "_away_nick", "away", "away_team"))
    s = s.where(~sh.eq(home), "HOME").where(~sh.eq(away), "AWAY")
    o["side_norm"] = s

    o["line"]  = pd.to_numeric(_as_str_series(o, "line","handicap","spread","total","points_total","ou_line"), errors="coerce")
    o["price"] = pd.to_numeric(_as_str_series(o, "odds","price","american","american_odds","us_odds"), errors="coerce")

    # ensure book exists
    if "book" not in o.columns:
        o["book"] = "API"

    o["_gk"] = (o["_date_iso_tmp"] + "|" + home + "|" + away).astype("string")

    o = (o.sort_values(["_date_iso_tmp"])
           .dropna(subset=["_gk"])
           .drop_duplicates(subset=["_gk","market_norm","side_norm","line"], keep="last"))
    return o[["_gk","market_norm","side_norm","line","price","book"]]


# ---------------- Join Odds + Scores ----------------
def attach_scores(edges: pd.DataFrame, scores: pd.DataFrame, year_min: int | None = None) -> tuple[pd.DataFrame, dict]:
    e = _ensure_nicks(_ensure_date_iso(edges.copy()))
    sc = _ensure_nicks(_ensure_date_iso(scores.copy()))

    if year_min is not None:
        e_season = _season_from_date_str(_as_str_series(e, "_date_iso", "date", "game_date"))
        e = e.loc[e_season >= year_min].copy()

    e_date = _as_str_series(e, "_date_iso")
    e_home = _nickify(_as_str_series(e, "_home_nick","home","home_team"))
    e_away = _nickify(_as_str_series(e, "_away_nick","away","away_team"))
    e_key_fw = (e_date + "|" + e_home + "|" + e_away).astype("string")
    e_key_sw = (e_date + "|" + e_away + "|" + e_home).astype("string")

    sc_date = _as_str_series(sc, "_date_iso")
    sc_home = _nickify(_as_str_series(sc, "_home_nick","home","home_team"))
    sc_away = _nickify(_as_str_series(sc, "_away_nick","away","away_team"))
    sc_key_fw = (sc_date + "|" + sc_home + "|" + sc_away).astype("string")

    key_union = pd.Index(e_key_fw).unique().union(pd.Index(e_key_sw).unique())
    sc_small = sc.loc[sc_key_fw.isin(key_union)].copy()

    if sc_small.empty:
        out = e.copy()
        out["home_score"] = pd.NA; out["away_score"] = pd.NA
        out["season"] = pd.NA; out["week"] = pd.NA; out["_date_iso_sc"] = pd.NA
        stats = {"rows": int(len(out)), "with_scores": 0, "coverage_pct": 0.0}
        return out, stats

    sc_small["_sc_key"] = sc_key_fw.loc[sc_small.index].astype("string")
    sc_small["_date_iso_sc"] = sc_date.loc[sc_small.index].astype("string")

    sc_small = sc_small.assign(
        _status_raw=_as_str_series(sc_small, "status","Status","game_status","is_final","IsFinal"),
        _dt=pd.to_datetime(_as_str_series(sc_small, "_date_iso_sc"), errors="coerce"),
        home_score=pd.to_numeric(_as_str_series(sc_small,"home_score","HomeScore"), errors="coerce"),
        away_score=pd.to_numeric(_as_str_series(sc_small,"away_score","AwayScore"), errors="coerce"),
        season=pd.to_numeric(_as_str_series(sc_small,"season","Season"), errors="coerce"),
        week=pd.to_numeric(_as_str_series(sc_small,"week","Week"), errors="coerce"),
    )
    sr = sc_small["_status_raw"].astype("string").str.lower()
    sc_small["_is_final"] = sr.fillna("").str.contains("final|ft|finished|complete|closed|ended|end of game") | sr.isin(["final","true","1"])
    sc_small["_has_scores"] = sc_small["home_score"].notna() & sc_small["away_score"].notna()

    sc_small = (sc_small.sort_values(["_is_final","_has_scores","_dt"], ascending=[False,False,False])
                        .drop_duplicates(subset=["_sc_key"], keep="first"))

    map_home_fw   = sc_small.set_index("_sc_key")["home_score"].to_dict()
    map_away_fw   = sc_small.set_index("_sc_key")["away_score"].to_dict()
    map_season_fw = sc_small.set_index("_sc_key")["season"].to_dict()
    map_week_fw   = sc_small.set_index("_sc_key")["week"].to_dict()
    map_dt_fw     = sc_small.set_index("_sc_key")["_date_iso_sc"].to_dict()

    out = e.copy()
    out["_join_key_fw"] = e_key_fw; out["_join_key_sw"] = e_key_sw

    out["home_score"]   = out["_join_key_fw"].map(map_home_fw)
    out["away_score"]   = out["_join_key_fw"].map(map_away_fw)
    out["season"]       = out["_join_key_fw"].map(map_season_fw)
    out["week"]         = out["_join_key_fw"].map(map_week_fw)
    out["_date_iso_sc"] = out["_join_key_fw"].map(map_dt_fw)

    miss = out["home_score"].isna() | out["away_score"].isna()
    if bool(miss.any()):
        out.loc[miss, "home_score"]   = out.loc[miss, "_join_key_sw"].map(map_away_fw)
        out.loc[miss, "away_score"]   = out.loc[miss, "_join_key_sw"].map(map_home_fw)
        out.loc[miss, "season"]       = out.loc[miss, "_join_key_sw"].map(map_season_fw)
        out.loc[miss, "week"]         = out.loc[miss, "_join_key_sw"].map(map_week_fw)
        out.loc[miss, "_date_iso_sc"] = out.loc[miss, "_join_key_sw"].map(map_dt_fw)

    out["total_points"] = pd.to_numeric(out["home_score"], errors="coerce") + pd.to_numeric(out["away_score"], errors="coerce")

    if "season" not in out.columns or out["season"].isna().all():
        dt2 = pd.to_datetime(_as_str_series(out, "_date_iso","_date_iso_sc"), errors="coerce")
        yr2, mo2 = dt2.dt.year, dt2.dt.month
        out["season"] = (yr2 - (mo2 <= 2).astype(int)).astype("Int64")

    with_scores = out["home_score"].notna() & out["away_score"].notna()
    stats = {"rows": int(len(out)), "with_scores": int(with_scores.sum()), "coverage_pct": float(with_scores.mean() * 100 if len(out) else 0.0)}

    out.drop(columns=["_join_key_fw","_join_key_sw"], inplace=True)
    return out, stats

# ---------------- Nearest-date rescue (optional) ----------------
def fill_scores_by_nearest_date(joined: pd.DataFrame, scores: pd.DataFrame, max_days: int = 1) -> pd.DataFrame:
    """
    For rows still missing scores, try matching by team pair (unordered) and the nearest
    score date within ±max_days. Helps with timezone/overnight drift.
    """
    if joined is None or scores is None or len(joined) == 0 or len(scores) == 0:
        return joined

    j = _ensure_nicks(_ensure_date_iso(joined.copy()))
    sc = _ensure_nicks(_ensure_date_iso(scores.copy()))

    def pair_key(h, a):
        h1 = _nickify(h); a1 = _nickify(a)
        return np.where(h1 <= a1, h1 + "|" + a1, a1 + "|" + h1)

    j["_pair"]  = pair_key(_as_str_series(j, "_home_nick","home","home_team"),
                           _as_str_series(j, "_away_nick","away","away_team"))
    sc["_pair"] = pair_key(_as_str_series(sc, "_home_nick","home","home_team"),
                           _as_str_series(sc, "_away_nick","away","away_team"))

    j["_dt"]  = pd.to_datetime(_as_str_series(j, "_date_iso"), errors="coerce")
    sc["_dt"] = pd.to_datetime(_as_str_series(sc, "_date_iso"), errors="coerce")

    miss = j["home_score"].isna() | j["away_score"].isna()
    if not bool(miss.any()):
        return j.drop(columns=["_pair","_dt"], errors="ignore")

    sc_small = sc[["_pair","_dt","home_score","away_score","season","week"]].copy()

    cand = j.loc[miss, ["_pair","_dt"]].merge(sc_small, on="_pair", how="left")
    cand["abs_diff"] = (cand["_dt_x"] - cand["_dt_y"]).abs()
    cand = cand[cand["abs_diff"] <= pd.Timedelta(days=max_days)]
    if cand.empty:
        return j.drop(columns=["_pair","_dt"], errors="ignore")

    cand = cand.sort_values(["_dt_x","abs_diff"]).drop_duplicates(subset=["_pair","_dt_x"], keep="first")
    cand.index = pd.MultiIndex.from_frame(cand[["_pair","_dt_x"]])

    idx = pd.MultiIndex.from_frame(j.loc[miss, ["_pair","_dt"]])
    j.loc[miss, "home_score"] = cand.reindex(idx)["home_score"].values
    j.loc[miss, "away_score"] = cand.reindex(idx)["away_score"].values
    j.loc[miss, "season"]     = cand.reindex(idx)["season"].values
    j.loc[miss, "week"]       = cand.reindex(idx)["week"].values

    return j.drop(columns=["_pair","_dt"], errors="ignore")

# ---------------- Load ALL odds (API + historical) ----------------
def _load_all_odds() -> pd.DataFrame | None:
    """
    Combine exports/odds_lines_all.csv (API) with exports/historical_odds_oddsshark.csv (if present),
    normalize, and return one long odds table keyed by _gk.
    """
    api = _load_odds_long()
    exp = _exports_dir()
    hist_path = exp / "historical_odds_oddsshark.csv"
    hist = None

    if hist_path.exists():
        h = pd.read_csv(hist_path)
        h = _ensure_nicks(_ensure_date_iso(h))
        m = _as_str_series(h, "market_norm","market").str.upper()
        m = (m.replace({"SPREADS":"SPREAD","TOTALS":"TOTAL"})
               .mask(m.str.contains("MONEY|ML|H2H"), "H2H")
               .mask(m.str.contains("SPREAD|HANDICAP|ATS"), "SPREAD")
               .mask(m.str.contains("TOTAL|OU|O/U|O-U"), "TOTAL"))
        h["market_norm"] = m

        s = _as_str_series(h, "side_norm","side","selection").str.upper()
        home = _nickify(_as_str_series(h, "_home_nick","home","home_team"))
        away = _nickify(_as_str_series(h, "_away_nick","away","away_team"))
        s_team = _nickify(s)
        s = s.where(~s_team.eq(home), "HOME").where(~s_team.eq(away), "AWAY")
        s = s.mask(s.str.contains(r"\bOVER\b"), "OVER").mask(s.str.contains(r"\bUNDER\b"), "UNDER")
        h["side_norm"] = s

        h["line"]  = pd.to_numeric(_as_str_series(h, "line","handicap","spread","total","points_total","ou_line"), errors="coerce")
        h["price"] = pd.to_numeric(_as_str_series(h, "odds","price","american","american_odds","us_odds"), errors="coerce")
        d = _as_str_series(h, "_date_iso")
        h["_gk"] = (d + "|" + home + "|" + away).astype("string")

        # ensure book exists
        if "book" not in h.columns:
            h["book"] = "HIST"

        hist = h[["_gk","market_norm","side_norm","line","price","book"]]

    if api is None and hist is None:
        return None
    if api is None:
        return hist
    if hist is None:
        return api

    out = pd.concat([api, hist], ignore_index=True)
    out = out.sort_values(["_gk"]).drop_duplicates(["_gk","market_norm","side_norm","line"], keep="last")
    return out

# ---------------- Backfill missing lines & odds ----------------
def _backfill_lines_and_odds(bets: pd.DataFrame, all_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing line/odds in SPREAD/TOTAL/H2H bets using any available odds rows.
    Order:
      1) exact (_gk, market, side, line)
      2) (_gk, market, side) regardless of line
      3) SPREAD: infer from opposite side sign
      4) TOTAL: median line per (_gk, market)
      5) SPREAD: median line per (_gk, market) with HOME=+med, AWAY=-med
    Also fills 'odds' from 'price' when available; ensures 'book' column exists.
    """
    x = bets.copy()
    if all_odds is None or all_odds.empty or x.empty:
        return x

    # Normalize numeric types early
    x["line"] = pd.to_numeric(x.get("line"), errors="coerce")
    all_odds = all_odds.copy()
    all_odds["line"] = pd.to_numeric(all_odds.get("line"), errors="coerce")
    all_odds["price"] = pd.to_numeric(all_odds.get("price"), errors="coerce")
    if "book" not in x.columns:
        x["book"] = pd.NA

    # --- STEP 1: exact match
    exact = x.merge(
        all_odds, how="left",
        left_on=["_gk","market_norm","side_norm","line"],
        right_on=["_gk","market_norm","side_norm","line"],
        suffixes=("", "__od1")
    )
    if "book" not in exact.columns:
        exact["book"] = pd.NA  # safety in case left lacked it

    exact["odds"] = pd.to_numeric(exact.get("odds"), errors="coerce")
    exact["odds"] = exact["odds"].where(exact["odds"].notna(), exact.get("price"))

    # Clean helper cols from STEP 1
    drop1 = [c for c in exact.columns if c.endswith("__od1")] + ["price"]
    exact = exact.drop(columns=drop1, errors="ignore")

    # Compute market and missing-line mask (current state)
    mkt = _as_str_series(exact, "market_norm", "market").str.upper().replace({"SPREADS":"SPREAD","TOTALS":"TOTAL"})
    miss_line = (mkt.isin(["SPREAD","TOTAL"]) & exact["line"].isna())

    if bool(miss_line.any()):
        # --- STEP 2: fill by (_gk, market, side) ignoring line
        by_side = (
            all_odds.dropna(subset=["line"])
                    .drop_duplicates(["_gk","market_norm","side_norm"], keep="last")
                    .rename(columns={"line":"line__fill1","price":"price__fill1","book":"book__fill1"})
        )
        exact = exact.merge(by_side, how="left", on=["_gk","market_norm","side_norm"])
        exact.loc[miss_line, "line"] = exact.loc[miss_line, "line"].where(
            exact.loc[miss_line, "line"].notna(), exact.loc[miss_line, "line__fill1"]
        )
        exact["odds"] = exact["odds"].where(exact["odds"].notna(), exact.get("price__fill1"))
        exact["book"] = exact["book"].where(exact["book"].notna(), exact.get("book__fill1"))

        # --- STEP 3: SPREAD infer from opposite side if still missing
        still = (mkt.eq("SPREAD") & exact["line"].isna())
        if bool(still.any()):
            opp = all_odds.dropna(subset=["line"]).copy()
            opp["side_norm_opp"] = opp["side_norm"].map({"HOME":"AWAY","AWAY":"HOME"}).fillna(opp["side_norm"])
            opp = opp.rename(columns={"line":"line_opp","price":"price_opp"})
            opp = opp[["_gk","market_norm","side_norm_opp","line_opp","price_opp"]]

            exact = exact.merge(
                opp, how="left",
                left_on=["_gk","market_norm","side_norm"],
                right_on=["_gk","market_norm","side_norm_opp"]
            )
            # Use boolean masks (aligned) — no index mixing
            mask_still = still
            exact.loc[mask_still, "line"] = exact.loc[mask_still, "line"].where(
                exact.loc[mask_still, "line"].notna(), -exact.loc[mask_still, "line_opp"]
            )
            exact["odds"] = exact["odds"].where(exact["odds"].notna(), exact.get("price_opp"))

        # --- STEP 4: TOTAL median per game
        still = (mkt.eq("TOTAL") & exact["line"].isna())
        if bool(still.any()):
            tot = all_odds[all_odds["market_norm"].str.upper().eq("TOTAL")]
            tot_med = tot.groupby(["_gk","market_norm"])["line"].median().rename("line_med").reset_index()
            exact = exact.merge(tot_med, how="left", on=["_gk","market_norm"])
            exact.loc[still, "line"] = exact.loc[still, "line"].where(
                exact.loc[still, "line"].notna(), exact.loc[still, "line_med"]
            )

        # --- STEP 5: SPREAD median per game (HOME=+med, AWAY=-med)
        still = (mkt.eq("SPREAD") & exact["line"].isna())
        if bool(still.any()):
            sp = all_odds[all_odds["market_norm"].str.upper().eq("SPREAD")]
            sp_med = sp.groupby(["_gk","market_norm"])["line"].median().rename("line_med").reset_index()
            exact = exact.merge(sp_med, how="left", on=["_gk","market_norm"])
            is_home = exact["side_norm"].eq("HOME")
            mask_home = still & is_home
            mask_away = still & ~is_home
            exact.loc[mask_home, "line"] = exact.loc[mask_home, "line"].where(
                exact.loc[mask_home, "line"].notna(), exact.loc[mask_home, "line_med"]
            )
            exact.loc[mask_away, "line"] = exact.loc[mask_away, "line"].where(
                exact.loc[mask_away, "line"].notna(), -exact.loc[mask_away, "line_med"]
            )

    # Final odds cleanup
    exact["odds"] = pd.to_numeric(exact.get("odds"), errors="coerce")

    # Drop helper columns left around
    drop_cols = [c for c in exact.columns if c.endswith("_fill1") or c.endswith("_opp") or c.endswith("_med") or c == "side_norm_opp"]
    exact = exact.drop(columns=drop_cols, errors="ignore")

    return exact

# ---------------- UI ----------------
st.title("Backtest — Scores Browser")
st.caption(f"Build: {BUILD_TAG}")

# 1) Load base tables
edges, scores = load_edges_scores()

# 2) Join scores (limit to LIVE_START_YEAR+)
joined, stats_all = attach_scores(edges, scores, year_min=LIVE_START_YEAR)

# 3) Nearest-date rescue (±3 days)
joined = fill_scores_by_nearest_date(joined, scores, max_days=3)

# 4) Normalize markets/sides and create season_ui
joined = _fix_side_labels(_normalize_market_side(joined))
season_from_date = _season_from_date_str(_as_str_series(joined, "_date_iso", "_date_iso_sc"))
joined["season_ui"] = (
    pd.to_numeric(joined.get("season"), errors="coerce")
      .fillna(season_from_date)
      .astype("Int64")
)

# 5) Game key for odds merge
_dt = _as_str_series(joined, "_date_iso")
_hm = _nickify(_as_str_series(joined, "_home_nick", "home", "home_team"))
_aw = _nickify(_as_str_series(joined, "_away_nick", "away", "away_team"))
joined["_gk"] = (_dt + "|" + _hm + "|" + _aw).astype("string")

# 6) Load ALL odds (API + historical) and backfill missing line/odds
_all_odds = _load_all_odds()
if _all_odds is not None and not _all_odds.empty:
    joined = _backfill_lines_and_odds(joined, _all_odds)

# 7) Coverage message
st.info(
    f"Join coverage (2020+): {stats_all['with_scores']:,}/{stats_all['rows']:,} "
    f"({stats_all['coverage_pct']:.1f}%) matched with scores."
)
st.success(f"Loaded edges={len(edges):,} • scores file rows={len(scores):,}")

archive = joined.copy()

# ---- Filter Season/Week (defines `live`) ----
st.markdown("### Filter: Season / Week")
with st.expander("Filter: Season / Week", expanded=True):
    seasons_all = (
        pd.to_numeric(joined.get("season_ui"), errors="coerce")
        .dropna().astype(int).unique().tolist()
    )
    seasons_all = sorted(seasons_all)
    include_archive = st.checkbox(f"Include archive (≤ {LIVE_START_YEAR-1})", value=False)

    fixed_2020_2026 = list(range(2020, 2027))
    season_choices = fixed_2020_2026[:] if not include_archive else sorted(set(fixed_2020_2026) | set(seasons_all))
    default_seasons = [s for s in fixed_2020_2026 if s in seasons_all] or ([seasons_all[-1]] if seasons_all else [])

    if default_seasons:
        sel_seasons = st.multiselect("Seasons", options=season_choices, default=default_seasons)
        season_num = pd.to_numeric(joined["season_ui"], errors="coerce")
        work_season = joined[season_num.isin(sel_seasons)].copy() if sel_seasons else joined.copy()
    else:
        st.warning("No seasons detected — showing all joined rows.")
        work_season = joined.copy()

    weeks_all = (
        pd.to_numeric(work_season.get("week"), errors="coerce")
        .dropna().astype(int).sort_values().unique().tolist()
    )
    if weeks_all:
        sel_weeks = st.multiselect("Weeks", options=weeks_all, default=weeks_all)
        week_num = pd.to_numeric(work_season["week"], errors="coerce")
        live = work_season[week_num.isin(sel_weeks)].copy() if sel_weeks else work_season.copy()
    else:
        st.caption("No week numbers available in this selection.")
        live = work_season.copy()

st.info(f"Rows in memory (post-join): {len(joined):,} • Filtered view: {len(live):,}")

# ---------------- Settle & Analyze ----------------
st.header("Settle & Analyze")

live = _prep_for_settlement(live)
live = _fix_side_labels(live)  # ensure HOME/AWAY/OVER/UNDER
base = settle_rows(live.copy())

# --- Download helpers ---
settled_df = base[base["_settled"]].copy()
na_df      = base[~base["_settled"]].copy()
m = _as_str_series(base, "market_norm", "market").str.upper()
line = pd.to_numeric(base.get("line"), errors="coerce")
needs_line_df = base[((m.eq("SPREAD") | m.eq("TOTAL")) & line.isna())].copy()

colz = ["season_ui","_date_iso","_home_nick","_away_nick","market_norm","side_norm","line","odds","home_score","away_score","_result","book"]
def _safe_cols(df):
    keep = [c for c in colz if c in df.columns]
    return df[keep] if keep else df

st.caption("⬇️ Downloads")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "Settled rows (CSV)",
        data=_safe_cols(settled_df).to_csv(index=False).encode("utf-8"),
        file_name="settled_rows.csv", mime="text/csv"
    )
with c2:
    st.download_button(
        "Unsettled (NA) rows (CSV)",
        data=_safe_cols(na_df).to_csv(index=False).encode("utf-8"),
        file_name="na_rows.csv", mime="text/csv"
    )
with c3:
    st.download_button(
        "Needs line (SPREAD/TOTAL) (CSV)",
        data=_safe_cols(needs_line_df).to_csv(index=False).encode("utf-8"),
        file_name="needs_line_rows.csv", mime="text/csv"
    )

left, right = st.columns(2)
with left:
    st.metric("Rows joined", f"{len(base):,}")
    st.metric("With scores", f"{int(base['_settled'].sum()):,}")
with right:
    by_res = (
        base["_result"].value_counts(dropna=False).rename_axis("result").reset_index(name="n")
        if not base.empty
        else pd.DataFrame({"result": [], "n": []})
    )
    st.dataframe(by_res, use_container_width=True)

with st.expander("🩺 Why are some results NA? (diagnostics)"):
    na = base[~base["_settled"]].copy()
    st.write(f"Unsettled rows: {len(na):,}")
    if na.empty:
        st.success("No NA rows — everything is settled.")
    else:
        m = _as_str_series(na, "market_norm", "market").str.upper().replace({"SPREADS":"SPREAD","TOTALS":"TOTAL"})
        s = _as_str_series(na, "side_norm", "side").str.upper()
        line = pd.to_numeric(na.get("line"), errors="coerce")
        hs = pd.to_numeric(na.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(na.get("away_score"), errors="coerce")
        reasons = pd.DataFrame({
            "no_scores": (hs.isna() | as_.isna()),
            "side_unknown": ~s.isin(["HOME","AWAY","OVER","UNDER"]),
            "h2h_no_side": (m.eq("H2H") & ~s.isin(["HOME","AWAY"])),
            "spread_no_line": (m.eq("SPREAD") & line.isna()),
            "total_no_line": (m.eq("TOTAL") & line.isna()),
        })
        counts = reasons.sum().rename_axis("reason").reset_index(name="rows")
        st.dataframe(counts, use_container_width=True)

only_settled = st.toggle("Show only settled", value=True)
view = base[base["_settled"]] if only_settled else base

# ---- P&L / ROI ----
stake = st.number_input("Stake per bet ($)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
work = view.copy()
odds_col = _first_present(work, ODDS_CANDIDATES) or "odds"
work["_american"] = pd.to_numeric(work.get(odds_col), errors="coerce")
work["_stake"] = float(stake)

work["_pnl"] = np.nan
mask = work["_result"].isin(["WIN", "LOSS", "PUSH"])
amer = work.loc[mask, "_american"].fillna(0.0).astype(float)
res  = work.loc[mask, "_result"]

win_payout = np.where(amer > 0, stake * (amer / 100.0), stake * (100.0 / np.maximum(1.0, np.abs(amer))))
work.loc[mask, "_pnl"] = np.where(res == "PUSH", 0.0, np.where(res == "WIN", win_payout, -stake))

bets = int(mask.sum())
wins = int((work["_result"] == "WIN").sum())
push = int((work["_result"] == "PUSH").sum())
net  = float(np.nansum(work["_pnl"]))
roi  = net / max(1.0, (bets * stake))

cA, cB, cC, cD = st.columns(4)
cA.metric("Bets", f"{bets:,}")
cB.metric("Win %", f"{(wins / max(1, bets - push) * 100):.1f}%")
cC.metric("Net P&L", f"${net:,.2f}")
cD.metric("ROI / bet", f"{roi*100:.2f}%")

# ---- Grouped rollups ----
group_by = st.multiselect("Group by…", ["season_ui", "week", "market_norm", "book"], default=["season_ui", "market_norm"])
group_cols = [c for c in group_by if c in work.columns or (c == "season_ui" and "season_ui" in base.columns)]
display_df = work.copy()
if "season_ui" in base.columns and "season_ui" not in display_df.columns:
    display_df = display_df.merge(
        base[["_date_iso","_home_nick","_away_nick","season_ui"]].drop_duplicates(),
        how="left", on=["_date_iso","_home_nick","_away_nick"]
    )

if group_by and not group_cols:
    st.warning("Selected group-by columns are not present in the data.")
else:
    g = (
        display_df.assign(
            _bet=display_df["_result"].isin(["WIN", "LOSS", "PUSH"]).astype(int),
            _win=(display_df["_result"] == "WIN").astype(int),
            _push=(display_df["_result"] == "PUSH").astype(int),
            _american=pd.to_numeric(display_df.get(odds_col), errors="coerce"),
        )
        .groupby(group_cols, dropna=False)
        .agg(bets=("_bet", "sum"),
             wins=("_win", "sum"),
             pushes=("_push", "sum"),
             pnl=("_pnl", "sum"),
             avg_odds=("_american", "mean"))
        .reset_index()
    )
    g["win_pct"] = g["wins"] / np.maximum(1, g["bets"] - g["pushes"]) * 100.0
    g["roi_%"]   = g["pnl"] / np.maximum(1.0, g["bets"] * stake) * 100.0
    st.dataframe(g, use_container_width=True)

# ---- Results table ----
view_u = _ensure_unique_columns(view)
pref = [
    c for c in (group_cols + ["season_ui","_date_iso","_home_nick","_away_nick","market_norm","side_norm","line",
                              "home_score","away_score","total_points","_result","odds","book"])
    if c in view_u.columns
]
pref = list(dict.fromkeys(pref))
cols = pref + [c for c in view_u.columns if c not in pref]

st.subheader("Results")
st.dataframe(view_u[cols], use_container_width=True)

# ---- Quick downloads for filtered view and archive ----
needs_line_mask = view["market_norm"].isin(["SPREAD","TOTAL"]) & view["line"].isna()
na_mask = ~view["_settled"]
st.caption(f"🔍 Needs line: {int(needs_line_mask.sum()):,} • NA rows: {int(na_mask.sum()):,}")

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("⬇️ Download ALL (filtered view)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="backtest_filtered_all.csv", mime="text/csv")
with c2:
    st.download_button("⬇️ Download NEEDS_LINE only",
        data=view.loc[needs_line_mask].to_csv(index=False).encode("utf-8"),
        file_name="needs_line_rows.csv", mime="text/csv")
with c3:
    st.download_button("⬇️ Download NA (unsettled) only",
        data=view.loc[na_mask].to_csv(index=False).encode("utf-8"),
        file_name="na_rows.csv", mime="text/csv")

st.download_button(
    "⬇️ Download filtered results (CSV)",
    data=view_u[cols].to_csv(index=False).encode("utf-8"),
    file_name="backtest_filtered.csv",
    mime="text/csv",
)
st.download_button(
    "⬇️ Download full joined archive (CSV)",
    data=archive.to_csv(index=False).encode("utf-8"),
    file_name="backtest_archive_joined.csv",
    mime="text/csv",
)

# ---- Unmatched / Debug ----
with st.expander("🔎 Unmatched / No scores (debug)"):
    miss = joined[joined["home_score"].isna() | joined["away_score"].isna()].copy()
    st.write(f"{len(miss):,} rows without scores after nearest-date rescue")
    if not miss.empty:
        miss["_key_fw"]  = (_as_str_series(miss, "_date_iso") + "|" +
                            _nickify(_as_str_series(miss, "_home_nick","home","home_team")) + "|" +
                            _nickify(_as_str_series(miss, "_away_nick","away","away_team"))).astype("string")
        cols_dbg = [c for c in ["_date_iso","_date_iso_sc","_home_nick","_away_nick","_key_fw",
                                "market_norm","side_norm","line","book","season","week"] if c in miss.columns]
        st.dataframe(_ensure_unique_columns(miss[cols_dbg]).head(200), use_container_width=True)
        st.download_button(
            "⬇️ Download ALL unmatched rows (CSV)",
            data=miss.to_csv(index=False).encode("utf-8"),
            file_name="unmatched_after_rescue.csv",
            mime="text/csv",
        )

# ---- Archive (peek) ----
with st.expander("📜 Full Archive (joined rows preview)"):
    st.dataframe(archive.head(100), use_container_width=True)
