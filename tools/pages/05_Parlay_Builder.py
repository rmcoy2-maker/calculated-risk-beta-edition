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

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    from lib.access import require_allowed_page, beta_banner
except Exception:
    def require_allowed_page(*args, **kwargs):
        return None
    def beta_banner(*args, **kwargs):
        return None

try:
    from lib.compliance_gate import require_eligibility
except Exception:
    def require_eligibility(*args, **kwargs):
        return True

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
    from app.utils.nudge import (
        begin_session,
        touch_session,
        session_duration_str,
        bump_usage,
        show_nudge,
    )
except Exception:
    def begin_session(): pass
    def touch_session(): pass
    def session_duration_str() -> str: return ""
    def bump_usage(*args, **kwargs): return None
    def show_nudge(*args, **kwargs): return None

try:
    from app.utils.parlay_cart import read_cart, clear_cart
except Exception:
    def read_cart():
        return pd.DataFrame()
    def clear_cart():
        return None

try:
    from app.utils.parlay_ui import selectable_odds_table
except Exception:
    def selectable_odds_table(*args, **kwargs):
        return None


st.set_page_config(page_title="Parlay Builder", page_icon="🧱", layout="wide")
require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})
require_allowed_page("pages/05_Parlay_Builder.py")
beta_banner()

begin_session()
touch_session()
if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")
bump_usage("page_visit")
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")

PAGE_PROTECTED = False
auth = login(required=PAGE_PROTECTED)
if not getattr(auth, "ok", True):
    st.stop()
show_logout()

auth = login(required=False)
if not getattr(auth, "authenticated", True):
    st.info("You are in read-only mode.")
show_logout()

diag = mount_in_sidebar("05_Parlay_Builder")

TZ = "America/New_York"


def _exports_dir() -> Path:
    env = os.environ.get("EDGE_EXPORTS_DIR", "").strip()
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p

    here = Path(__file__).resolve()
    for up in [here.parent] + list(here.parents):
        if up.name.lower() == "edge-finder":
            p = up / "exports"
            p.mkdir(parents=True, exist_ok=True)
            return p

    p = Path.cwd() / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _latest_csv(paths: list[Path]) -> Optional[Path]:
    paths = [p for p in paths if p and p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _best_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


_ALIAS = {
    "REDSKINS": "COMMANDERS",
    "WASHINGTON": "COMMANDERS",
    "FOOTBALL": "COMMANDERS",
    "OAKLAND": "RAIDERS",
    "LV": "RAIDERS",
    "LAS": "RAIDERS",
    "VEGAS": "RAIDERS",
    "SD": "CHARGERS",
    "STL": "RAMS",
}


def _nickify(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.upper()
    s = s.str.replace(r"[^A-Z0-9 ]+", "", regex=True).str.strip().replace(_ALIAS)
    return s.str.replace(r"\s+", "_", regex=True)


def _ensure_nicks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    home_c = _best_col(df, ["_home_nick", "home_nick", "home", "home_team", "Home", "HOME", "team_home"])
    away_c = _best_col(df, ["_away_nick", "away_nick", "away", "away_team", "Away", "AWAY", "team_away"])

    out = df.copy()
    if home_c is None:
        out["_home_nick"] = pd.Series([""] * len(out), dtype="string")
    else:
        out["_home_nick"] = _nickify(out[home_c].astype("string"))

    if away_c is None:
        out["_away_nick"] = pd.Series([""] * len(out), dtype="string")
    else:
        out["_away_nick"] = _nickify(out[away_c].astype("string"))

    return out


def _norm_market(m) -> str:
    m = (str(m) or "").strip().lower()
    if m in {"h2h", "ml", "moneyline", "money line"}:
        return "H2H"
    if m.startswith("spread") or m in {"spread", "spreads"}:
        return "SPREADS"
    if m.startswith("total") or m in {"total", "totals"}:
        return "TOTALS"
    return m.upper()


def _ensure_date_iso(df: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        return out

    for c in candidates:
        if c in out.columns:
            s = pd.to_datetime(out[c], errors="coerce", utc=True)
            out["_date_iso"] = s.dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
            break

    if "_date_iso" not in out.columns:
        out["_date_iso"] = pd.Series(pd.NA, index=out.index, dtype="string")

    return out


def latest_batch(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for col in ["_snapshot_ts_utc", "snapshot_ts_utc", "snapshot_ts", "_ts", "ts"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=True)
            last = ts.max()
            return df[ts == last].copy()
    return df


def within_next_week(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "_date_iso" not in df.columns:
        return df
    d = pd.to_datetime(df["_date_iso"], errors="coerce", utc=True).dt.tz_convert(TZ)
    today = pd.Timestamp.now(tz=TZ).normalize()
    end = today + pd.Timedelta(days=7)
    return df[(d >= today) & (d <= end)].copy()


def refresh_button(key: Optional[str] = None):
    if st.button("🔄 Refresh data", key=key or f"refresh_{__name__}"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()


def _american_to_decimal(odds):
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(o > 0, 1 + o / 100.0, np.where(o < 0, 1 + 100.0 / np.abs(o), np.nan)),
        index=getattr(odds, "index", None),
    )


@st.cache_data(ttl=60)
def load_live() -> tuple[pd.DataFrame, Path]:
    exp = _exports_dir()
    cand = [
        exp / "lines_live.csv",
        exp / "lines_live_latest.csv",
        exp / "lines_live_normalized.csv",
        exp / "lines_live_fixed.csv",
    ]
    p = _latest_csv(cand) or cand[0]
    df = pd.read_csv(p, low_memory=False, encoding="utf-8-sig") if p.exists() else pd.DataFrame()
    return df, p


@st.cache_data(ttl=60)
def load_edges() -> tuple[pd.DataFrame, Path]:
    exp = _exports_dir()
    names = [
        "edges_standardized.csv",
        "edges_graded_full_normalized_std.csv",
        "edges_graded_full.csv",
        "edges_normalized.csv",
        "edges_master.csv",
    ]
    paths = [exp / n for n in names]
    p = _latest_csv(paths) or paths[0]
    df = pd.read_csv(p, low_memory=False, encoding="utf-8-sig") if p.exists() else pd.DataFrame()
    return df, p


st.title("Parlay Builder — Your / House / Computer")
refresh_button(key="refresh_05_parlay")

edges, edges_p = load_edges()
live, live_p = load_live()

st.caption(f"Edges: `{edges_p}` · Live: `{live_p}`")

live = within_next_week(
    latest_batch(
        _ensure_date_iso(live, ["_date_iso", "event_date", "commence_time", "date", "game_date", "Date"])
    )
)
edges = within_next_week(
    _ensure_date_iso(edges, ["_date_iso", "date", "game_date", "_key_date", "Date"])
)

edges = _ensure_nicks(edges)
live = _ensure_nicks(live)

for df in (edges, live):
    df["_market_norm"] = df.get("_market_norm", df.get("market", pd.Series(index=df.index))).map(_norm_market)
    if "_ev_per_$1" not in df.columns:
        o = pd.to_numeric(df.get("price", pd.Series(index=df.index)), errors="coerce")
        p = pd.to_numeric(df.get("p_win", pd.Series(index=df.index)), errors="coerce").clip(0, 1)
        dec = np.where(o > 0, 1 + o / 100.0, np.where(o < 0, 1 + 100.0 / np.abs(o), np.nan))
        df["_ev_per_$1"] = p * (dec - 1) - (1 - p)

st.sidebar.header("Parlay Options")
legs = st.sidebar.slider("Number of legs", min_value=3, max_value=10, value=5)
avoid_same_game = st.sidebar.checkbox("Avoid same game/market", value=True)
market_pick = st.sidebar.multiselect("Markets", options=["H2H", "SPREADS", "TOTALS"], default=[])


def _filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if market_pick:
        out = out[out["_market_norm"].isin(market_pick)]
    return out


def _suggest_computer(df: pd.DataFrame, max_legs: int = 5) -> pd.DataFrame:
    dd = df.copy()
    sort_cols = ["_ev_per_$1"] if "_ev_per_$1" in dd.columns else ["p_win"] if "p_win" in dd.columns else ["price"]
    dd = dd.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    km = [c for c in ["_date_iso", "_away_nick", "_home_nick", "_market_norm"] if c in dd.columns]
    if km:
        dd = dd.drop_duplicates(km)
    return dd.head(max_legs)


pool = _filter(edges if not edges.empty else live).copy()
pool = _ensure_nicks(pool)


def _pretty_name(s):
    s = s.astype("string").fillna("")
    return s.str.replace("_", " ", regex=False).str.title()


if "_home_nick" in pool.columns and "home" not in pool.columns:
    pool["home"] = _pretty_name(pool["_home_nick"])
if "_away_nick" in pool.columns and "away" not in pool.columns:
    pool["away"] = _pretty_name(pool["_away_nick"])

display_cols = [
    c for c in [
        "_date_iso",
        "home",
        "away",
        "_home_nick",
        "_away_nick",
        "_market_norm",
        "side",
        "line",
        "price",
        "p_win",
        "_ev_per_$1",
    ] if c in pool.columns
]

if "your_parlay" not in st.session_state:
    st.session_state["your_parlay"] = pd.DataFrame()

st.subheader("Your Parlay (pool preview)")
st.dataframe(pool[display_cols].head(1000), hide_index=True, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    i_start = st.number_input("Add rows start index", min_value=0, value=0, step=1)
with c2:
    i_stop = st.number_input("Add rows stop index (exclusive)", min_value=1, value=min(legs, max(1, len(pool))), step=1)

if st.button("➕ Add to Your Parlay"):
    take = pool.iloc[int(i_start):int(i_stop)].copy()
    if not take.empty:
        st.session_state["your_parlay"] = pd.concat([st.session_state["your_parlay"], take], ignore_index=True)

your = st.session_state["your_parlay"].head(legs).copy()
house = _filter(live).head(legs).copy()
computer = _suggest_computer(_filter(edges), max_legs=legs)

if avoid_same_game and {"_date_iso", "_home_nick", "_away_nick"} <= set(computer.columns):
    computer = computer.drop_duplicates(["_date_iso", "_home_nick", "_away_nick"])


def _card(title: str, df: pd.DataFrame):
    with st.container(border=True):
        st.markdown(f"### {title}")
        if df.empty:
            st.info("No legs.")
            return
        cols = [c for c in ["_date_iso", "_away_nick", "_home_nick", "_market_norm", "side", "line", "price", "p_win", "_ev_per_$1"] if c in df.columns]
        st.dataframe(df[cols].reset_index(drop=True), hide_index=True, use_container_width=True)
        if "price" in df.columns and len(df) > 0:
            dec = _american_to_decimal(df["price"]).fillna(1.0)
            parlay_dec = float(np.prod(dec))
            st.caption(f"Implied decimal payout: **{parlay_dec:.2f}x**")


cols = st.columns(3)
with cols[0]:
    _card("Your Parlay", your)
with cols[1]:
    _card("House", house)
with cols[2]:
    _card("Computer", computer)

with st.expander("🧺 Your Cart (staged picks)", expanded=True):
    _cart = read_cart()
    if not isinstance(_cart, pd.DataFrame):
        _cart = pd.DataFrame()
    st.caption(f"{len(_cart):,} item(s) in cart)")
    if not _cart.empty:
        st.dataframe(_cart, hide_index=True, use_container_width=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Use cart as Your Parlay", key="use_cart", type="primary"):
                st.session_state["your_parlay"] = _cart.copy()
                st.success("Loaded cart into Your Parlay.")
                st.rerun()
        with c2:
            if st.button("Clear cart", key="clear_cart"):
                clear_cart()
                st.success("Cart cleared.")
                st.rerun()

try:
    if isinstance(live, pd.DataFrame) and not live.empty:
        selectable_odds_table(
            live,
            page_key="parlay_live",
            page_name="05_Parlay_Builder",
            allow_same_game=False,
            one_per_market_per_game=True,
        )
except Exception:
    pass
