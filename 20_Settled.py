from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Recovered environment shims
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve()
for base in [HERE.parent, *HERE.parents]:
    serving = base / "serving_ui"
    if serving.exists() and str(serving) not in sys.path:
        sys.path.insert(0, str(serving))
    if (base / "app").exists() and str(base) not in sys.path:
        sys.path.insert(0, str(base))

try:
    from app.lib.compliance_gate import require_eligibility
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
    from app.utils.nudge import begin_session, touch_session, session_duration_str, bump_usage, show_nudge
except Exception:
    def begin_session(): return None
    def touch_session(): return None
    def session_duration_str(): return ""
    def bump_usage(*args, **kwargs): return None
    def show_nudge(*args, **kwargs): return None

import importlib.util

def _load_clv_helper():
    try:
        from app.utils.clv import apply_clv_columns, american_to_decimal
        return apply_clv_columns, american_to_decimal
    except Exception:
        pass

    try:
        from utils.clv import apply_clv_columns, american_to_decimal
        return apply_clv_columns, american_to_decimal
    except Exception:
        pass

    app_dir = HERE.parents[1]   # .../app
    clv_path = app_dir / "utils" / "clv.py"
    if not clv_path.exists():
        raise ModuleNotFoundError(f"CLV helper not found at {clv_path}")

    spec = importlib.util.spec_from_file_location("edgefinder_clv", clv_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load CLV helper from {clv_path}")

    clv_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = clv_mod
    spec.loader.exec_module(clv_mod)

    return clv_mod.apply_clv_columns, clv_mod.american_to_decimal


apply_clv_columns, american_to_decimal = _load_clv_helper()


st.set_page_config(page_title="20 • Settled", page_icon="📈", layout="wide")
require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})

auth = login(required=False)
if not getattr(auth, "ok", True):
    st.stop()
if not getattr(auth, "authenticated", True):
    st.info("You are in read-only mode.")
show_logout()

begin_session()
touch_session()
bump_usage("page_visit")
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")
if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")
mount_in_sidebar("20_Settled")


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def repo_root() -> Path:
    env = os.environ.get("EDGE_FINDER_ROOT") or os.environ.get("EDGE_EXPORTS_ROOT")
    if env:
        return Path(env)
    for p in [HERE.parent, *HERE.parents]:
        if (p / "exports").exists() or (p / "data").exists():
            return p
    return Path.cwd()


def candidate_files() -> list[Path]:
    root = repo_root()
    folders = [root / "exports", root / "data", HERE.parent, Path.cwd() / "exports"]
    names = [
        "settled.csv",
        "settled.parquet",
        "history.csv",
        "history.parquet",
        "your_history.csv",
        "your_history.parquet",
        "master_likes.parquet",
        "master_likes.csv",
    ]
    paths: list[Path] = []
    for folder in folders:
        for name in names:
            p = folder / name
            if p.exists():
                paths.append(p)
    return sorted(set(paths), key=lambda p: (p.suffix, p.name))


def read_any(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_existing(df: pd.DataFrame, names: Iterable[str], default=np.nan):
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def normalize_result(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"win", "won", "w", "1", "true", "push_win"}:
        return "win"
    if text in {"loss", "lost", "l", "0", "false"}:
        return "loss"
    if text in {"push", "void", "tie"}:
        return "push"
    return ""


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c) for c in out.columns]

    rename_map = {
        "date": "timestamp",
        "placed_at": "timestamp",
        "created_at": "timestamp",
        "bet_line": "line_at_pick",
        "line": "line_at_pick",
        "odds": "payout_odds",
        "book_name": "book",
        "sportsbook": "book",
        "pick_side": "side",
        "bet_side": "side",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    out["timestamp"] = pd.to_datetime(_first_existing(out, ["timestamp", "commence_time", "game_date"]), errors="coerce")
    out["market"] = _first_existing(out, ["market", "market_type"], "").astype("string")
    out["side"] = _first_existing(out, ["side", "selection"], "").astype("string")
    out["book"] = _first_existing(out, ["book"], "").astype("string")
    out["home"] = _first_existing(out, ["home", "home_team"], "").astype("string")
    out["away"] = _first_existing(out, ["away", "away_team"], "").astype("string")
    out["line_at_pick"] = pd.to_numeric(_first_existing(out, ["line_at_pick"]), errors="coerce")
    out["closing_line"] = pd.to_numeric(_first_existing(out, ["closing_line"]), errors="coerce")
    out["payout_odds"] = pd.to_numeric(_first_existing(out, ["payout_odds", "american_odds"]), errors="coerce")
    out["closing_odds"] = pd.to_numeric(_first_existing(out, ["closing_odds"]), errors="coerce")
    out["model_prob"] = pd.to_numeric(_first_existing(out, ["model_prob", "win_prob", "probability"]), errors="coerce")
    out["edge"] = pd.to_numeric(_first_existing(out, ["edge", "expected_edge", "realized_edge"]), errors="coerce")

    if "result" in out.columns:
        out["result"] = out["result"].map(normalize_result)
    else:
        out["result"] = _first_existing(out, ["outcome"], "").map(normalize_result)

    if "hit_bool" not in out.columns:
        out["hit_bool"] = np.where(out["result"].eq("win"), 1.0, np.where(out["result"].eq("loss"), 0.0, np.nan))
    else:
        out["hit_bool"] = pd.to_numeric(out["hit_bool"], errors="coerce")

    out["decimal_odds"] = pd.to_numeric(_first_existing(out, ["decimal_odds"]), errors="coerce")
    missing_dec = out["decimal_odds"].isna()
    if missing_dec.any():
        out.loc[missing_dec, "decimal_odds"] = out.loc[missing_dec, "payout_odds"].apply(american_to_decimal)

    out = apply_clv_columns(out)
    return out


@st.cache_data(show_spinner=False)
def load_settled() -> pd.DataFrame:
    frames = [prepare_history(read_any(path)) for path in candidate_files()]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True, sort=False)
    settled = data[data["hit_bool"].notna() | data["result"].isin(["win", "loss", "push"])].copy()
    if settled.empty:
        return pd.DataFrame(columns=data.columns)
    settled = settled.sort_values("timestamp", ascending=False, na_position="last")
    settled = settled.drop_duplicates(subset=[c for c in ["timestamp", "home", "away", "market", "side", "book"] if c in settled.columns])
    return settled


def roi_flat_stake(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    settled = df[df["hit_bool"].notna()].copy()
    if settled.empty:
        return float("nan")
    dec = pd.to_numeric(settled["decimal_odds"], errors="coerce").fillna(1.909)
    wins = pd.to_numeric(settled["hit_bool"], errors="coerce").fillna(0.0)
    profit = np.where(wins == 1.0, dec - 1.0, np.where(wins == 0.0, -1.0, 0.0))
    return float(np.nanmean(profit))


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.title("📉 Settled")
st.caption("Clean rebuild with shared CLV helper, missing-column tolerance, and empty-data safety.")

df = load_settled()

with st.sidebar:
    st.subheader("Filters")
    books = ["All"] + sorted([x for x in df.get("book", pd.Series(dtype="string")).dropna().astype(str).unique() if x])
    markets = ["All"] + sorted([x for x in df.get("market", pd.Series(dtype="string")).dropna().astype(str).unique() if x])
    book_pick = st.selectbox("Book", books, index=0)
    market_pick = st.selectbox("Market", markets, index=0)
    result_pick = st.selectbox("Result", ["All", "win", "loss", "push"], index=0)

if df.empty:
    st.warning("No settled dataset found yet. Expected files like exports/settled.csv, exports/history.csv, or data/master_likes.parquet.")
    st.dataframe(pd.DataFrame(columns=["timestamp", "home", "away", "market", "side", "result", "clv"]))
    st.stop()

work = df.copy()
if book_pick != "All":
    work = work[work["book"].astype(str) == book_pick]
if market_pick != "All":
    work = work[work["market"].astype(str) == market_pick]
if result_pick != "All":
    work = work[work["result"].astype(str) == result_pick]

settled_count = len(work)
win_rate = float(work["hit_bool"].dropna().mean()) if work["hit_bool"].notna().any() else float("nan")
roi = roi_flat_stake(work)
avg_clv = float(work["clv"].dropna().mean()) if work["clv"].notna().any() else float("nan")
beat_close = float(work["beat_closing"].dropna().mean()) if work["beat_closing"].notna().any() else float("nan")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Settled picks", f"{settled_count:,}")
c2.metric("Win rate", "—" if np.isnan(win_rate) else f"{win_rate:.1%}")
c3.metric("ROI / pick", "—" if np.isnan(roi) else f"{roi:.1%}")
c4.metric("Avg CLV", "—" if np.isnan(avg_clv) else f"{avg_clv:.3f}")
c5.metric("Beat close", "—" if np.isnan(beat_close) else f"{beat_close:.1%}")

st.subheader("Results by market")
if work.empty:
    st.info("No rows match the selected filters.")
else:
    by_market = (
        work.groupby("market", dropna=False)
        .agg(picks=("market", "size"), win_rate=("hit_bool", "mean"), avg_clv=("clv", "mean"))
        .reset_index()
        .sort_values("picks", ascending=False)
    )
    st.dataframe(by_market, use_container_width=True)

st.subheader("Settled history")
show_cols = [
    c for c in [
        "timestamp", "home", "away", "market", "side", "book", "line_at_pick", "closing_line",
        "payout_odds", "closing_odds", "model_prob", "edge", "result", "hit_bool", "clv", "beat_closing"
    ] if c in work.columns
]
st.dataframe(work[show_cols], use_container_width=True, hide_index=True)
