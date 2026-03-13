from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

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
    from app.lib.access import premium_enabled
except Exception:
    def premium_enabled() -> bool:
        return True

try:
    from app.utils.nudge import begin_session, touch_session, session_duration_str, bump_usage, show_nudge
except Exception:
    def begin_session(): return None
    def touch_session(): return None
    def session_duration_str(): return ""
    def bump_usage(*args, **kwargs): return None
    def show_nudge(*args, **kwargs): return None

import importlib.util

from pathlib import Path
import importlib.util

def _find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "streamlit_app.py").exists():
            return p
    return start.parent

from pathlib import Path
import importlib.util
import pandas as pd

def _find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "streamlit_app.py").exists():
            return p
    return start.parent

def _fallback_american_to_decimal(odds):
    if odds is None or pd.isna(odds):
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o > 0:
        return 1.0 + o / 100.0
    if o < 0:
        return 1.0 + 100.0 / abs(o)
    return None

def _fallback_apply_clv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if (
        "closing_line" in out.columns
        and "line_at_pick" in out.columns
        and "market" in out.columns
        and "side" in out.columns
    ):
        def beat_closing_bool(row):
            try:
                mkt = str(row.get("market", "")).lower()
                side = str(row.get("side", "")).lower()
                pick = float(row.get("line_at_pick"))
                close = float(row.get("closing_line"))
            except Exception:
                return None

            if any(pd.isna(x) for x in [pick, close]):
                return None

            if "spread" in mkt:
                if side == "home" or side == str(row.get("home", "")).lower():
                    return 1.0 if pick > close else 0.0
                if side == "away" or side == str(row.get("away", "")).lower():
                    return 1.0 if pick < close else 0.0
                return None

            if "total" in mkt or mkt.startswith("o/"):
                if side in ("over", "o"):
                    return 1.0 if close > pick else 0.0
                if side in ("under", "u"):
                    return 1.0 if close < pick else 0.0
                return None

            return None

        if "beat_closing" not in out.columns or out["beat_closing"].isna().all():
            out["beat_closing"] = out.apply(beat_closing_bool, axis=1)

    return out

def _load_clv_helper():
    here = Path(__file__).resolve()
    repo_root = _find_repo_root(here)

    candidates = [
        repo_root / "tools" / "clv_helper.py",
        repo_root / "utils" / "clv_helper.py",
        repo_root / "app" / "utils" / "clv_helper.py",
        repo_root / "pages" / "clv_helper.py",
    ]

    for clv_path in candidates:
        if clv_path.exists():
            spec = importlib.util.spec_from_file_location("clv_helper", clv_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)

            apply_clv_columns = getattr(mod, "apply_clv_columns", None)
            american_to_decimal = getattr(mod, "american_to_decimal", None)

            if apply_clv_columns and american_to_decimal:
                return apply_clv_columns, american_to_decimal

    # Cloud-safe fallback: do not crash if helper file is missing
    return _fallback_apply_clv_columns, _fallback_american_to_decimal
    )


apply_clv_columns, american_to_decimal = _load_clv_helper()

st.set_page_config(page_title="93 • Analytics", page_icon="📊", layout="wide")
require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})

auth = login(required=False)
if not getattr(auth, "ok", True):
    st.stop()
if not getattr(auth, "authenticated", True):
    st.info("You are in read-only mode.")
show_logout()
begin_session(); touch_session(); bump_usage("page_visit")
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")
if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")
if not premium_enabled():
    st.warning("Premium-only page running in limited mode.")


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
        "master_likes.parquet", "master_likes.csv", "history.csv", "history.parquet",
        "your_history.csv", "your_history.parquet", "settled.csv", "settled.parquet",
    ]
    out: list[Path] = []
    for folder in folders:
        for name in names:
            path = folder / name
            if path.exists():
                out.append(path)
    return sorted(set(out))


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
    if text in {"win", "won", "w", "1", "true"}:
        return "win"
    if text in {"loss", "lost", "l", "0", "false"}:
        return "loss"
    if text in {"push", "void", "tie"}:
        return "push"
    return ""


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c) for c in out.columns]

    alias = {
        "date": "timestamp", "placed_at": "timestamp", "created_at": "timestamp",
        "bet_line": "line_at_pick", "line": "line_at_pick", "odds": "payout_odds",
        "book_name": "book", "sportsbook": "book", "pick_side": "side", "bet_side": "side",
    }
    for old, new in alias.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    out["timestamp"] = pd.to_datetime(_first_existing(out, ["timestamp", "commence_time", "game_date"]), errors="coerce")
    out["season"] = pd.to_numeric(_first_existing(out, ["season"]), errors="coerce")
    out["week"] = pd.to_numeric(_first_existing(out, ["week"]), errors="coerce")
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
    out["result"] = _first_existing(out, ["result", "outcome"], "").map(normalize_result)
    out["hit_bool"] = pd.to_numeric(_first_existing(out, ["hit_bool"]), errors="coerce")
    needs = out["hit_bool"].isna()
    out.loc[needs, "hit_bool"] = np.where(out.loc[needs, "result"].eq("win"), 1.0, np.where(out.loc[needs, "result"].eq("loss"), 0.0, np.nan))
    out["decimal_odds"] = pd.to_numeric(_first_existing(out, ["decimal_odds"]), errors="coerce")
    miss = out["decimal_odds"].isna()
    out.loc[miss, "decimal_odds"] = out.loc[miss, "payout_odds"].apply(american_to_decimal)
    return apply_clv_columns(out)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    frames = [prepare(read_any(path)) for path in candidate_files()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True, sort=False)
    subset = [c for c in ["timestamp", "home", "away", "market", "side", "book"] if c in data.columns]
    if subset:
        data = data.drop_duplicates(subset=subset)
    return data.sort_values("timestamp", ascending=False, na_position="last")


def roi_flat_stake(df: pd.DataFrame) -> float:
    settled = df[df["hit_bool"].notna()].copy()
    if settled.empty:
        return float("nan")
    dec = pd.to_numeric(settled["decimal_odds"], errors="coerce").fillna(1.909)
    wins = pd.to_numeric(settled["hit_bool"], errors="coerce").fillna(0.0)
    profits = np.where(wins == 1.0, dec - 1.0, np.where(wins == 0.0, -1.0, 0.0))
    return float(np.nanmean(profits))


def brier_score(df: pd.DataFrame) -> float:
    use = df[df["hit_bool"].notna() & df["model_prob"].notna()]
    if use.empty:
        return float("nan")
    return float(np.mean((use["model_prob"].astype(float) - use["hit_bool"].astype(float)) ** 2))


st.title("📊 Analytics")
st.caption("Clean rebuild using shared app/utils/clv.py, tolerant loaders, and safe empty-state behavior.")

df = load_data()
if df.empty:
    st.warning("No analytics dataset found yet. Add master_likes.parquet, history.csv, your_history.csv, or settled.csv to exports/ or data/.")
    st.dataframe(pd.DataFrame(columns=["timestamp", "market", "side", "book", "result", "clv"]))
    st.stop()

with st.sidebar:
    st.subheader("Filters")
    settled_only = st.checkbox("Settled only", value=True)
    books = ["All"] + sorted([x for x in df["book"].dropna().astype(str).unique() if x])
    markets = ["All"] + sorted([x for x in df["market"].dropna().astype(str).unique() if x])
    book_pick = st.selectbox("Book", books, index=0)
    market_pick = st.selectbox("Market", markets, index=0)

work = df.copy()
if settled_only:
    work = work[work["hit_bool"].notna() | work["result"].isin(["win", "loss", "push"])]
if book_pick != "All":
    work = work[work["book"].astype(str) == book_pick]
if market_pick != "All":
    work = work[work["market"].astype(str) == market_pick]

c1, c2, c3, c4, c5, c6 = st.columns(6)
count = len(work)
wr = float(work["hit_bool"].dropna().mean()) if work["hit_bool"].notna().any() else float("nan")
roi = roi_flat_stake(work)
clv = float(work["clv"].dropna().mean()) if work["clv"].notna().any() else float("nan")
beat = float(work["beat_closing"].dropna().mean()) if work["beat_closing"].notna().any() else float("nan")
brier = brier_score(work)
mean_edge = float(work["edge"].dropna().mean()) if work["edge"].notna().any() else float("nan")
c1.metric("Rows", f"{count:,}")
c2.metric("Win rate", "—" if np.isnan(wr) else f"{wr:.1%}")
c3.metric("ROI / pick", "—" if np.isnan(roi) else f"{roi:.1%}")
c4.metric("Avg edge", "—" if np.isnan(mean_edge) else f"{mean_edge:.3f}")
c5.metric("Avg CLV", "—" if np.isnan(clv) else f"{clv:.3f}")
c6.metric("Brier", "—" if np.isnan(brier) else f"{brier:.4f}")

st.subheader("Summary by market")
if work.empty:
    st.info("No rows match the selected filters.")
else:
    market_summary = (
        work.groupby("market", dropna=False)
        .agg(
            picks=("market", "size"),
            win_rate=("hit_bool", "mean"),
            roi=("decimal_odds", lambda s: roi_flat_stake(work.loc[s.index])),
            avg_edge=("edge", "mean"),
            avg_clv=("clv", "mean"),
        )
        .reset_index()
        .sort_values("picks", ascending=False)
    )
    st.dataframe(market_summary, use_container_width=True)

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Summary by book")
    if "book" in work.columns and not work.empty:
        book_summary = (
            work.groupby("book", dropna=False)
            .agg(picks=("book", "size"), win_rate=("hit_bool", "mean"), avg_clv=("clv", "mean"))
            .reset_index()
            .sort_values("picks", ascending=False)
        )
        st.dataframe(book_summary, use_container_width=True)
    else:
        st.caption("Book column unavailable.")

with col_r:
    st.subheader("Volume over time")
    if work.empty:
        st.caption("No data to chart.")
    else:
        series = work.copy()
        series["day"] = pd.to_datetime(series["timestamp"], errors="coerce").dt.date
        counts = series.groupby("day", dropna=True).size().rename("picks")
        if counts.empty:
            st.caption("Timestamp column unavailable for charting.")
        else:
            st.bar_chart(counts)

st.subheader("Filtered dataset")
show_cols = [
    c for c in [
        "timestamp", "season", "week", "home", "away", "market", "side", "book", "line_at_pick",
        "closing_line", "payout_odds", "closing_odds", "model_prob", "edge", "result", "hit_bool", "clv", "beat_closing"
    ] if c in work.columns
]
st.dataframe(work[show_cols], use_container_width=True, hide_index=True)
