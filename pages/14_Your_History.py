from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve()
APP_DIR = HERE.parents[1]         # .../app
PROJECT_ROOT = HERE.parents[2]    # .../serving_ui_recovered

for p in (PROJECT_ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


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

    clv_path = APP_DIR / "utils" / "clv.py"
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


st.set_page_config(page_title="14 • Your History", page_icon="📜", layout="wide")
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
show_nudge(
    feature="analytics",
    metric="page_visit",
    threshold=10,
    period="1D",
    demo_unlock=True,
    location="inline",
)

if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")


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
    folders = [
        root / "exports",
        root / "data",
        HERE.parent,
        Path.cwd() / "exports",
        Path.cwd() / "data",
    ]
    names = [
        "history.csv",
        "history.parquet",
        "your_history.csv",
        "your_history.parquet",
        "settled.csv",
        "settled.parquet",
        "master_likes.parquet",
        "master_likes.csv",
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
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _first_existing(df: pd.DataFrame, names: Iterable[str], default=np.nan) -> pd.Series:
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
        "date": "timestamp",
        "placed_at": "timestamp",
        "created_at": "timestamp",
        "bet_line": "line_at_pick",
        "line": "line_at_pick",
        "pick_line": "line_at_pick",
        "odds": "payout_odds",
        "american_odds": "payout_odds",
        "sportsbook": "book",
        "book_name": "book",
        "pick_side": "side",
        "bet_side": "side",
        "selection": "side",
        "market_type": "market",
        "home_team": "home",
        "away_team": "away",
        "outcome": "result",
    }

    for old, new in alias.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    out["timestamp"] = pd.to_datetime(
        _first_existing(out, ["timestamp", "commence_time", "game_date"]),
        errors="coerce",
    )
    out["market"] = _first_existing(out, ["market"], "").astype("string")
    out["side"] = _first_existing(out, ["side"], "").astype("string")
    out["book"] = _first_existing(out, ["book"], "").astype("string")
    out["home"] = _first_existing(out, ["home"], "").astype("string")
    out["away"] = _first_existing(out, ["away"], "").astype("string")

    out["line_at_pick"] = pd.to_numeric(
        _first_existing(out, ["line_at_pick"]),
        errors="coerce",
    )
    out["closing_line"] = pd.to_numeric(
        _first_existing(out, ["closing_line", "close_line"]),
        errors="coerce",
    )
    out["payout_odds"] = pd.to_numeric(
        _first_existing(out, ["payout_odds"]),
        errors="coerce",
    )
    out["closing_odds"] = pd.to_numeric(
        _first_existing(out, ["closing_odds", "close_odds"]),
        errors="coerce",
    )
    out["model_prob"] = pd.to_numeric(
        _first_existing(out, ["model_prob", "win_prob", "probability"]),
        errors="coerce",
    )
    out["edge"] = pd.to_numeric(
        _first_existing(out, ["edge", "expected_edge", "realized_edge"]),
        errors="coerce",
    )
    out["result"] = _first_existing(out, ["result"], "").map(normalize_result)
    out["hit_bool"] = pd.to_numeric(
        _first_existing(out, ["hit_bool"]),
        errors="coerce",
    )

    needs_hit = out["hit_bool"].isna()
    out.loc[needs_hit, "hit_bool"] = np.where(
        out.loc[needs_hit, "result"].eq("win"),
        1.0,
        np.where(out.loc[needs_hit, "result"].eq("loss"), 0.0, np.nan),
    )

    out["decimal_odds"] = pd.to_numeric(
        _first_existing(out, ["decimal_odds"]),
        errors="coerce",
    )
    miss_decimal = out["decimal_odds"].isna()
    out.loc[miss_decimal, "decimal_odds"] = out.loc[miss_decimal, "payout_odds"].apply(american_to_decimal)

    try:
        out = apply_clv_columns(out)
    except Exception:
        if "clv" not in out.columns:
            out["clv"] = np.nan
        if "beat_closing" not in out.columns:
            out["beat_closing"] = np.nan

    return out


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    frames = [prepare(read_any(path)) for path in candidate_files()]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)

    if "timestamp" in data.columns:
        data = data.sort_values("timestamp", ascending=False, na_position="last")

    subset = [c for c in ["timestamp", "home", "away", "market", "side", "book"] if c in data.columns]
    if subset:
        data = data.drop_duplicates(subset=subset)

    return data


st.title("📜 Your History")
st.caption("Clean rebuild with centralized CLV logic and resilient empty-state handling.")

df = load_history()

with st.sidebar:
    st.subheader("Filters")

    settled_only = st.checkbox("Settled only", value=False)

    book_series = df["book"] if "book" in df.columns else pd.Series(dtype="string")
    market_series = df["market"] if "market" in df.columns else pd.Series(dtype="string")

    books = ["All"] + sorted([x for x in book_series.dropna().astype(str).unique() if x])
    markets = ["All"] + sorted([x for x in market_series.dropna().astype(str).unique() if x])

    book_pick = st.selectbox("Book", books, index=0)
    market_pick = st.selectbox("Market", markets, index=0)

if df.empty:
    st.warning(
        "No history dataset found yet. Drop exports/history.csv, exports/your_history.csv, "
        "settled.csv, or master_likes.parquet into the recovered environment."
    )
    st.dataframe(
        pd.DataFrame(columns=["timestamp", "home", "away", "market", "side", "book"]),
        use_container_width=True,
        hide_index=True,
    )
    st.stop()

work = df.copy()

if settled_only:
    result_series = work["result"] if "result" in work.columns else pd.Series("", index=work.index)
    hit_series = work["hit_bool"] if "hit_bool" in work.columns else pd.Series(np.nan, index=work.index)
    work = work[hit_series.notna() | result_series.isin(["win", "loss", "push"])]

if book_pick != "All" and "book" in work.columns:
    work = work[work["book"].astype(str) == book_pick]

if market_pick != "All" and "market" in work.columns:
    work = work[work["market"].astype(str) == market_pick]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(work):,}")

if len(work):
    result_series = work["result"] if "result" in work.columns else pd.Series("", index=work.index)
    hit_series = work["hit_bool"] if "hit_bool" in work.columns else pd.Series(np.nan, index=work.index)
    settled_pct = float((hit_series.notna() | result_series.isin(["win", "loss", "push"])).mean())
else:
    settled_pct = float("nan")
c2.metric("Settled share", "—" if np.isnan(settled_pct) else f"{settled_pct:.1%}")

edge_series = work["edge"] if "edge" in work.columns else pd.Series(np.nan, index=work.index)
mean_edge = float(edge_series.dropna().mean()) if edge_series.notna().any() else float("nan")
c3.metric("Avg edge", "—" if np.isnan(mean_edge) else f"{mean_edge:.3f}")

clv_series = work["clv"] if "clv" in work.columns else pd.Series(np.nan, index=work.index)
mean_clv = float(clv_series.dropna().mean()) if clv_series.notna().any() else float("nan")
c4.metric("Avg CLV", "—" if np.isnan(mean_clv) else f"{mean_clv:.3f}")

st.subheader("History table")
cols = [
    c for c in [
        "timestamp",
        "home",
        "away",
        "market",
        "side",
        "book",
        "line_at_pick",
        "closing_line",
        "payout_odds",
        "closing_odds",
        "decimal_odds",
        "model_prob",
        "edge",
        "result",
        "hit_bool",
        "clv",
        "beat_closing",
    ] if c in work.columns
]

if work.empty:
    st.info("No rows match the selected filters.")
else:
    st.dataframe(work[cols], use_container_width=True, hide_index=True)

st.subheader("Recent activity")
if work.empty:
    st.info("No rows match the selected filters.")
else:
    series = work.copy()
    if "timestamp" in series.columns:
        series["day"] = pd.to_datetime(series["timestamp"], errors="coerce").dt.date
        activity = series.groupby("day", dropna=True).size().rename("picks")
        if activity.empty:
            st.caption("No timestamp values available for activity chart.")
        else:
            st.line_chart(activity)
    else:
        st.caption("No timestamp column available for activity chart.")