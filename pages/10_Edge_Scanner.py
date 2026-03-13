from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Safe shims for recovered environments
# -----------------------------------------------------------------------------
try:
    from app.lib.auth import login, show_logout  # type: ignore
except Exception:
    def login(required: bool = False):
        class _Auth:
            ok = True
            authenticated = True
        return _Auth()

    def show_logout():
        return None


try:
    from app.lib.compliance_gate import require_eligibility  # type: ignore
except Exception:
    def require_eligibility(*args, **kwargs):
        return True


try:
    from app.utils.diagnostics import mount_in_sidebar  # type: ignore
except Exception:
    def mount_in_sidebar(page_name: str):
        return None


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="10 Edge Scanner",
    page_icon="📈",
    layout="wide",
)

auth = login(required=False)
if not getattr(auth, "ok", True):
    st.stop()

show_logout()

try:
    require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})
except Exception:
    pass


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
        if (up / "exports").exists():
            return up / "exports"

    p = Path.cwd() / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _latest_existing(paths: list[Path]) -> Optional[Path]:
    found = [p for p in paths if p.exists() and p.is_file()]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def _safe_read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return pd.DataFrame()


def american_to_prob(odds) -> float:
    if pd.isna(odds):
        return np.nan
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return abs(o) / (abs(o) + 100.0)
    return np.nan


def american_to_decimal(odds) -> float:
    if pd.isna(odds):
        return np.nan
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 1.0 + (o / 100.0)
    if o < 0:
        return 1.0 + (100.0 / abs(o))
    return np.nan


def payout_per_1u(odds) -> float:
    dec = american_to_decimal(odds)
    if pd.isna(dec):
        return np.nan
    return dec - 1.0


def _best_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def _first_present(df: pd.DataFrame, candidates: list[str], default=None) -> pd.Series:
    c = _best_col(df, candidates)
    if c is not None:
        return df[c]
    if default is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    return pd.Series([default] * len(df), index=df.index)


@st.cache_data(ttl=60)
def load_edges() -> tuple[pd.DataFrame, Optional[Path]]:
    exp = _exports_dir()
    names = [
        "edges_standardized.csv",
        "edges_graded_full_normalized_std.csv",
        "edges_graded_full.csv",
        "edges_normalized.csv",
        "edges_master.csv",
    ]
    p = _latest_existing([exp / n for n in names])
    df = _safe_read_csv(p)
    return df, p


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
diag = mount_in_sidebar("10_Edge_Scanner")

st.title("📈 Edge Scanner")

edges, edges_path = load_edges()

if edges_path is None:
    st.warning("No edges CSV found in exports/.")
    st.stop()

st.caption(f"Edges file: `{edges_path}` · rows={len(edges):,}")

if edges.empty:
    st.warning("Edges file loaded but contains no rows.")
    st.stop()


# -----------------------------------------------------------------------------
# Normalize fields
# -----------------------------------------------------------------------------
work = edges.copy()

work["_game_id"] = _first_present(work, ["game_id", "event_id", "id"], default="")
work["_market"] = _first_present(work, ["market", "market_type", "bet_type"], default="")
work["_side"] = _first_present(work, ["side", "selection", "pick_side", "team", "outcome"], default="")
work["_book"] = _first_present(work, ["book", "sportsbook", "book_name"], default="")

work["_odds"] = pd.to_numeric(
    _first_present(work, ["odds", "price", "american_odds", "line_price"]),
    errors="coerce",
)

work["_p_win"] = pd.to_numeric(
    _first_present(work, ["p_win", "win_prob", "model_prob", "pred_prob", "probability"]),
    errors="coerce",
)

# fallback to implied probability if model probability missing
work["_implied_prob"] = work["_odds"].map(american_to_prob)
work["_p_win"] = work["_p_win"].where(work["_p_win"].notna(), work["_implied_prob"])

work["_decimal"] = work["_odds"].map(american_to_decimal)
work["_payout_per_1u"] = work["_odds"].map(payout_per_1u)

work["_ev_per_1u"] = np.where(
    work["_p_win"].notna() & work["_payout_per_1u"].notna(),
    (work["_p_win"] * work["_payout_per_1u"]) - (1.0 - work["_p_win"]),
    np.nan,
)

# edge % if not already present
existing_edge = pd.to_numeric(
    _first_present(work, ["edge_pct", "edge", "ev_pct", "expected_edge"]),
    errors="coerce",
)
work["_edge_pct"] = existing_edge.where(
    existing_edge.notna(),
    (work["_p_win"] - work["_implied_prob"]) * 100.0,
)


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
left, right, right2 = st.columns(3)

with left:
    min_ev = st.slider("Min EV per 1u", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)

with right:
    min_p = st.slider("Min win probability", min_value=0.0, max_value=1.0, value=0.45, step=0.01)

with right2:
    min_edge = st.slider("Min edge %", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)

market_opts = sorted([str(x) for x in work["_market"].dropna().astype(str).unique() if str(x).strip()])
book_opts = sorted([str(x) for x in work["_book"].dropna().astype(str).unique() if str(x).strip()])

col1, col2 = st.columns(2)
with col1:
    markets_pick = st.multiselect("Markets", market_opts, default=market_opts)
with col2:
    books_pick = st.multiselect("Books", book_opts, default=book_opts)

side_query = st.text_input("Side contains", "")

view = work.copy()

view = view[view["_ev_per_1u"].fillna(-999) >= min_ev]
view = view[view["_p_win"].fillna(-1) >= min_p]
view = view[view["_edge_pct"].fillna(-999) >= min_edge]

if markets_pick:
    view = view[view["_market"].astype(str).isin(markets_pick)]

if books_pick:
    view = view[view["_book"].astype(str).isin(books_pick)]

if side_query.strip():
    q = side_query.strip().upper()
    view = view[view["_side"].astype(str).str.upper().str.contains(q, na=False)]

view = view.sort_values(["_ev_per_1u", "_edge_pct"], ascending=[False, False], na_position="last")


# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{len(view):,}")
k2.metric("Avg EV/1u", "—" if view["_ev_per_1u"].dropna().empty else f"{view['_ev_per_1u'].dropna().mean():.3f}")
k3.metric("Avg p_win", "—" if view["_p_win"].dropna().empty else f"{view['_p_win'].dropna().mean():.3f}")
k4.metric("Avg edge %", "—" if view["_edge_pct"].dropna().empty else f"{view['_edge_pct'].dropna().mean():.2f}")


# -----------------------------------------------------------------------------
# Table
# -----------------------------------------------------------------------------
display_cols = [
    c for c in [
        "_game_id",
        "_market",
        "_side",
        "_book",
        "_odds",
        "_implied_prob",
        "_p_win",
        "_edge_pct",
        "_ev_per_1u",
    ]
    if c in view.columns
]

pretty = view[display_cols].copy()
pretty = pretty.rename(columns={
    "_game_id": "game_id",
    "_market": "market",
    "_side": "side",
    "_book": "book",
    "_odds": "odds",
    "_implied_prob": "implied_prob",
    "_p_win": "p_win",
    "_edge_pct": "edge_pct",
    "_ev_per_1u": "ev_per_1u",
})

st.dataframe(
    pretty.reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Rows after filters: {len(pretty):,}")

csv_bytes = pretty.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered edges.csv",
    data=csv_bytes,
    file_name="edges_filtered.csv",
    mime="text/csv",
)