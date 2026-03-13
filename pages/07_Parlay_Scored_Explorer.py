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

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Parlay Scored Explorer", page_icon="📊", layout="wide")
require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})

# === Nudge + Session ===
begin_session()
touch_session()
if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")
bump_usage("page_visit")
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")
# === /Nudge + Session ===

# -------------- PAGE BODY --------------
require_allowed_page("pages/07_Parlay_Scored_Explorer.py")
beta_banner()
mount_in_sidebar("07_Parlay_Scored_Explorer")

PAGE_PROTECTED = False
auth = login(required=PAGE_PROTECTED)
if not getattr(auth, "ok", True):
    st.stop()

if not getattr(auth, "authenticated", True):
    st.info("You are in read-only mode.")
show_logout()

st.title("Parlay Scored Explorer")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "streamlit_app.py").exists():
            return p
    return Path.cwd()


def _exports_dir() -> Path:
    return _repo_root() / "exports"


def load_edges_or_scores() -> tuple[pd.DataFrame, str]:
    exports = _exports_dir()
    scores = exports / "parlay_scores.csv"
    edges = exports / "edges.csv"

    if scores.exists() and scores.is_file() and scores.stat().st_size > 0:
        try:
            return pd.read_csv(scores, low_memory=False), str(scores)
        except Exception as e:
            st.warning(f"Could not load parlay scores file `{scores.name}`: {e}")

    if edges.exists() and edges.is_file() and edges.stat().st_size > 0:
        try:
            return pd.read_csv(edges, low_memory=False), str(edges)
        except Exception as e:
            st.warning(f"Could not load edges file `{edges.name}`: {e}")

    return pd.DataFrame(), str(scores)


def _num_unique(df: pd.DataFrame, col: str) -> list[int]:
    if col not in df.columns:
        return []
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return sorted(s.astype(int).unique().tolist())


df, src_path = load_edges_or_scores()

st.write(f"**Source:** `{src_path}` • **Rows:** {len(df):,}")

if df.empty:
    st.warning("No parlay scores or edges file was found in exports/.")
    st.stop()

has_score = "parlay_proba" in df.columns
if not has_score:
    st.warning(
        "No `parlay_proba` found. Did you run `predict_parlay_score.py`? Showing base edges instead.",
        icon="⚠️",
    )

seasons_all = _num_unique(df, "season")
weeks_all = _num_unique(df, "week")

left, right = st.columns(2)
with left:
    season_options = ["All"] + [str(y) for y in seasons_all]
    season_sel = st.multiselect("Season", season_options, default=["All"])

with right:
    week_options = ["All"] + [str(w) for w in (weeks_all or list(range(1, 23)))]
    week_sel = st.multiselect("Week", week_options, default=["All"])

work = df.copy()

if "All" not in season_sel and seasons_all and "season" in work.columns:
    keep = {int(x) for x in season_sel if str(x).isdigit()}
    season_num = pd.to_numeric(work["season"], errors="coerce")
    work = work[season_num.isin(keep)]

if "All" not in week_sel and "week" in work.columns:
    keep = {int(x) for x in week_sel if str(x).isdigit()}
    week_num = pd.to_numeric(work["week"], errors="coerce")
    work = work[week_num.isin(keep)]

if has_score:
    thr = st.slider("Min parlay probability", 0.0, 1.0, 0.7, 0.01)
    work = work[pd.to_numeric(work["parlay_proba"], errors="coerce").fillna(0) >= thr]
    st.caption(f"{len(work):,} rows ≥ {thr:.2f}")

pref_cols = [
    c
    for c in [
        "ts",
        "season",
        "week",
        "sport",
        "league",
        "market",
        "side",
        "line",
        "odds",
        "p_win",
        "ev",
        "parlay_proba",
        "dec_comb",
        "legs",
        "parlay_stake",
        "team_name",
        "home",
        "away",
        "game_id",
    ]
    if c in work.columns
]

with st.expander("Preview (top 2,500 rows)", expanded=True):
    st.dataframe((work[pref_cols] if pref_cols else work).head(2500), width="stretch")

st.download_button(
    "Download filtered as CSV",
    data=work.to_csv(index=False).encode("utf-8"),
    file_name="parlay_scored_filtered.csv",
    mime="text/csv",
)

if has_score:
    arr = pd.to_numeric(work.get("parlay_proba"), errors="coerce").dropna().to_numpy()
    if arr.size:
        qs = np.quantile(arr, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        st.write(
            "**Parlay probability quantiles:**",
            {
                k: round(float(v), 3)
                for k, v in zip(
                    ["min", "25%", "50%", "75%", "90%", "95%", "99%", "max"],
                    qs,
                )
            },
        )

try:
    selectable_odds_table(
        work,
        page_key="parlay_scored",
        page_name="07_Parlay_Scored_Explorer",
    )
except Exception:
    pass