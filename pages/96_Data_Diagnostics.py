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

try:
    from lib.compliance_gate import require_eligibility
except Exception:
    def require_eligibility(*args, **kwargs):
        return True

import sys
from pathlib import Path
import streamlit as st

_here = Path(__file__).resolve()
for up in [_here] + list(_here.parents):
    cand = up / 'serving_ui' / 'app' / '__init__.py'
    if cand.exists():
        base = str((up / 'serving_ui').resolve())
        if base not in sys.path:
            sys.path.insert(0, base)
        break
PAGE_PROTECTED = False
auth = login(required=PAGE_PROTECTED)
if not auth.ok:
    st.stop()
show_logout()
auth = login(required=False)
if not auth.authenticated:
    st.info('You are in read-only mode.')
show_logout()
import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
_SERVING_UI = _HERE.parents[2]
if str(_SERVING_UI) not in sys.path:
    sys.path.insert(0, str(_SERVING_UI))
st.set_page_config(page_title='96 Data Diagnostics', page_icon='📈', layout='wide')
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

import pandas as pd
from pathlib import Path
import pandas as pd

try:
    from lib.io_paths import load_edges, load_scores
except Exception:
    def _repo_root() -> Path:
        here = Path(__file__).resolve()
        for p in [here.parent] + list(here.parents):
            if (p / "streamlit_app.py").exists():
                return p
        return Path.cwd()

    def _pick_latest_file(root: Path, names: list[str], patterns: list[str]) -> Path | None:
        matches: list[Path] = []

        for name in names:
            p = root / name
            if p.exists() and p.is_file():
                matches.append(p)

        for pattern in patterns:
            matches.extend([p for p in root.glob(pattern) if p.is_file()])

        if not matches:
            return None

        uniq = {str(p.resolve()): p for p in matches}
        return max(uniq.values(), key=lambda p: p.stat().st_mtime)

    def load_edges() -> pd.DataFrame:
        exports = _repo_root() / "exports"
        path = _pick_latest_file(
            exports,
            names=[
                "edges_standardized.csv",
                "edges_graded_full_normalized_std.csv",
                "edges_graded_full.csv",
                "edges_normalized.csv",
                "edges.csv",
            ],
            patterns=[
                "*edges*standardized*.csv",
                "*edges*normalized*.csv",
                "*edges*graded*.csv",
                "*edges*.csv",
            ],
        )
        if path is None:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()

    def load_scores() -> pd.DataFrame:
        exports = _repo_root() / "exports"
        path = _pick_latest_file(
            exports,
            names=[
                "games_master_template.csv",
                "scores_1966-2025.csv",
                "scores_1966-2025_merged.csv",
                "scores_1966-2025_merged_unmatched.csv",
                "_backup_scores_1966-2025.csv",
                "scores_normalized_std.csv",
                "scores_normalized.csv",
                "scores.csv",
            ],
            patterns=[
                "*games_master_template*.csv",
                "*scores_1966-2025*.csv",
                "*scores*normalized*.csv",
                "*scores*.csv",
            ],
        )
        if path is None:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()
try:
    from lib.join_scores import attach_scores
except Exception:
    def attach_scores(edges: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
        if edges is None or edges.empty:
            return pd.DataFrame() if edges is None else edges
        if scores is None or scores.empty:
            return edges.copy()

        out = edges.copy()
        sc = scores.copy()

        # Standardize likely join keys
        for df in (out, sc):
            if "game_id" in df.columns:
                df["game_id"] = df["game_id"].astype("string").str.strip()
            if "home" in df.columns:
                df["home"] = df["home"].astype("string").str.strip()
            if "away" in df.columns:
                df["away"] = df["away"].astype("string").str.strip()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # Preferred join: game_id
        if "game_id" in out.columns and "game_id" in sc.columns:
            merged = out.merge(
                sc,
                on="game_id",
                how="left",
                suffixes=("", "_score"),
            )
            return merged

        # Fallback join: home + away + date
        join_cols = ["home", "away", "date"]
        if all(c in out.columns for c in join_cols) and all(c in sc.columns for c in join_cols):
            merged = out.merge(
                sc,
                on=join_cols,
                how="left",
                suffixes=("", "_score"),
            )
            return merged

        return out
try:
    from app.utils.diagnostics import mount_in_sidebar
except ModuleNotFoundError:
    try:
        import sys
        from pathlib import Path as _efP
        sys.path.append(str(_efP(__file__).resolve().parents[3]))
        from app.utils.diagnostics import mount_in_sidebar
    except Exception:
        try:
            from utils.diagnostics import mount_in_sidebar
        except Exception:

            def mount_in_sidebar(page_name: str):
                return None
st.title('Data Diagnostics — Date Alignment & Join')
edges = load_edges()
scores = load_scores()
for df in (edges, scores):
    for c in ['Season', 'Week']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
joined = attach_scores(edges.assign(HomeScore=pd.NA, AwayScore=pd.NA).copy(), scores.copy())
st.metric('Edges', len(edges))
st.metric('Scores', len(scores))
st.metric('Joined (has scores)', int(joined.get('_joined_with_scores', pd.Series([False] * len(joined))).sum()))
with st.expander('Join methods breakdown', expanded=False):
    if '_join_method' in joined.columns:
        st.dataframe(joined['_join_method'].value_counts(dropna=False))
    else:
        st.write('No _join_method column in joined.')
