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
    from lib.compliance_gate import require_eligibility
except Exception:
    def require_eligibility(*args, **kwargs):
        return True

import streamlit as st
st.set_page_config(page_title='11 Hedge Finder', page_icon='📈', layout='wide')
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

# -------------- PAGE BODY --------------
require_allowed_page('pages/11_Hedge_Finder.py')
beta_banner()

import os, time
from itertools import combinations
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

try:
    from app.utils.diagnostics import mount_in_sidebar
except Exception:
    def mount_in_sidebar(page_name: str): return None


PAGE_PROTECTED = False
auth = login(required=PAGE_PROTECTED)
if not auth.ok: st.stop()
show_logout()
auth = login(required=False)
if not auth.authenticated: st.info('You are in read-only mode.')
show_logout()

TZ = 'America/New_York'

def _exports_dir() -> Path:
    env = os.environ.get('EDGE_EXPORTS_DIR', '').strip()
    if env:
        p = Path(env); p.mkdir(parents=True, exist_ok=True); return p
    here = Path(__file__).resolve()
    for up in [here.parent] + list(here.parents):
        if up.name.lower() == 'edge-finder':
            p = up / 'exports'; p.mkdir(parents=True, exist_ok=True); return p
    p = Path.cwd() / 'exports'; p.mkdir(parents=True, exist_ok=True); return p

def _latest_csv(paths: list[Path]) -> Optional[Path]:
    paths = [p for p in paths if p and p.exists()]
    if not paths: return None
    return max(paths, key=lambda p: p.stat().st_mtime)

@st.cache_data(ttl=60)
def load_edges() -> tuple[pd.DataFrame, Path]:
    exp = _exports_dir()
    names = ['edges_standardized.csv', 'edges_graded_full_normalized_std.csv', 'edges_graded_full.csv', 'edges_normalized.csv', 'edges_master.csv']
    paths = [exp / n for n in names]
    p = _latest_csv(paths) or paths[0]
    df = pd.read_csv(p, low_memory=False, encoding='utf-8-sig') if p.exists() else pd.DataFrame()
    return (df, p)

diag = mount_in_sidebar('11_Hedge_Finder')
st.title('Hedge Finder')
edges, edges_p = load_edges()
st.caption(f'Edges: `{edges_p}` · rows={len(edges):,}')
if edges.empty: st.stop()

for need in ['game_id', 'market', 'side', 'book']:
    if need not in edges.columns:
        edges[need] = pd.NA
edges['odds'] = pd.to_numeric(edges.get('odds', edges.get('price', pd.Series(index=edges.index))), errors='coerce')

top = st.columns([1, 1.2, 2, 1])
with top[0]:
    newest_first = st.toggle('Newest first', value=True)
with top[1]:
    mode = st.selectbox('Side match', ['Strict (OU/Home-Away)', 'Loose (any different)'], index=1)
with top[2]:
    odds_min, odds_max = st.slider('Odds window (American, abs)', 100, 300, (100, 200))
with top[3]:
    run = st.button('Find hedges', type='primary', use_container_width=True)

if newest_first and 'sort_ts' in edges.columns and pd.to_datetime(edges['sort_ts'], errors='coerce').notna().any():
    edges = edges.sort_values('sort_ts', ascending=False, na_position='last')

games = edges['game_id'].dropna().astype(str).unique().tolist()
game = st.selectbox('Game', ['All'] + games)
subset = edges if game == 'All' else edges[edges['game_id'].astype(str) == game]
subset = subset[subset['odds'].abs().between(odds_min, odds_max, inclusive='both')]

def is_opposite(a: str, b: str, mode: str) -> bool:
    if not a or not b: return False
    A, B = (str(a).strip().lower(), str(b).strip().lower())
    if mode == 'Strict (OU/Home-Away)':
        return (A, B) in {('over','under'), ('home','away'), ('away','home')} or (B, A) in {('over','under'), ('home','away'), ('away','home')}
    return A != B

if not run:
    st.info('Adjust filters and click **Find hedges**.')
    st.stop()

rows = []
for (gid, mkt), g in subset.groupby(['game_id', 'market'], dropna=False):
    g = g.reset_index(drop=True)
    for i, j in combinations(range(len(g)), 2):
        a, b = (g.loc[i], g.loc[j])
        if is_opposite(a.get('side', ''), b.get('side', ''), mode):
            rows.append({'game_id': str(gid), 'market': str(mkt),
                         'a_side': str(a.get('side', '')), 'a_odds': a.get('odds'), 'a_book': str(a.get('book', '')),
                         'b_side': str(b.get('side', '')), 'b_odds': b.get('odds'), 'b_book': str(b.get('book', ''))})

if not rows:
    st.info('No hedges found at current filters.')
else:
    out = pd.DataFrame(rows)
    st.dataframe(out, width='stretch')
    st.download_button('Download hedges.csv', data=out.to_csv(index=False).encode('utf-8'), file_name='hedges.csv', mime='text/csv')

try:
    if 'edges' in globals() and isinstance(edges, pd.DataFrame):
        selectable_odds_table(edges, page_key='hedge_finder', page_name='11_Hedge_Finder')
except Exception:
    pass
