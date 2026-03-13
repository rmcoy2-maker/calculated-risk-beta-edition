from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
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

_HERE = Path(__file__).resolve()

try:
    _SERVING_UI = _HERE.parents[2]
except Exception:
    _SERVING_UI = Path.cwd()

if str(_SERVING_UI) not in sys.path:
    sys.path.insert(0, str(_SERVING_UI))

if str(_SERVING_UI) not in sys.path:
    sys.path.insert(0, str(_SERVING_UI))
if live_enabled():
    do_expensive_refresh()
else:
    pass
import math
import pandas as pd
import numpy as np
try:
    from app.utils.diagnostics import mount_in_sidebar
except Exception:

    def mount_in_sidebar(_: str | None=None):
        return None
st.set_page_config(page_title='Micro Calculations — Locks & Moonshots', page_icon='🌙', layout='wide')
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

mount_in_sidebar('12_Micro_Calculations')

def _exports_dir() -> Path:
    here = _HERE
    for up in [here] + list(here.parents):
        if up.name.lower() == 'edge-finder':
            p = up / 'exports'
            p.mkdir(parents=True, exist_ok=True)
            return p
    p = here.parents[2] / 'exports'
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_edges() -> pd.DataFrame:
    exp = _exports_dir()
    for name in ['edges_graded_full_normalized_std.csv', 'edges_graded_full_normalized.csv', 'edges_graded_full.csv', 'edges.csv']:
        f = exp / name
        if f.exists():
            try:
                df = pd.read_csv(f)
                df['_source_file'] = str(f)
                return df
            except Exception:
                pass
    return pd.DataFrame()

def _american_to_decimal(odds: float | int) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o >= 100:
        return 1 + o / 100
    if o <= -100:
        return 1 + 100 / abs(o)
    return np.nan

def _to_decimal(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if s.notna().any():
        guess_is_decimal = (s >= 1.01) & (s <= 1000)
        out = pd.Series(np.where(guess_is_decimal, s, np.nan), index=series.index, dtype='float64')
        need = out.isna()
        if need.any():
            out[need] = s[need].map(_american_to_decimal)
        return out
    return pd.to_numeric(series, errors='coerce')

def _implied_prob_from_decimal(d: pd.Series) -> pd.Series:
    d = pd.to_numeric(d, errors='coerce')
    return np.where(d > 0, 1.0 / d, np.nan)

def _ev_per_dollar(p: pd.Series, d: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors='coerce')
    d = pd.to_numeric(d, errors='coerce')
    return p * (d - 1) - (1 - p)

def _parlay_decimal(ds: list[float]) -> float:
    x = 1.0
    for v in ds:
        x *= float(v)
    return x

def _parlay_prob(ps: list[float]) -> float:
    x = 1.0
    for v in ps:
        x *= float(v)
    return x
st.title('Micro Calculations — Locks & Moonshots')
st.caption('Locks: singles with a win-threshold and positive EV. Moonshots: parlays meeting your min odds & with positive EV.')
edges = _load_edges()
st.button('🔄 Refresh data', on_click=lambda: st.rerun())
colL, colR = st.columns(2)
with colL:
    lock_min_ev = st.number_input('Locks min EV per $1', value=0.02, step=0.01, format='%.2f')
with colR:
    moon_min_odds = st.number_input('Moonshots: min decimal odds', value=6.0, step=0.5, format='%.2f')
    moon_legs = st.number_input('Moonshots: legs', value=3, min_value=2, max_value=10, step=1)
st.markdown('### Locks')
if edges.empty:
    st.info('No locks yet. Load edges CSVs in your `exports/` folder.')
else:
    work = edges.copy()
    if 'odds' in work.columns:
        work['decimal'] = _to_decimal(work['odds'])
    elif 'decimal' not in work.columns:
        work['decimal'] = np.nan
    pcols = [c for c in work.columns if str(c).lower() in ('p', 'p_win', 'prob', 'prob_win', 'implied_prob')]
    if pcols:
        p = pd.to_numeric(work[pcols[0]], errors='coerce')
    else:
        p = pd.Series(_implied_prob_from_decimal(work['decimal']), index=work.index)
    work['p_win'] = p
    work['_ev_per_$1'] = _ev_per_dollar(work['p_win'], work['decimal'])
    locks = work.loc[(work['_ev_per_$1'] >= float(lock_min_ev)) & work['p_win'].notna() & work['decimal'].notna()]
    st.dataframe(locks[[c for c in ['_date_iso', 'home', 'away', 'market', 'side', 'line', 'book', 'decimal', 'p_win', '_ev_per_$1'] if c in locks.columns]].sort_values('_ev_per_$1', ascending=False), hide_index=True, width='stretch')
st.markdown('### Moonshots')
st.caption('This is a simple helper: it groups by game & market, then picks the top EV legs and forms sample parlays.')
if edges.empty:
    st.info('No data to build moonshots.')
else:
    w = edges.copy()
    if 'decimal' not in w.columns:
        w['decimal'] = _to_decimal(w.get('odds', np.nan))
    w['p_win'] = pd.to_numeric(w.get('p_win', w.get('prob', w.get('implied_prob', np.nan))), errors='coerce')
    w['_ev_per_$1'] = _ev_per_dollar(w['p_win'], w['decimal'])
    pool = w[(w['_ev_per_$1'] > 0) & w['p_win'].notna() & w['decimal'].notna()].copy()
    if pool.empty:
        st.info('No positive-EV legs found.')
    else:
        keys = [c for c in ['game_id', 'market'] if c in pool.columns]
        if keys:
            pool = pool.sort_values('_ev_per_$1', ascending=False).drop_duplicates(subset=keys, keep='first')
        sample = pool.sort_values('_ev_per_$1', ascending=False).head(int(moon_legs))
        if len(sample) < int(moon_legs):
            st.warning(f'Only found {len(sample)} legs ≥0 EV. Showing a partial sample.')
        parl_odds = _parlay_decimal(sample['decimal'].tolist())
        parl_prob = _parlay_prob(sample['p_win'].tolist())
        ev_per1 = parl_prob * (parl_odds - 1) - (1 - parl_prob)
        st.write('**Sample parlay** (top EV legs):')
        st.dataframe(sample[[c for c in ['home', 'away', 'market', 'side', 'line', 'book', 'decimal', 'p_win', '_ev_per_$1'] if c in sample.columns]], hide_index=True, width='stretch')
        st.metric('Parlay decimal odds', f'{parl_odds:,.2f}')
        st.metric('Parlay win prob (model)', f'{parl_prob:.2%}')
        st.metric('EV per $1', f'{ev_per1:+.3f}')
        if parl_odds >= float(moon_min_odds) and ev_per1 > 0:
            st.success('✅ Meets moonshot thresholds.')
        else:
            st.info('Does not meet moonshot thresholds yet.')
