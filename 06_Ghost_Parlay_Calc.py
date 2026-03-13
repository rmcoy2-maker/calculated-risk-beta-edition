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
import math
import itertools

import sys
from pathlib import Path

try:
    from lib.compliance_gate import require_eligibility
except Exception:
    def require_eligibility(*args, **kwargs):
        return True

import streamlit as st
# ---- Auth fallback for recovered UI ----
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
# ---- Feature flag fallback ----
def live_enabled():
    return False

def do_expensive_refresh():
    return None
# -------------------------------
# ---------------------------------------
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
if live_enabled():
    do_expensive_refresh()
else:
    pass
import os, time, math, itertools
from typing import Optional, List
import numpy as np
import pandas as pd
try:
    from app.utils.parlay_ui import selectable_odds_table
except Exception:

    def selectable_odds_table(*_, **__):
        pass
try:
    from app.utils.diagnostics import mount_in_sidebar
except Exception:

    def mount_in_sidebar(_: str | None=None):
        return None
st.set_page_config(page_title='Ghost Parlay Calculator', page_icon='👻', layout='wide')
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

diag = mount_in_sidebar('06_Ghost_Parlay_Calc')
st.title('Ghost Parlay Calculator')
TZ = 'America/New_York'

def _exports_dir() -> Path:
    env = os.environ.get('EDGE_EXPORTS_DIR', '').strip()
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    here = Path(__file__).resolve()
    for up in [here.parent] + list(here.parents):
        if up.name.lower() == 'edge-finder':
            p = up / 'exports'
            p.mkdir(parents=True, exist_ok=True)
            return p
    p = Path.cwd() / 'exports'
    p.mkdir(parents=True, exist_ok=True)
    return p

def _latest_existing(paths: list[Path]) -> Optional[Path]:
    paths = [p for p in paths if p and p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)

def refresh_button(key: Optional[str]=None):
    if st.button('🔄 Refresh data', key=key or f'refresh_{__name__}'):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()

def _row_book(r: dict) -> str:
    return str(r.get('book', '')).strip().upper()

def _row_game_id(r: dict) -> str:
    for cand in ('game_id', 'event_id', 'id', 'gid', '_gid', 'GameID'):
        v = r.get(cand, '')
        if v not in (None, '', float('nan')):
            return str(v)
    date = str(r.get('_DateISO', r.get('_date_iso', r.get('date', '')))).strip()
    h = str(r.get('_home_nick', r.get('home', ''))).upper().strip()
    a = str(r.get('_away_nick', r.get('away', ''))).upper().strip()
    teams = '|'.join(sorted([h, a]))
    return f'{date}__{teams}'

def american_to_decimal_one(o: float | int | str) -> float:
    try:
        v = float(o)
    except Exception:
        return math.nan
    if v > 0:
        return 1.0 + v / 100.0
    if v < 0:
        return 1.0 + 100.0 / abs(v)
    return math.nan

def to_decimal(series: pd.Series) -> pd.Series:
    """Best-effort: if value looks like decimal (>1.01) keep; else treat as American."""
    s = pd.to_numeric(series, errors='coerce')
    is_decimalish = (s >= 1.01) & (s <= 1000)
    out = pd.Series(np.where(is_decimalish, s, np.nan), index=series.index, dtype='float64')
    need = out.isna()
    if need.any():
        out.loc[need] = s.loc[need].map(american_to_decimal_one)
    return out

def implied_prob_from_american(o: float | int | str) -> float:
    try:
        v = float(o)
    except Exception:
        return math.nan
    if v > 0:
        return 100.0 / (v + 100.0)
    if v < 0:
        return abs(v) / (abs(v) + 100.0)
    return math.nan

def ev_per_dollar(p: float, d: float) -> float:
    """EV per $1 stake given model probability p and decimal odds d."""
    if not 0.0 <= float(p) <= 1.0 or not float(d) > 1.0:
        return math.nan
    p = float(p)
    d = float(d)
    return p * (d - 1.0) - (1.0 - p)

def parlay_ev_per_dollar(p_list: list[float], d_list: list[float]) -> float:
    if not p_list or not d_list or len(p_list) != len(d_list):
        return math.nan
    p_all, d_prod = (1.0, 1.0)
    for p, d in zip(p_list, d_list):
        if not 0.0 <= float(p) <= 1.0 or not float(d) > 1.0:
            return math.nan
        p_all *= float(p)
        d_prod *= float(d)
    return p_all * (d_prod - 1.0) - (1.0 - p_all)

@st.cache_data(ttl=60)
def load_edges() -> tuple[pd.DataFrame, Path | None]:
    exp = _exports_dir()
    names = ['edges_graded_full_normalized_std.csv', 'edges_graded_full_normalized.csv', 'edges_graded_full.csv', 'edges_standardized.csv', 'edges_normalized.csv', 'edges.csv', 'edges_master.csv']
    paths = [exp / n for n in names]
    p = _latest_existing(paths)
    df = pd.read_csv(p, low_memory=False, encoding='utf-8-sig') if p and p.exists() else pd.DataFrame()
    return (df, p)
refresh_button(key='refresh_06_ghost')
edges, edges_p = load_edges()
st.caption(f'Edges: `{edges_p}` · rows={len(edges):,}' if edges_p else 'Edges: <not found>')
if edges.empty:
    st.info('No edges loaded. Place an edges CSV in your `exports/` directory.')
    st.stop()

def _first_col(df: pd.DataFrame, cands: list[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None
odds_col = _first_col(edges, ['odds', 'price', 'american', 'american_odds'])
prob_col = _first_col(edges, ['p_win', 'prob', 'win_prob', 'model_p', 'p'])
work = edges.copy()
if odds_col is not None:
    work['decimal'] = to_decimal(work[odds_col])
elif 'decimal' in work.columns:
    work['decimal'] = pd.to_numeric(work['decimal'], errors='coerce')
else:
    work['decimal'] = np.nan
if prob_col is not None:
    work['p_use'] = pd.to_numeric(work[prob_col], errors='coerce')
elif odds_col is not None:
    work['p_use'] = pd.to_numeric(work[odds_col], errors='coerce').map(implied_prob_from_american)
elif 'decimal' in work.columns:
    work['p_use'] = np.where(work['decimal'] > 0, 1.0 / work['decimal'], np.nan)
else:
    work['p_use'] = np.nan
pool = work[(work['decimal'] > 1.0) & work['p_use'].between(0.0, 1.0)].copy()
if pool.empty:
    st.warning('No usable rows (need decimal>1 and 0≤p≤1).')
    st.stop()
try:
    selectable_odds_table(pool, page_key='ghost', page_name='06_Ghost_Parlay_Calc', allow_same_game=False, one_per_market_per_game=True)
except Exception:
    pass
st.sidebar.header('Build Controls')
max_legs = st.sidebar.slider('Max legs', 2, 10, 3)
min_leg_prob = st.sidebar.slider('Min leg win prob', 0.3, 0.85, 0.5, 0.01)
min_abs_odds = st.sidebar.number_input('Min |American odds|', 50, 10000, 100, 10)
limit_combos = st.sidebar.number_input('Cap combinations (0=no cap)', 0, 2000000, 5000, 500)
show_top_n = st.sidebar.number_input('Show top N', 5, 250, 25, 5)
if odds_col is not None:
    pool = pool[pool[odds_col].abs() >= float(min_abs_odds)]
pool = pool[pool['p_use'] >= float(min_leg_prob)]
pool = pool.sort_values(['p_use', 'decimal'], ascending=[False, False])
st.caption(f'Pool after filters: {len(pool):,} rows')

def score_combos(df: pd.DataFrame, k: int, cap: int, topN: int) -> list[dict]:
    rows = df.to_dict('records')
    rows.sort(key=lambda r: float(r['p_use']) * (float(r['decimal']) - 1.0), reverse=True)
    best: list[dict] = []
    seen = 0
    by_book: dict[str, list[dict]] = {}
    for r in rows[:min(len(rows), 5000)]:
        by_book.setdefault(_row_book(r), []).append(r)
    for book, base_rows in by_book.items():
        base = base_rows[:min(len(base_rows), 1000)]
        for combo in itertools.combinations(base, r=int(k)):
            books = {_row_book(r) for r in combo}
            if len(books) != 1:
                continue
            gids = {_row_game_id(r) for r in combo}
            if len(gids) != len(combo):
                continue
            seen += 1
            probs = [float(r['p_use']) for r in combo]
            decs = [float(r['decimal']) for r in combo]
            ev = parlay_ev_per_dollar(probs, decs)
            best.append({'legs': combo, 'ev': ev})
            if 0 < topN < len(best):
                best = sorted(best, key=lambda x: (x['ev'], len(x['legs'])), reverse=True)[:topN]
            if 0 < cap <= seen:
                break
    return sorted(best, key=lambda x: (x['ev'], len(x['legs'])), reverse=True)[:topN]
if st.button('Generate Ghost Parlays', type='secondary'):
    res = score_combos(pool, int(max_legs), int(limit_combos), int(show_top_n))
    if not res:
        st.warning('No parlays found with current filters.')
    else:
        rows_out = []
        for r in res:
            legs = list(r['legs'])
            d_prod = float(np.prod([float(x['decimal']) for x in legs]))
            p_all = float(np.prod([float(x['p_use']) for x in legs]))
            ev_d = parlay_ev_per_dollar([float(x['p_use']) for x in legs], [float(x['decimal']) for x in legs])
            ocol = odds_col if odds_col in (pool.columns if odds_col else []) else None

            def fmt_leg(x):
                book = x.get('book', '?')
                side = x.get('side', x.get('market', '?'))
                odds = int(float(x.get(ocol, 0))) if ocol else round(float(x.get('decimal', 0.0)), 2)
                return f'{side} ({book} {odds:+d})' if ocol else f'{side} ({book} d={odds})'
            title = ' + '.join([fmt_leg(x) for x in legs])
            rows_out.append({'n_legs': len(legs), 'EV/$1': round(ev_d, 4), 'Parlay Prob': round(p_all, 6), 'Parlay Dec': round(d_prod, 4), 'Title': title})
        out_df = pd.DataFrame(rows_out).sort_values(['EV/$1', 'Parlay Prob'], ascending=[False, False]).head(int(show_top_n))
        st.subheader(f'Top {len(out_df)} Parlays')
        st.dataframe(out_df, hide_index=True, height=560, width='stretch')
        st.download_button('⬇️ Download Top Parlays (CSV)', data=out_df.to_csv(index=False).encode('utf-8'), file_name='ghost_parlays_top.csv', mime='text/csv')
else:
    st.info('Set filters and click **Generate Ghost Parlays**.')
