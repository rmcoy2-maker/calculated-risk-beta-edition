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

try:
    from lib.compliance_gate import require_eligibility
except Exception:
    def require_eligibility(*args, **kwargs):
        return True

import streamlit as st
import sys
from pathlib import Path

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
st.set_page_config(page_title='09 Lines Explorer', page_icon='📈', layout='wide')
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

import os, time, math
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
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

def _age_str(p: Path) -> str:
    try:
        secs = int(time.time() - p.stat().st_mtime)
        return f'{secs}s' if secs < 60 else f'{secs // 60}m'
    except Exception:
        return 'n/a'
_ALIAS = {'REDSKINS': 'COMMANDERS', 'WASHINGTON': 'COMMANDERS', 'FOOTBALL': 'COMMANDERS', 'OAKLAND': 'RAIDERS', 'LV': 'RAIDERS', 'LAS': 'RAIDERS', 'VEGAS': 'RAIDERS', 'SD': 'CHARGERS', 'STL': 'RAMS'}

def _nickify(series: pd.Series) -> pd.Series:
    s = series.astype('string').fillna('').str.upper()
    s = s.str.replace('[^A-Z0-9 ]+', '', regex=True).str.strip().replace(_ALIAS)
    return s.str.replace('\\s+', '_', regex=True)

def _best_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _ensure_nicks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    home_c = _best_col(df, ['_home_nick', 'home_nick', 'home', 'home_team', 'Home', 'HOME', 'team_home'])
    away_c = _best_col(df, ['_away_nick', 'away_nick', 'away', 'away_team', 'Away', 'AWAY', 'team_away'])
    if home_c is None:
        df['_home_nick'] = pd.Series([''] * len(df), dtype='string')
    else:
        df['_home_nick'] = _nickify(df[home_c].astype('string'))
    if away_c is None:
        df['_away_nick'] = pd.Series([''] * len(df), dtype='string')
    else:
        df['_away_nick'] = _nickify(df[away_c].astype('string'))
    return df

def _norm_market(m) -> str:
    m = (str(m) or '').strip().lower()
    if m in {'h2h', 'ml', 'moneyline', 'money line'}:
        return 'H2H'
    if m.startswith('spread') or m in {'spread', 'spreads'}:
        return 'SPREADS'
    if m.startswith('total') or m in {'total', 'totals'}:
        return 'TOTALS'
    return m.upper()

def _odds_to_decimal(o: pd.Series) -> pd.Series:
    o = pd.to_numeric(o, errors='coerce')
    return np.where(o > 0, 1 + o / 100.0, np.where(o < 0, 1 + 100.0 / np.abs(o), np.nan))

def _ensure_date_iso(df: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    if len(df) == 0:
        return df
    for c in candidates:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors='coerce', utc=True)
            df['_date_iso'] = s.dt.tz_convert(TZ).dt.strftime('%Y-%m-%d')
            break
    if '_date_iso' not in df.columns:
        df['_date_iso'] = pd.Series(pd.NA, index=df.index, dtype='string')
    return df
def latest_batch(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for col in ['_snapshot_ts_utc', 'snapshot_ts_utc', 'snapshot_ts', '_ts', 'ts']:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors='coerce', utc=True)
            last = ts.max()
            return df[ts == last].copy()
    return df

def within_next_week(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if '_date_iso' not in df.columns:
        return df
    d = pd.to_datetime(df['_date_iso'], errors='coerce', utc=True).dt.tz_convert(TZ)
    today = pd.Timestamp.now(tz=TZ).normalize()
    end = today + pd.Timedelta(days=7)
    return df[(d >= today) & (d <= end)].copy()
def refresh_button(label: str = "🔄 Refresh", key: str | None = None, help: str | None = None):
    import streamlit as _st
    if _st.button(label, key=key, help=help):
        _st.rerun()

if st.button("🔄 Refresh data", use_container_width=False):
    st.rerun()


def _latest_csv(paths: list[Path]) -> Optional[Path]:
    paths = [p for p in paths if p and p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)

@st.cache_data(ttl=60)
def load_live() -> tuple[pd.DataFrame, Path]:
    exp = _exports_dir()
    cand = [exp / 'lines_live.csv', exp / 'lines_live_latest.csv']
    p = _latest_csv(cand) or cand[0]
    df = pd.read_csv(p, low_memory=False, encoding='utf-8-sig') if p.exists() else pd.DataFrame()
    return (df, p)

@st.cache_data(ttl=60)
def load_open_close() -> tuple[pd.DataFrame, Path]:
    exp = _exports_dir()
    cand = [exp / 'lines_open_close.csv', exp / 'lines_open_close_latest.csv']
    p = _latest_csv(cand) or cand[0]
    df = pd.read_csv(p, low_memory=False, encoding='utf-8-sig') if p.exists() else pd.DataFrame()
    return (df, p)

@st.cache_data(ttl=60)
def load_edges() -> tuple[pd.DataFrame, Path]:
    exp = _exports_dir()
    names = ['edges_standardized.csv', 'edges_graded_full_normalized_std.csv', 'edges_graded_full.csv', 'edges_normalized.csv', 'edges_master.csv']
    paths = [exp / n for n in names]
    p = _latest_csv(paths) or paths[0]
    df = pd.read_csv(p, low_memory=False, encoding='utf-8-sig') if p.exists() else pd.DataFrame()
    return (df, p)
diag = mount_in_sidebar('09_Lines_Explorer')
st.title('Lines — Explorer')
refresh_button(key='refresh_09_lines')
live, live_path = load_live()
oc, oc_path = load_open_close()
st.caption(f'Live lines: `{live_path}` · rows={len(live):,} · age={_age_str(live_path)}')
st.caption(f'Open/Close: `{oc_path}` · rows={len(oc):,}' + ('' if len(oc) else ' (optional)'))
if live.empty:
    st.warning('No live lines found. Point EDGE_EXPORTS_DIR to your exports folder.')
    st.stop()
live = _ensure_date_iso(live, ['_date_iso', 'event_date', 'commence_time', 'date', 'game_date', 'Date']).copy()
live = _ensure_nicks(live)
live['_market_norm'] = live.get('_market_norm', live.get('market', pd.Series(index=live.index))).map(_norm_market)
live['decimal'] = _odds_to_decimal(live.get('price', pd.Series(index=live.index)))
live['imp_prob'] = 1.0 / live['decimal']
live['_key'] = live['_date_iso'].astype('string') + '|' + live['_home_nick'].astype('string') + '|' + live['_away_nick'].astype('string') + '|' + live['_market_norm'].astype('string') + '|' + live.get('side', pd.Series(index=live.index)).astype('string') + '|' + live.get('book', pd.Series(index=live.index)).astype('string')
today_iso = pd.Timestamp.now(tz=TZ).strftime('%Y-%m-%d')
dates_avail = sorted([d for d in live['_date_iso'].dropna().unique().tolist()])
default_date = today_iso if today_iso in dates_avail else dates_avail[-1] if dates_avail else None
st.sidebar.header('Slate & Filters')
date_pick = st.sidebar.selectbox('Date', options=dates_avail, index=dates_avail.index(default_date) if default_date in dates_avail else 0)
view = live[live['_date_iso'] == date_pick].copy()
mk_opts = ['(all)'] + sorted(view['_market_norm'].dropna().unique().tolist())
book_opts = sorted([b for b in view.get('book', pd.Series(dtype='string')).dropna().unique().tolist()])
side_opts = sorted([s for s in view.get('side', pd.Series(dtype='string')).dropna().unique().tolist()])
colA, colB = st.sidebar.columns(2)
with colA:
    market_pick = st.selectbox('Market', options=mk_opts)
with colB:
    sides_pick = st.multiselect('Sides', options=side_opts, default=side_opts)
books_pick = st.sidebar.multiselect('Books', options=book_opts, default=book_opts)
team_query = st.sidebar.text_input('Team contains (RAIDERS, 49ERS, etc.)', '')
pr = pd.to_numeric(view.get('price', pd.Series(index=view.index)), errors='coerce')
ln = pd.to_numeric(view.get('line', pd.Series(index=view.index)), errors='coerce')
p_min, p_max = (int(np.nanmin(pr)) if pr.notna().any() else -500, int(np.nanmax(pr)) if pr.notna().any() else 500)
l_min, l_max = (float(np.nanmin(ln)) if ln.notna().any() else -30.0, float(np.nanmax(ln)) if ln.notna().any() else 30.0)
price_rng = st.sidebar.slider('American odds (price) range', p_min, p_max, (p_min, p_max), step=5)
line_rng = st.sidebar.slider('Line range', float(l_min), float(l_max), (float(l_min), float(l_max)))
if market_pick != '(all)':
    view = view[view['_market_norm'] == market_pick]
if len(sides_pick):
    view = view[view.get('side', pd.Series(index=view.index)).isin(sides_pick)]
if len(books_pick):
    view = view[view.get('book', pd.Series(index=view.index)).isin(books_pick)]
if team_query.strip():
    q = team_query.strip().upper()
    view = view[view['_home_nick'].str.contains(q, na=False) | view['_away_nick'].str.contains(q, na=False)]
view = view[pr.between(price_rng[0], price_rng[1], inclusive='both')]
view = view[ln.between(line_rng[0], line_rng[1], inclusive='both')]
if not oc.empty:
    need = ['_date_iso', '_home_nick', '_away_nick', '_market_norm', 'side', 'book']
    for c in need:
        if c not in oc.columns:
            oc[c] = pd.NA
    oc['_key'] = oc['_date_iso'].astype('string') + '|' + oc['_home_nick'].astype('string') + '|' + oc['_away_nick'].astype('string') + '|' + oc['_market_norm'].astype('string') + '|' + oc['side'].astype('string') + '|' + oc['book'].astype('string')
    cols = ['_key', 'open_line', 'open_price', 'close_line', 'close_price', 'open_ts_utc', 'close_ts_utc']
    oc_cols = [c for c in cols if c in oc.columns]
    view = view.merge(oc[oc_cols], on='_key', how='left')
cols = [c for c in ['_date_iso', '_away_nick', '_home_nick', 'book', '_market_norm', 'side', 'line', 'price', 'decimal', 'imp_prob', 'open_line', 'open_price', 'close_line', 'close_price'] if c in view.columns]
st.write(f'Showing {len(view):,} rows for {date_pick}')
st.dataframe(view[cols], hide_index=True, width='stretch')
try:
    import pandas as _ef_pd
    from pathlib import Path as _ef_Path
    _ef = locals().get('diag', None)
    if _ef:
        for _nm in ('edges_p', 'live_p', 'oc_path', 'edges_path', 'live_path', 'scores_path', 'scores_p', 'epath', 'spath', '_lines_p', '_edges_p'):
            _p = locals().get(_nm, None)
            if _p:
                try:
                    _ef.check_file(_ef_Path(str(_p)), required=False, label=_nm)
                except Exception:
                    pass
        for _dfn in ('edges', 'live', 'oc', 'scores', 'joined', 'view'):
            _df = locals().get(_dfn, None)
            try:
                if isinstance(_df, _ef_pd.DataFrame):
                    _ef.log_df(_df, _dfn)
            except Exception:
                pass
except Exception:
    pass
