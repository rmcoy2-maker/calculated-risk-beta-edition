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
if live_enabled():
    do_expensive_refresh()
else:
    pass
st.set_page_config(page_title='96 Compare Models', page_icon='📈', layout='wide')
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
import math
from pathlib import Path
from typing import Dict, Any, Iterable
import numpy as np
import pandas as pd
st.title('📊 Compare Models')
BUILD_TAG = 'compare-models-v2'

def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parents[3], here.parents[2], here.parents[1]]:
        if (p / 'exports').exists():
            return p
    return here.parents[3] if len(here.parents) >= 4 else Path.cwd()

def _exports_dir() -> Path:
    root = _project_root()
    exp = root / 'exports'
    exp.mkdir(parents=True, exist_ok=True)
    return exp

def _safe_str_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out

def _to_series(x) -> pd.Series:
    """Normalize scalars/arrays/Series/DataFrames to a 1D float Series."""
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors='coerce')
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series([], dtype='float64')
        return pd.to_numeric(x.iloc[:, 0], errors='coerce')
    if np.isscalar(x) or x is None:
        return pd.Series([x], dtype='float64')
    return pd.to_numeric(pd.Series(x), errors='coerce')

def american_to_decimal(odds: Any) -> float | np.nan:
    """Convert American odds to decimal payout multiplier (incl. stake)."""
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if math.isnan(o):
        return np.nan
    if o > 0:
        return 1.0 + o / 100.0
    if o < 0:
        return 1.0 + 100.0 / abs(o)
    return np.nan

def add_probs_and_ev(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: ensure p_win, payout_decimal, _ev_per_$1 exist."""
    out = df.copy()
    p_candidates = ['p_win', 'prob', 'p', 'prob_win', 'win_prob']
    p = None
    for c in p_candidates:
        if c in out.columns:
            p = pd.to_numeric(out[c], errors='coerce').clip(0, 1)
            break
    if p is None:
        out['p_win'] = np.nan
    else:
        out['p_win'] = p
    dec_candidates = ['_payout_decimal', 'payout_decimal', 'decimal_odds', 'payout']
    amer_candidates = ['american_odds', 'odds', 'american']
    dec = None
    for c in dec_candidates:
        if c in out.columns:
            dec = pd.to_numeric(out[c], errors='coerce')
            break
    if dec is None:
        amer = None
        for c in amer_candidates:
            if c in out.columns:
                amer = out[c]
                break
        if amer is not None:
            out['_payout_decimal'] = pd.to_numeric(amer, errors='coerce').map(american_to_decimal)
        else:
            out['_payout_decimal'] = np.nan
    else:
        out['_payout_decimal'] = dec
    p = pd.to_numeric(out.get('p_win'), errors='coerce')
    d = pd.to_numeric(out.get('_payout_decimal'), errors='coerce')
    out['_ev_per_$1'] = p * d - (1.0 - p)
    return out

def ev_metrics(df_like: Dict[str, Any] | pd.DataFrame) -> Dict[str, Any]:
    """Compute robust EV metrics from dict-like or DataFrame row/columns."""
    get = df_like.get if hasattr(df_like, 'get') else lambda k, default=None: getattr(df_like, k, default)
    p = _to_series(get('p_win', np.nan))
    payout = _to_series(get('_payout_decimal', get('payout_decimal', get('payout', np.nan))))
    ev = _to_series(get('_ev_per_$1', get('ev', np.nan)))
    n = int(len(p))
    with_p = int(p.notna().sum())
    with_payout = int(payout.notna().sum())
    with_ev = int(ev.notna().sum())
    ev_mean = float(ev.mean(skipna=True)) if with_ev else float('nan')
    ev_median = float(ev.median(skipna=True)) if with_ev else float('nan')
    ev_pos = int((ev > 0).sum()) if with_ev else 0
    ev_nonneg = int((ev >= 0).sum()) if with_ev else 0
    ev_p95 = float(ev.quantile(0.95)) if with_ev else float('nan')
    ev_p05 = float(ev.quantile(0.05)) if with_ev else float('nan')
    return {'rows': n, 'with_p': with_p, 'with_payout': with_payout, 'with_ev': with_ev, 'ev_mean': ev_mean, 'ev_median': ev_median, 'ev_p95': ev_p95, 'ev_p05': ev_p05, 'ev_pos': ev_pos, 'ev_nonneg': ev_nonneg}

def summarize_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Group by a model column and compute EV metrics per group."""
    if by not in df.columns:
        return pd.DataFrame(columns=['model', 'rows', 'with_p', 'with_payout', 'with_ev', 'ev_mean', 'ev_median', 'ev_p95', 'ev_p05', 'ev_pos', 'ev_nonneg'])
    recs = []
    for model_name, g in df.groupby(by, dropna=False):
        met = ev_metrics(g)
        recs.append({'model': str(model_name), **met})
    out = pd.DataFrame(recs).sort_values(['ev_mean', 'with_ev', 'rows'], ascending=[False, False, False], na_position='last')
    return out
exp = _exports_dir()
st.caption(f'Build: `{BUILD_TAG}` · exports: `{exp}`')
default_candidates = [exp / 'edges_models.csv', exp / 'edges_graded_plus.csv', exp / 'edges_graded_full.csv', exp / 'predictions.csv']
path_found = next((p for p in default_candidates if p.exists()), None)
left, right = st.columns([2, 1])
with left:
    st.write('**Source**')
    src_choice = st.radio('Pick a source file', ['Auto-detected', 'Upload CSV'], captions=['Use first existing default', 'Upload a custom file'], label_visibility='collapsed', horizontal=True)
    uploaded = None
    if src_choice == 'Upload CSV':
        uploaded = st.file_uploader('Upload model predictions CSV', type=['csv'], accept_multiple_files=False)
    elif path_found:
        st.caption(f'Auto: `{path_found}`')
    else:
        st.warning('No default file found in exports/. Upload a CSV.')
with right:
    st.write('**Model column**')
    model_col = st.text_input('Model name column', value='_model', help='Name of the column that identifies the model for each row.')
    st.write('**Probability column**')
    prob_col = st.text_input('p_win column', value='p_win', help='Probability of the pick winning (0..1).')
    st.write('**Payout column**')
    payout_col = st.text_input('Decimal payout column', value='_payout_decimal', help='Decimal odds incl. stake; leave if will be derived.')
    st.write('**American odds column (fallback)**')
    american_col = st.text_input('American odds column', value='american_odds', help='If decimal payout missing, this will be converted.')
df: pd.DataFrame
try:
    if uploaded is not None:
        df = pd.read_csv(uploaded, low_memory=False)
    elif path_found:
        df = pd.read_csv(path_found, low_memory=False)
    else:
        df = pd.DataFrame()
except Exception as e:
    st.error(f'Failed to read CSV: {e}')
    st.stop()
if df.empty:
    st.info('No data loaded yet.')
    st.stop()
df = add_probs_and_ev(df)
if prob_col in df.columns and prob_col != 'p_win':
    df['p_win'] = pd.to_numeric(df[prob_col], errors='coerce').clip(0, 1)
if payout_col in df.columns and payout_col != '_payout_decimal':
    df['_payout_decimal'] = pd.to_numeric(df[payout_col], errors='coerce')
if american_col in df.columns and df['_payout_decimal'].isna().all():
    df['_payout_decimal'] = pd.to_numeric(df[american_col], errors='coerce').map(american_to_decimal)
if model_col not in df.columns:
    for alt in ['model', 'model_name', 'algo', '_model_name']:
        if alt in df.columns:
            model_col = alt
            break
    else:
        df[model_col] = 'unknown'
with st.expander('Filters', expanded=False):
    leagues = None
    if '_league' in df.columns:
        leagues = sorted([x for x in df['_league'].dropna().astype(str).unique()])
        chosen_leagues = st.multiselect('League(s)', leagues, default=leagues[:4] if leagues else None)
        if chosen_leagues:
            df = df[df['_league'].astype(str).isin(chosen_leagues)]
    min_p = st.slider('Min p_win', 0.0, 1.0, 0.0, 0.01)
    max_p = st.slider('Max p_win', 0.0, 1.0, 1.0, 0.01)
    df = df[df['p_win'].between(min_p, max_p, inclusive='both')]
    if '_ev_per_$1' in df.columns:
        min_ev = st.slider('Min EV per $1', -1.0, 2.0, -1.0, 0.01)
        df = df[df['_ev_per_$1'] >= min_ev]
overall = ev_metrics(df)
st.subheader('Overall')
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric('Rows', f"{overall['rows']:,}")
c2.metric('With p', f"{overall['with_p']:,}")
c3.metric('With payout', f"{overall['with_payout']:,}")
c4.metric('With EV', f"{overall['with_ev']:,}")
c5.metric('EV mean', f"{overall['ev_mean']:.4f}" if not math.isnan(overall['ev_mean']) else '—')
c6.metric('EV median', f"{overall['ev_median']:.4f}" if not math.isnan(overall['ev_median']) else '—')
c7.metric('EV p95', f"{overall['ev_p95']:.4f}" if not math.isnan(overall['ev_p95']) else '—')
st.subheader('By Model')
by_model = summarize_by(df, model_col)
st.dataframe(by_model, width='stretch')
with st.expander('Sample Rows (cleaned view)', expanded=False):
    cols_show = [c for c in ['_league', model_col, 'p_win', '_payout_decimal', '_ev_per_$1', 'american_odds'] if c in df.columns]
    st.dataframe(df[cols_show].head(100), width='stretch')
colA, colB = st.columns(2)
with colA:
    st.download_button('Download per-model summary (CSV)', by_model.to_csv(index=False).encode('utf-8'), file_name='compare_models_summary.csv', mime='text/csv', width='stretch')
with colB:
    st.download_button('Download filtered rows (CSV)', df.to_csv(index=False).encode('utf-8'), file_name='compare_models_filtered.csv', mime='text/csv', width='stretch')
st.caption(f'✔ Ready · {len(df):,} rows after filters · model column: `{model_col}`')
