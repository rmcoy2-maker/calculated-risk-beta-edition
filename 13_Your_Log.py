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
st.set_page_config(page_title='13 Calculated Log', page_icon='📈', layout='wide')
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
try:
    pass
except Exception:
    pass
st.markdown('\n<style>\n  .block-container { max-width: none !important; padding-left: 1rem; padding-right: 1rem; }\n  [data-testid="stHeader"] { z-index: 9990; }\n</style>\n', unsafe_allow_html=True)
try:
    from app.utils.newest_first_patch import apply_newest_first_patch as __nfp_apply
except Exception:
    try:
        from utils.newest_first_patch import apply_newest_first_patch as __nfp_apply
    except Exception:

        def __nfp_apply(_):
            return
__nfp_apply(st)
st.markdown('\n<style>\n  .block-container {max-width: 1600px; padding-top: 0.5rem; padding-left: 1.0rem; padding-right: 1.0rem;}\n</style>\n', unsafe_allow_html=True)
from app.bootstrap import bootstrap_paths
bootstrap_paths()
from pathlib import Path
import pandas as pd
calcS_CSV = Path('exports/calcs_log.csv')
PARLAYS_CSV = Path('exports/parlays.csv')

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if getattr(path, 'stat', None) and path.stat().st_size == 0:
            return pd.DataFrame()
    except Exception:
        pass
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, engine='python')
        except Exception:
            return pd.DataFrame()
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out['sort_ts'] = pd.to_datetime(out.get('ts'), errors='coerce')
    for c in ('stake', 'profit', 'result'):
        if c not in out.columns:
            out[c] = 0 if c != 'result' else ''
    return out

def build_calclog() -> pd.DataFrame:
    calcs = _normalize(_read_csv(calcS_CSV))
    parlays = _normalize(_read_csv(PARLAYS_CSV))
    if calcs.empty and parlays.empty:
        cols = ['ts', 'stake', 'profit', 'result', 'sort_ts', 'origin']
        return pd.DataFrame(columns=cols)
    if not calcs.empty:
        calcs['origin'] = 'calc'
    if not parlays.empty:
        parlays['origin'] = 'parlay'
    cols = sorted(set(calcs.columns) | set(parlays.columns))
    combined = pd.concat([calcs.reindex(columns=cols), parlays.reindex(columns=cols)], ignore_index=True)
    return combined.sort_values('sort_ts').reset_index(drop=True)

def summarize(calcs: pd.DataFrame) -> dict:
    if calcs.empty:
        return dict(calcs=0, wins=0, losses=0, pushes=0, roi=None)
    res = calcs['result'].astype(str).str.lower()
    wins = (res == 'win').sum()
    losses = (res == 'loss').sum()
    pushes = (res == 'push').sum()
    stake_sum = pd.to_numeric(calcs['stake'], errors='coerce').fillna(0).sum()
    profit_sum = pd.to_numeric(calcs['profit'], errors='coerce').fillna(0).sum()
    roi = profit_sum / stake_sum if stake_sum > 0 else None
    return dict(calcs=len(calcs), wins=wins, losses=losses, pushes=pushes, roi=roi)

def equity_series(calcs: pd.DataFrame, starting_bankroll: float=100.0) -> pd.DataFrame:
    p = pd.to_numeric(calcs.get('profit', 0), errors='coerce').fillna(0)
    equity = starting_bankroll + p.cumsum()
    out = pd.DataFrame({'ts': calcs['sort_ts'], 'equity': equity})
    out.set_index('ts', inplace=True)
    return out
calcs_df = build_calclog()
st.subheader('Calculated Log — Summary')
S = summarize(calcs_df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('calcs', S['calcs'])
c2.metric('Wins', S['wins'])
c3.metric('Losses', S['losses'])
c4.metric('Pushes', S['pushes'])
c5.metric('ROI', '-' if S['roi'] is None else f"{S['roi'] * 100:.1f}%")
st.subheader('Bankroll equity')
start_bankroll = st.number_input('Starting bankroll (units)', 0.0, 1000000000.0, 100.0, 1.0)
if calcs_df.empty:
    st.info('No calcs/parlays found (empty or missing CSVs).')
else:
    st.line_chart(equity_series(calcs_df, starting_bankroll=start_bankroll), y='equity')
st.subheader('Rows')
st.dataframe(calcs_df, hide_index=True, width='stretch')
