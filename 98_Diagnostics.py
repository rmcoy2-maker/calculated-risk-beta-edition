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
st.set_page_config(page_title='98 Diagnostics', page_icon='📈', layout='wide')
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
import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import pandas as pd
st.title('🩺 Edge Finder — Unified Diagnostics')
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORTS = REPO_ROOT / 'exports'
LOG_DIR = EXPORTS / 'diag_logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

def load_logs() -> pd.DataFrame:
    rows = []
    for p in sorted(LOG_DIR.glob('diag-*.jsonl'))[-7:]:
        try:
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        rows.append(j)
                    except Exception:
                        pass
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['ts', 'page', 'kind', 'level', 'msg'])
logs = load_logs()
st.caption(f'Loaded {len(logs):,} log events from {LOG_DIR}')
st.subheader('Page Status')
if logs.empty:
    st.info('No diagnostics events recorded yet. Visit any data-handling page to generate logs.')
else:

    def _parse_ts(x):
        try:
            return datetime.fromisoformat(str(x))
        except Exception:
            return None
    logs['_dt'] = logs['ts'].map(_parse_ts)
    g = logs.groupby('page', dropna=True)
    summary = pd.DataFrame({'last_event_ts': g['_dt'].max(), 'events': g['page'].count(), 'warnings': g.apply(lambda df: int((df['level'] == 'warning').sum())), 'errors': g.apply(lambda df: int((df['level'] == 'error').sum()))}).reset_index().sort_values(['errors', 'warnings', 'last_event_ts'], ascending=[False, False, False])
    st.dataframe(summary, hide_index=True, width='stretch')
    st.subheader('Recent Events')
    pages = ['(all)'] + sorted(summary['page'].tolist())
    pick = st.selectbox('Filter by page', pages, index=0)
    view = logs if pick == '(all)' else logs[logs['page'] == pick]
    view = view.sort_values('_dt', ascending=False).head(500)[['ts', 'page', 'level', 'kind', 'msg']]
    st.dataframe(view, hide_index=True, width='stretch')
st.divider()
st.subheader('Exports Health')
files = {'bets_log.csv': EXPORTS / 'bets_log.csv', 'parlays.csv': EXPORTS / 'parlays.csv', 'edges.csv': EXPORTS / 'edges.csv', 'scores_1966-2025.csv': EXPORTS / 'scores_1966-2025.csv', 'settled.csv': EXPORTS / 'settled.csv'}
rows = []
for label, p in files.items():
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    row = {'file': label, 'exists': exists, 'size': size}
    if exists and size > 0:
        try:
            df = pd.read_csv(p, nrows=5)
            row['preview_rows'] = len(df)
            row['preview_cols'] = len(df.columns)
        except Exception as e:
            row['read_error'] = str(e)
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
