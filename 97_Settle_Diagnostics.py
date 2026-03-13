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
st.set_page_config(page_title='97 Settle Diagnostics', page_icon='📈', layout='wide')
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
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
EXPORTS = Path('exports')
EXPORTS.mkdir(exist_ok=True)
SCORES_PATH = EXPORTS / 'scores_1966-2025.csv'
EDGES_CANDIDATES = [EXPORTS / 'edges_graded.csv', EXPORTS / 'edges_enriched_norm.csv', EXPORTS / 'edges_graded_full.csv', EXPORTS / 'edges_normalized.csv', EXPORTS / 'edges_master.csv']

def _one_of(df: pd.DataFrame, names, default=None) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([default] * len(df))

def _norm_market(m: str) -> str:
    m = (str(m) if m is not None else '').strip().lower()
    if m in ('h2h', 'ml', 'moneyline', 'money line'):
        return 'H2H'
    if m.startswith('spread'):
        return 'SPREADS'
    if m.startswith('total'):
        return 'TOTALS'
    return m.upper()

def _normalize_team_text(s):
    if not isinstance(s, str) or not s.strip():
        return ''
    return s.strip().upper()

def _to_nick(s):
    s = _normalize_team_text(s)
    return s.split()[-1] if s else ''

def _parse_date_from_any(df: pd.DataFrame) -> pd.Series:
    for c in ['commence_time', 'CommenceTime', 'Date', '_date', 'game_date']:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors='coerce', utc=True).dt.tz_localize(None)
            if dt.notna().any():
                return dt.dt.date
    if 'game_id' in df.columns:
        m = df['game_id'].astype(str).str.extract('(\\d{4}-\\d{2}-\\d{2})', expand=False)
        dt = pd.to_datetime(m, errors='coerce')
        return dt.dt.date
    return pd.Series([pd.NaT] * len(df))

def _load_scores(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, low_memory=False)
    has_score = s[['HomeScore', 'AwayScore']].notna().all(axis=1) & ~(s['HomeScore'].fillna(0).eq(0) & s['AwayScore'].fillna(0).eq(0))
    s = s[has_score].copy()
    s['_date'] = pd.to_datetime(_one_of(s, ['_DateISO', 'Date']), errors='coerce').dt.date
    s['_home_nick'] = s['_HomeNick'].astype(str).str.upper()
    s['_away_nick'] = s['_AwayNick'].astype(str).str.upper()
    s['key_home'] = s['_home_nick'] + '@' + s['_away_nick'] + '@' + s['_date'].astype(str)
    s['key_away'] = s['_away_nick'] + '@' + s['_home_nick'] + '@' + s['_date'].astype(str)
    return s

def _pick_edges_file(candidates) -> Path | None:
    for p in candidates:
        try:
            if p.exists():
                tmp = pd.read_csv(p, nrows=5)
                if len(tmp) > 0:
                    return p
        except Exception:
            continue
    return None

def run_settle_diagnostics(scores_path: Path, edges_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores_played = _load_scores(scores_path)
    edges = pd.read_csv(edges_path, low_memory=False)
    home_cands = ['home', 'HomeTeam', '_home_team', '_home_abbr', '_home_abbr_norm', 'home_team']
    away_cands = ['away', 'AwayTeam', '_away_team', '_away_abbr', '_away_abbr_norm', 'away_team']
    side_cands = ['selection', 'Selection', 'side', 'team', '_pick_team']
    edges['_date'] = _parse_date_from_any(edges)
    edges['market_norm'] = [_norm_market(m) for m in _one_of(edges, ['market', 'Market']).tolist()]
    edges['_home_raw'] = _one_of(edges, home_cands, '')
    edges['_away_raw'] = _one_of(edges, away_cands, '')
    edges['_home_nick'] = edges['_home_raw'].map(_to_nick)
    edges['_away_nick'] = edges['_away_raw'].map(_to_nick)
    edges['_side_raw'] = _one_of(edges, side_cands, '').astype(str)
    edges['_side_nick'] = edges['_side_raw'].map(_to_nick)
    edges['key_home'] = edges['_home_nick'] + '@' + edges['_away_nick'] + '@' + edges['_date'].astype(str)
    edges['key_away'] = edges['_away_nick'] + '@' + edges['_home_nick'] + '@' + edges['_date'].astype(str)
    joined_home = edges.merge(scores_played[['key_home', 'key_away', 'HomeScore', 'AwayScore', 'HomeWin', 'AwayWin', 'Season', 'Week', '_date']], left_on='key_home', right_on='key_home', how='left', suffixes=('', '_sc'))
    needs_away = joined_home['HomeScore'].isna()
    joined = joined_home.copy()
    if needs_away.any():
        fill = edges[needs_away].merge(scores_played[['key_home', 'key_away', 'HomeScore', 'AwayScore', 'HomeWin', 'AwayWin', 'Season', 'Week', '_date']], left_on='key_away', right_on='key_away', how='left', suffixes=('', '_sc2'))
        for c in ['HomeScore', 'AwayScore', 'HomeWin', 'AwayWin', 'Season', 'Week', '_date']:
            joined.loc[needs_away, c] = fill[c].values
    diag = joined.copy()
    diag['has_date'] = diag['_date'].notna()
    diag['has_teams'] = diag['_home_nick'].ne('') & diag['_away_nick'].ne('')
    diag['matched_score'] = diag['HomeScore'].notna() & diag['AwayScore'].notna()

    def reason_row(r):
        if not r['has_date']:
            return 'no_date'
        if not r['has_teams']:
            return 'no_teams'
        if r['has_date'] and r['has_teams'] and (not r['matched_score']):
            return 'no_join_match'
        return ''
    diag['no_match_reason'] = diag.apply(reason_row, axis=1)
    total = len(diag)
    summary = pd.DataFrame({'metric': ['total_edges', 'with_dates', 'with_teams', 'matched_scores'], 'count': [total, int(diag['has_date'].sum()), int(diag['has_teams'].sum()), int(diag['matched_score'].sum())], 'percent': [1.0, diag['has_date'].mean() if total else 0.0, diag['has_teams'].mean() if total else 0.0, diag['matched_score'].mean() if total else 0.0]})
    unmatched = diag[~diag['matched_score']][['_date', '_home_raw', '_away_raw', '_home_nick', '_away_nick', 'market_norm', '_side_raw', 'key_home', 'key_away', 'no_match_reason']].head(50).copy()
    return (summary, unmatched)
with st.expander('🧪 Settle diagnostics', expanded=False):
    st.caption('Joins your edges to scores to show why some legs don’t settle (missing date, team mismatch, or no score).')
    auto_edges = _pick_edges_file(EDGES_CANDIDATES)
    chosen = st.selectbox('Edges file', options=[str(p) for p in EDGES_CANDIDATES if p.exists()], index=max(0, [i for i, p in enumerate(EDGES_CANDIDATES) if p == auto_edges][0]) if auto_edges else 0, help='Pick which edges file to diagnose.') if any((p.exists() for p in EDGES_CANDIDATES)) else None
    scores_in = st.text_input('Scores file', value=str(SCORES_PATH), help='Path to scores_1966-2025.csv')
    if st.button('Run settle diagnostics', type='primary', width='stretch'):
        try:
            edges_path = Path(chosen) if chosen else auto_edges
            scores_path = Path(scores_in)
            if not edges_path or not edges_path.exists():
                st.error('No usable edges file found. Check your exports folder.')
            elif not scores_path.exists():
                st.error(f'Scores file not found at: {scores_path}')
            else:
                summary, unmatched = run_settle_diagnostics(scores_path, edges_path)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_path = EXPORTS / f'settle_readiness_summary_{ts}.csv'
                unmatched_path = EXPORTS / f'settle_unmatched_examples_top50_{ts}.csv'
                summary.to_csv(summary_path, index=False)
                unmatched.to_csv(unmatched_path, index=False)
                st.success('Diagnostics complete.')
                st.write(f'**Summary saved:** `{summary_path}`')
                st.write(f'**Top 50 unmatched saved:** `{unmatched_path}`')
                st.subheader('Summary')
                st.dataframe(summary, width='stretch')
                st.subheader('Top 50 unmatched examples')
                st.dataframe(unmatched, width='stretch')
                tips = []
                if summary.loc[summary.metric == 'with_dates', 'percent'].item() < 0.95:
                    tips.append('Some edges are missing a reliable date. Ensure `commence_time` or `game_id` includes YYYY-MM-DD.')
                if summary.loc[summary.metric == 'with_teams', 'percent'].item() < 0.95:
                    tips.append('Team parsing failed for some edges. Normalize team fields and ensure they contain full names (nicknames parse from last word).')
                if summary.loc[summary.metric == 'matched_scores', 'percent'].item() < 0.8:
                    tips.append('Join rate is low. Verify scores file uses the same team vocabulary; consider mapping to consistent abbreviations.')
                if tips:
                    st.info('**Suggestions:**\n- ' + '\n- '.join(tips))
        except Exception as e:
            st.exception(e)
