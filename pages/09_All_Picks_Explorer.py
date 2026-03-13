from __future__ import annotations

import os
import sys
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

_here = Path(__file__).resolve()


# =============================================================================
# Streamlit-safe bootstrap for recovered UI environments
# =============================================================================

THIS = Path(__file__).resolve()

# Try a few likely roots
CANDIDATE_ROOTS = [
    THIS.parents[3] if len(THIS.parents) >= 4 else None,  # project root
    THIS.parents[2] if len(THIS.parents) >= 3 else None,  # serving_ui_recovered
    Path.cwd(),
]

for p in CANDIDATE_ROOTS:
    if p and p.exists():
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

# Optional app dir registration
for p in CANDIDATE_ROOTS:
    if not p:
        continue
    for app_sub in ("serving_ui_recovered/app", "serving_ui/app", "app"):
        cand = p / app_sub
        if cand.exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)

# Optional no-op gates for recovered mode
try:
    from app.lib.compliance_gate import require_eligibility  # type: ignore
except Exception:
    def require_eligibility(*args, **kwargs):
        return None

try:
    from app.lib.auth import login, show_logout  # type: ignore
except Exception:
    def login(*args, **kwargs):
        class _Auth:
            ok = True
            authenticated = False
        return _Auth()

    def show_logout(*args, **kwargs):
        return None


# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Pick Explorer",
    page_icon="🎯",
    layout="wide",
)

# Safe auth/compliance calls
try:
    auth = login(required=False)
    if getattr(auth, "ok", True) is False:
        st.stop()
    show_logout()
except Exception:
    pass

try:
    require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})
except Exception:
    pass


# =============================================================================
# Utility helpers
# =============================================================================

TEAM_ALIAS = {
    "WSH": "COMMANDERS",
    "WASHINGTON": "COMMANDERS",
    "REDSKINS": "COMMANDERS",
    "FOOTBALL_TEAM": "COMMANDERS",
    "LV": "RAIDERS",
    "LAS_VEGAS": "RAIDERS",
    "OAKLAND": "RAIDERS",
    "SD": "CHARGERS",
    "ST_LOUIS": "RAMS",
    "STL": "RAMS",
    "SF": "49ERS",
    "NINERS": "49ERS",
    "NO": "SAINTS",
    "NE": "PATRIOTS",
    "GB": "PACKERS",
    "TB": "BUCCANEERS",
    "KC": "CHIEFS",
}


def _exports_dir() -> Path:
    """
    Resolve exports/ robustly for recovered local environments.
    """
    env = os.environ.get("EDGE_EXPORTS_DIR", "").strip()
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Search upward for likely project roots
    for up in [THIS.parent] + list(THIS.parents):
        if (up / "exports").exists():
            return up / "exports"
        if up.name.lower() in {"edge-finder", "calculated-risk"}:
            p = up / "exports"
            p.mkdir(parents=True, exist_ok=True)
            return p

    p = Path.cwd() / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _age_str(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        secs = int(time.time() - path.stat().st_mtime)
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m"
        if secs < 86400:
            return f"{secs // 3600}h"
        return f"{secs // 86400}d"
    except Exception:
        return "n/a"


def _latest_existing(paths: Iterable[Path]) -> Optional[Path]:
    found = [p for p in paths if p.exists() and p.is_file()]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def _glob_latest(base: Path, patterns: list[str]) -> Optional[Path]:
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(base.glob(pat))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _pick_latest_file(base: Path, preferred_names: list[str], glob_patterns: list[str]) -> Optional[Path]:
    explicit = _latest_existing([base / name for name in preferred_names])
    globbed = _glob_latest(base, glob_patterns)

    if explicit and globbed:
        return max([explicit, globbed], key=lambda p: p.stat().st_mtime)
    return explicit or globbed


def _safe_read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"Could not read `{path.name}`: {e}")
        return pd.DataFrame()


def _best_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df.empty:
        return None
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def _first_present(df: pd.DataFrame, candidates: list[str], default=None):
    col = _best_col(df, candidates)
    if col is None:
        if default is None:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_str(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
    )


def _nickify(series: pd.Series) -> pd.Series:
    s = (
        _clean_str(series)
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]+", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip("_")
    )
    return s.replace(TEAM_ALIAS)


def _normalize_market(value) -> str:
    s = str(value or "").strip().lower()
    if s in {"h2h", "ml", "moneyline", "money_line", "money line"}:
        return "MONEYLINE"
    if s in {"spread", "spreads", "ats"} or s.startswith("spread"):
        return "SPREAD"
    if s in {"total", "totals", "ou", "over_under"} or s.startswith("total"):
        return "TOTAL"
    if "player" in s and "prop" in s:
        return "PLAYER_PROP"
    return str(value or "").strip().upper() or "UNKNOWN"


def _american_to_decimal(series: pd.Series) -> pd.Series:
    s = _to_numeric(series)
    return pd.Series(
        np.where(
            s > 0, 1 + (s / 100.0),
            np.where(s < 0, 1 + (100.0 / np.abs(s)), np.nan)
        ),
        index=series.index,
    )


def _american_implied_prob(series: pd.Series) -> pd.Series:
    s = _to_numeric(series)
    return pd.Series(
        np.where(
            s > 0, 100.0 / (s + 100.0),
            np.where(s < 0, np.abs(s) / (np.abs(s) + 100.0), np.nan)
        ),
        index=series.index,
    )


def _coalesce_date(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "_date_iso", "date", "game_date", "event_date", "commence_time",
        "start_time", "scheduled", "_key_date", "Date",
    ]
    raw = _first_present(df, candidates)
    dt = pd.to_datetime(raw, errors="coerce", utc=True)
    try:
        return dt.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    except Exception:
        return dt.dt.strftime("%Y-%m-%d")


def _coalesce_datetime(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "commence_time", "start_time", "scheduled", "event_date",
        "game_date", "date", "Date",
    ]
    raw = _first_present(df, candidates)
    dt = pd.to_datetime(raw, errors="coerce", utc=True)
    try:
        return dt.dt.tz_convert("America/New_York")
    except Exception:
        return dt


def _attach_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    out["_date_iso"] = _coalesce_date(out)
    out["_event_dt"] = _coalesce_datetime(out)

    out["_home"] = _clean_str(_first_present(out, ["home", "home_team", "team_home", "Home", "HOME"]))
    out["_away"] = _clean_str(_first_present(out, ["away", "away_team", "team_away", "Away", "AWAY"]))
    out["_home_nick"] = _nickify(out["_home"])
    out["_away_nick"] = _nickify(out["_away"])

    out["_market_raw"] = _clean_str(_first_present(out, ["market", "market_type", "bet_type"], default=""))
    out["_market_norm"] = out["_market_raw"].map(_normalize_market)

    out["_book"] = _clean_str(_first_present(out, ["book", "sportsbook", "book_name"], default=""))
    out["_league"] = _clean_str(_first_present(out, ["league", "sport", "sport_key"], default=""))
    out["_side"] = _clean_str(_first_present(out, ["side", "selection", "pick_side", "team", "outcome"], default=""))
    out["_player"] = _clean_str(_first_present(out, ["player_name", "player", "athlete"], default=""))

    out["_line"] = _to_numeric(_first_present(out, ["line", "point", "Line/Point", "spread", "total"]))
    out["_odds_american"] = _to_numeric(
        _first_present(out, ["odds", "price", "american_odds", "American"])
    )
    out["_odds_decimal"] = _american_to_decimal(out["_odds_american"])
    out["_implied_prob"] = _american_implied_prob(out["_odds_american"])

    out["_model_prob"] = _to_numeric(
        _first_present(
            out,
            ["model_prob", "win_prob", "pred_prob", "probability", "fair_prob", "edge_prob"]
        )
    )

    out["_edge_pct"] = _to_numeric(
        _first_present(out, ["edge_pct", "edge", "ev_pct", "expected_edge"])
    )

    out["_result"] = _clean_str(
        _first_present(out, ["result", "grade", "outcome", "bet_result", "settlement"], default="")
    ).str.upper()

    out["_source_file"] = getattr(df, "_source_name", "unknown")

    return out


def _derive_edge_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    # infer model probability if missing but fair odds exist
    fair_american = _to_numeric(_first_present(out, ["fair_odds", "fair_price", "model_odds"]))
    fair_prob = _american_implied_prob(fair_american)

    if "_model_prob" not in out.columns:
        out["_model_prob"] = np.nan

    out["_model_prob"] = out["_model_prob"].fillna(fair_prob)

    # derive EV if possible
    dec = out["_odds_decimal"]
    mp = out["_model_prob"]

    out["_ev_per_1u"] = np.where(
        dec.notna() & mp.notna(),
        (mp * (dec - 1.0)) - (1.0 - mp),
        np.nan,
    )

    if out["_edge_pct"].isna().all():
        out["_edge_pct"] = np.where(
            mp.notna() & out["_implied_prob"].notna(),
            (mp - out["_implied_prob"]) * 100.0,
            np.nan,
        )

    return out


def _merge_scores(edges: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if edges.empty or scores.empty:
        return edges

    left = edges.copy()
    right = scores.copy()

    # common normalized keys
    for df in (left, right):
        if "_date_iso" not in df.columns:
            df["_date_iso"] = _coalesce_date(df)
        if "_home_nick" not in df.columns:
            df["_home_nick"] = _nickify(_first_present(df, ["home", "home_team", "team_home", "Home", "HOME"]))
        if "_away_nick" not in df.columns:
            df["_away_nick"] = _nickify(_first_present(df, ["away", "away_team", "team_away", "Away", "AWAY"]))

    score_cols = []
    for c in ["home_score", "away_score", "score_home", "score_away", "final_home", "final_away", "result", "grade"]:
        if c in right.columns:
            score_cols.append(c)

    if not score_cols:
        return left

    keep = ["_date_iso", "_home_nick", "_away_nick"] + score_cols
    right = right[keep].drop_duplicates()

    merged = left.merge(
        right,
        on=["_date_iso", "_home_nick", "_away_nick"],
        how="left",
        suffixes=("", "_score"),
    )
    return merged


def _settled_flag(series: pd.Series) -> pd.Series:
    s = _clean_str(series).str.upper()
    settled_values = {
        "WIN", "LOSS", "LOSE", "PUSH", "VOID", "HALF_WIN", "HALF_LOSS", "W", "L"
    }
    return s.isin(settled_values)


def _show_file_status(label: str, path: Optional[Path], df: pd.DataFrame):
    if path is None:
        st.caption(f"{label}: not found")
        return
    st.caption(f"{label}: `{path.name}` · rows={len(df):,} · age={_age_str(path)}")


# =============================================================================
# Data loading
# =============================================================================

EXPORTS = _exports_dir()

EDGES_PATH = _pick_latest_file(
    EXPORTS,
    preferred_names=[
        "edges_standardized.csv",
        "edges_graded_full_normalized_std.csv",
        "edges_graded_full.csv",
        "edges_normalized.csv",
        "edges_master.csv",
    ],
    glob_patterns=[
        "*edges*standardized*.csv",
        "*edges*normalized*.csv",
        "*edges*graded*.csv",
        "*edges*.csv",
    ],
)

SCORES_PATH = _pick_latest_file(
    EXPORTS,
    preferred_names=[
        "scores_normalized_std.csv",
        "scores_normalized.csv",
        "scores.csv",
    ],
    glob_patterns=[
        "*scores*normalized*.csv",
        "*scores*.csv",
    ],
)

PARLAY_PATH = _pick_latest_file(
    EXPORTS,
    preferred_names=[
        "parlay_scores.csv",
        "parlay_scores_latest.csv",
    ],
    glob_patterns=[
        "*parlay*score*.csv",
        "*parlay*.csv",
    ],
)


@st.cache_data(ttl=60)
def load_data(edges_path: Optional[Path], scores_path: Optional[Path], parlay_path: Optional[Path]):
    edges = _safe_read_csv(edges_path)
    scores = _safe_read_csv(scores_path)
    parlay = _safe_read_csv(parlay_path)

    edges._source_name = edges_path.name if edges_path else "edges_missing"
    scores._source_name = scores_path.name if scores_path else "scores_missing"
    parlay._source_name = parlay_path.name if parlay_path else "parlay_missing"

    edges = _attach_common_fields(edges)
    scores = _attach_common_fields(scores)
    parlay = _attach_common_fields(parlay)

    edges = _derive_edge_fields(edges)
    scores = _derive_edge_fields(scores)
    parlay = _derive_edge_fields(parlay)

    # Merge score context into edges if available
    edges = _merge_scores(edges, scores)

    return edges, scores, parlay


edges_df, scores_df, parlay_df = load_data(EDGES_PATH, SCORES_PATH, PARLAY_PATH)


# =============================================================================
# UI
# =============================================================================

st.title("🎯 Pick Explorer")
st.write("Browse model edges, inspect picks, and filter recovered exports safely.")

col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])

with col_a:
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

with col_b:
    dataset_choice = st.selectbox(
        "Dataset",
        ["Edges", "Scores", "Parlay Scores"],
        index=0,
    )

with col_c:
    st.metric("Exports dir", str(EXPORTS))

with col_d:
    total_rows = {
        "Edges": len(edges_df),
        "Scores": len(scores_df),
        "Parlay Scores": len(parlay_df),
    }[dataset_choice]
    st.metric("Rows", f"{total_rows:,}")

_show_file_status("Edges file", EDGES_PATH, edges_df)
_show_file_status("Scores file", SCORES_PATH, scores_df)
_show_file_status("Parlay file", PARLAY_PATH, parlay_df)

df = {
    "Edges": edges_df,
    "Scores": scores_df,
    "Parlay Scores": parlay_df,
}[dataset_choice].copy()

if df.empty:
    st.warning("No rows available. Put one or more supported CSVs into `exports/` and refresh.")
    st.stop()

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------

st.sidebar.header("Filters")

date_values = sorted([d for d in df["_date_iso"].dropna().unique().tolist() if str(d) != "NaT"])
market_values = sorted([m for m in df["_market_norm"].dropna().unique().tolist() if m])
book_values = sorted([b for b in df["_book"].dropna().unique().tolist() if b])
league_values = sorted([l for l in df["_league"].dropna().unique().tolist() if l])

default_markets = [m for m in ["MONEYLINE", "SPREAD", "TOTAL", "PLAYER_PROP"] if m in market_values]

selected_dates = st.sidebar.multiselect("Date", date_values)
selected_leagues = st.sidebar.multiselect("League", league_values)
selected_markets = st.sidebar.multiselect(
    "Market",
    market_values,
    default=default_markets if default_markets else None,
)
selected_books = st.sidebar.multiselect("Book", book_values)

team_search = st.sidebar.text_input("Team contains", "")
player_search = st.sidebar.text_input("Player contains", "")
side_search = st.sidebar.text_input("Selection / side contains", "")

min_edge = st.sidebar.number_input("Min edge %", value=0.0, step=0.5)
min_model_prob = st.sidebar.number_input("Min model prob", value=0.0, step=0.01, min_value=0.0, max_value=1.0)
min_ev = st.sidebar.number_input("Min EV / 1u", value=-999.0, step=0.01)

settled_mode = st.sidebar.selectbox("Settlement", ["All", "Settled only", "Unsettled only"])

q = df.copy()

if selected_dates:
    q = q[q["_date_iso"].isin(selected_dates)]
if selected_leagues:
    q = q[q["_league"].isin(selected_leagues)]
if selected_markets:
    q = q[q["_market_norm"].isin(selected_markets)]
if selected_books:
    q = q[q["_book"].isin(selected_books)]

if team_search.strip():
    needle = team_search.strip().upper()
    q = q[
        q["_home_nick"].str.contains(needle, na=False) |
        q["_away_nick"].str.contains(needle, na=False) |
        q["_home"].str.upper().str.contains(needle, na=False) |
        q["_away"].str.upper().str.contains(needle, na=False)
    ]

if player_search.strip():
    needle = player_search.strip().upper()
    q = q[q["_player"].str.upper().str.contains(needle, na=False)]

if side_search.strip():
    needle = side_search.strip().upper()
    q = q[q["_side"].str.upper().str.contains(needle, na=False)]

if "_edge_pct" in q.columns:
    q = q[q["_edge_pct"].fillna(-999999) >= min_edge]

if "_model_prob" in q.columns:
    q = q[q["_model_prob"].fillna(-1) >= min_model_prob]

if "_ev_per_1u" in q.columns:
    q = q[q["_ev_per_1u"].fillna(-999999) >= min_ev]

if settled_mode != "All":
    result_col = q["_result"] if "_result" in q.columns else pd.Series([""] * len(q), index=q.index)
    settled = _settled_flag(result_col)
    q = q[settled] if settled_mode == "Settled only" else q[~settled]

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("Filtered rows", f"{len(q):,}")

with k2:
    avg_edge = q["_edge_pct"].dropna().mean() if "_edge_pct" in q.columns else np.nan
    st.metric("Avg edge %", "—" if pd.isna(avg_edge) else f"{avg_edge:.2f}")

with k3:
    avg_prob = q["_model_prob"].dropna().mean() if "_model_prob" in q.columns else np.nan
    st.metric("Avg model prob", "—" if pd.isna(avg_prob) else f"{avg_prob:.3f}")

with k4:
    avg_ev = q["_ev_per_1u"].dropna().mean() if "_ev_per_1u" in q.columns else np.nan
    st.metric("Avg EV / 1u", "—" if pd.isna(avg_ev) else f"{avg_ev:.3f}")

with k5:
    settled_count = int(_settled_flag(q["_result"]).sum()) if "_result" in q.columns else 0
    st.metric("Settled rows", f"{settled_count:,}")

# -----------------------------------------------------------------------------
# Sort / column selection
# -----------------------------------------------------------------------------

st.subheader("Explorer")

preferred_cols = [
    "_date_iso",
    "_league",
    "_home",
    "_away",
    "_market_norm",
    "_player",
    "_side",
    "_line",
    "_odds_american",
    "_odds_decimal",
    "_implied_prob",
    "_model_prob",
    "_edge_pct",
    "_ev_per_1u",
    "_book",
    "_result",
]

available_preferred = [c for c in preferred_cols if c in q.columns]
remaining_cols = [c for c in q.columns if c not in available_preferred and not c.startswith("_event_dt")]
display_cols = available_preferred + remaining_cols

default_display = available_preferred[:]
chosen_cols = st.multiselect(
    "Columns",
    options=display_cols,
    default=default_display if default_display else display_cols[:15],
)

sort_options = [c for c in [
    "_date_iso", "_league", "_home", "_away", "_market_norm",
    "_edge_pct", "_model_prob", "_ev_per_1u", "_book"
] if c in q.columns]

sort_col = st.selectbox("Sort by", sort_options, index=sort_options.index("_edge_pct") if "_edge_pct" in sort_options else 0)
sort_desc = st.checkbox("Descending", value=True)

show_limit = st.slider("Max rows shown", min_value=50, max_value=5000, value=500, step=50)

view = q.copy()
if sort_col in view.columns:
    view = view.sort_values(sort_col, ascending=not sort_desc, na_position="last")

view = view.head(show_limit)

formatters = {}
for c in view.columns:
    if c in {"_model_prob", "_implied_prob"}:
        formatters[c] = lambda x: "" if pd.isna(x) else f"{x:.3f}"
    elif c in {"_edge_pct"}:
        formatters[c] = lambda x: "" if pd.isna(x) else f"{x:.2f}"
    elif c in {"_ev_per_1u"}:
        formatters[c] = lambda x: "" if pd.isna(x) else f"{x:.3f}"
    elif c in {"_line"}:
        formatters[c] = lambda x: "" if pd.isna(x) else f"{x:.1f}"
    elif c in {"_odds_decimal"}:
        formatters[c] = lambda x: "" if pd.isna(x) else f"{x:.3f}"
    elif c in {"_odds_american"}:
        formatters[c] = lambda x: "" if pd.isna(x) else f"{int(x):d}" if pd.notna(x) else ""

st.dataframe(
    view[chosen_cols].style.format(formatters),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# Detail inspector
# -----------------------------------------------------------------------------

st.subheader("Row Inspector")

if len(view) == 0:
    st.info("No rows match the current filters.")
else:
    inspector_idx = st.number_input(
        "Inspect row number from current table",
        min_value=0,
        max_value=max(len(view) - 1, 0),
        value=0,
        step=1,
    )
    record = view.iloc[int(inspector_idx)].to_dict()
    st.json(record)

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------

csv_bytes = view[chosen_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered CSV",
    data=csv_bytes,
    file_name="pick_explorer_filtered.csv",
    mime="text/csv",
    use_container_width=False,
)