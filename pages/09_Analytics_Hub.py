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

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ------------------ Page Config ------------------
st.set_page_config(page_title="Analytics Hub — Team & Market", layout="wide")
require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})


# === Nudge+Session ===
begin_session()
touch_session()
if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")
bump_usage("page_visit")
show_nudge(feature="analytics", metric="page_visit", threshold=10, period="1D", demo_unlock=True, location="inline")
# === /Nudge+Session ===


# ------------------ Entitlements ------------------
TIERS = ("basic", "advanced", "premium")

FEATURES_BY_TIER = {
    "basic": {
        "team": {"ppg_basic", "pa_basic"},
        "market": {"matchup_ctx_basic"},
        "shared": {"parlay_builder_lite"},
    },
    "advanced": {
        "team": {"ppg_basic", "pa_basic", "mov_basic", "ats_overall", "totals_over_under"},
        "market": {"matchup_ctx_basic", "market_ats_simple"},
        "shared": {"download_table"},
    },
    "premium": {
        "team": {
            "ppg_basic", "pa_basic", "mov_basic",
            "ppg_volatility", "pa_volatility", "mov_volatility",
            "mol_basic", "ats_overall", "ats_covered_win_loss",
            "splits_home_away", "splits_fav_dog", "csv_export"
        },
        "market": {
            "matchup_ctx_basic", "market_ats_simple", "market_similar_spots",
            "distributions", "alerts"
        },
        "shared": {"api_access", "download_table", "export_csv"},
    },
}


def has_feature(tier: str, scope: str, feat: str) -> bool:
    allowed = FEATURES_BY_TIER.get(tier, {})
    return feat in allowed.get(scope, set()) or feat in allowed.get("shared", set())


# ------------------ Helpers ------------------
def _pick_ci(df: pd.DataFrame, *options: str) -> pd.Series:
    if df is None or not hasattr(df, "columns"):
        return pd.Series(dtype="object")

    lower_map = {str(c).lower(): c for c in df.columns}
    for opt in options:
        if opt in df.columns:
            return df[opt]
        key = str(opt).lower()
        if key in lower_map:
            return df[lower_map[key]]

    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _split_matchup_column(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = series.astype("string").fillna("").str.strip()

    away = pd.Series([None] * len(s), index=s.index, dtype="object")
    home = pd.Series([None] * len(s), index=s.index, dtype="object")

    mask_at = s.str.contains(r"\s@\s", regex=True, na=False)
    if mask_at.any():
        parts = s[mask_at].str.split(r"\s@\s", n=1, expand=True)
        away.loc[mask_at] = parts[0]
        home.loc[mask_at] = parts[1]

    mask_vs = s.str.contains(r"\svs\.?\s", regex=True, na=False)
    if mask_vs.any():
        parts = s[mask_vs].str.split(r"\svs\.?\s", n=1, expand=True)
        away.loc[mask_vs] = parts[0]
        home.loc[mask_vs] = parts[1]

    return home, away


def _ensure_home_away_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not hasattr(df, "columns"):
        return df

    out = df.copy()

    home = _pick_ci(
        out,
        "home", "home_team", "team_home",
        "HomeTeam", "Home_Team",
        "homeName", "Home", "HOME",
        "host", "home_name", "team1_home", "homeTeam"
    )
    away = _pick_ci(
        out,
        "away", "away_team", "team_away",
        "AwayTeam", "Away_Team",
        "awayName", "Away", "AWAY",
        "visitor", "away_name", "team1_away", "awayTeam"
    )

    if home.isna().all() or (home.astype("string").fillna("").str.strip() == "").all():
        cand = _pick_ci(out, "team2", "Team2", "team_b", "opponent2")
        if not cand.isna().all():
            home = cand

    if away.isna().all() or (away.astype("string").fillna("").str.strip() == "").all():
        cand = _pick_ci(out, "team1", "Team1", "team_a", "opponent1")
        if not cand.isna().all():
            away = cand

    if home.isna().all() and away.isna().all():
        winner = _pick_ci(out, "winner", "winning_team")
        loser = _pick_ci(out, "loser", "losing_team")
        if not winner.isna().all() and not loser.isna().all():
            home = winner
            away = loser

    if home.isna().all() and away.isna().all():
        matchup = _pick_ci(out, "matchup", "game", "fixture", "teams", "event")
        if not matchup.isna().all():
            parsed_home, parsed_away = _split_matchup_column(matchup)
            home = parsed_home
            away = parsed_away

    out["home"] = home.astype("string").str.strip()
    out["away"] = away.astype("string").str.strip()

    out.loc[out["home"].isin(["", "nan", "None", "<NA>"]), "home"] = pd.NA
    out.loc[out["away"].isin(["", "nan", "None", "<NA>"]), "away"] = pd.NA

    return out


def _ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not hasattr(df, "columns"):
        return df

    out = df.copy()

    out["home_score"] = pd.to_numeric(
        _pick_ci(
            out,
            "home_score", "HomeScore", "Home_Score",
            "score_home", "final_home", "home_points", "home_pts"
        ),
        errors="coerce",
    )

    out["away_score"] = pd.to_numeric(
        _pick_ci(
            out,
            "away_score", "AwayScore", "Away_Score",
            "score_away", "final_away", "away_points", "away_pts"
        ),
        errors="coerce",
    )

    return out


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not hasattr(df, "columns"):
        return df

    out = df.copy()
    lower_map = {str(c).lower(): c for c in out.columns}

    for cand in ["_DateISO", "_dateiso", "game_date", "Date", "date"]:
        if cand in out.columns:
            out["date"] = pd.to_datetime(out[cand], errors="coerce").dt.strftime("%Y-%m-%d")
            return out
        key = cand.lower()
        if key in lower_map:
            src = lower_map[key]
            out["date"] = pd.to_datetime(out[src], errors="coerce").dt.strftime("%Y-%m-%d")
            return out

    out["date"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
    return out


def _ensure_numeric_column(df: pd.DataFrame, target: str, *alternates: str) -> pd.DataFrame:
    if df is None or not hasattr(df, "columns"):
        return df
    out = df.copy()
    if target in out.columns:
        out[target] = pd.to_numeric(out[target], errors="coerce")
        return out
    out[target] = pd.to_numeric(_pick_ci(out, *alternates), errors="coerce")
    return out


def _ensure_season_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not hasattr(df, "columns"):
        return df

    out = df.copy()

    if "Season" in out.columns:
        out["season"] = pd.to_numeric(out["Season"], errors="coerce")
        return out

    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce")
        return out

    if "year" in out.columns:
        out["season"] = pd.to_numeric(out["year"], errors="coerce")
        return out

    if "date" in out.columns:
        out["season"] = pd.to_numeric(out["date"].astype("string").str[:4], errors="coerce")
    else:
        out["season"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="float")

    return out


def _season_values_from_scores(df: pd.DataFrame) -> list[int]:
    if df is None or df.empty:
        return []

    if "season" in df.columns:
        years = pd.to_numeric(df["season"], errors="coerce")
    else:
        years = pd.to_numeric(df["date"].astype("string").str[:4], errors="coerce")

    return sorted(years.dropna().astype(int).unique().tolist())


# ------------------ Data Loading ------------------
def _exports_dir() -> Path:
    env_val = ""
    try:
        env_val = st.secrets.get("EDGE_EXPORTS_DIR", "")
    except Exception:
        env_val = ""

    if str(env_val).strip():
        p = Path(str(env_val).strip())
        p.mkdir(parents=True, exist_ok=True)
        return p

    here = Path(__file__).resolve()
    for up in [here.parent] + list(here.parents):
        if (up / "exports").exists():
            return up / "exports"
        if up.name.lower() == "edge-finder":
            p = up / "exports"
            p.mkdir(parents=True, exist_ok=True)
            return p

    p = here.parent / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pick_latest_file(root: Path, preferred_names: list[str], glob_patterns: list[str]) -> Path | None:
    matches: list[Path] = []

    for name in preferred_names:
        p = root / name
        if p.exists() and p.is_file():
            matches.append(p)

    for pattern in glob_patterns:
        matches.extend([p for p in root.glob(pattern) if p.is_file()])

    if not matches:
        return None

    uniq = {str(p.resolve()): p for p in matches}
    return max(uniq.values(), key=lambda p: p.stat().st_mtime)


def _safe_read_csv(path: Path | None, label: str) -> pd.DataFrame:
    if path is None:
        st.warning(f"{label} file not found in exports/.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        df.attrs["source_path"] = str(path)
        return df
    except Exception as e:
        st.warning(f"Could not load {label} file `{path.name}`: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_data():
    root = _exports_dir()

    edges_path = _pick_latest_file(
        root,
        preferred_names=[
            "edges_standardized.csv",
            "edges_graded_full_normalized_std.csv",
            "edges_graded_full.csv",
            "edges_normalized.csv",
            "edges.csv",
        ],
        glob_patterns=[
            "*edges*standardized*.csv",
            "*edges*normalized*.csv",
            "*edges*graded*.csv",
            "*edges*.csv",
        ],
    )

    scores_path = _pick_latest_file(
        root,
        preferred_names=[
            "games_master_template.csv",
            "scores_1966-2025.csv",
            "_backup_scores_1966-2025.csv",
            "scores_normalized_std.csv",
            "scores_normalized.csv",
            "scores.csv",
        ],
        glob_patterns=[
            "*games_master_template*.csv",
            "*scores_1966-2025*.csv",
            "*scores*normalized*.csv",
            "*scores*.csv",
        ],
    )

    odds_path = _pick_latest_file(
        root,
        preferred_names=[
            "games_with_odds.csv",
            "odds_lines_all.csv",
            "odds.csv",
            "lines.csv",
        ],
        glob_patterns=[
            "*games_with_odds*.csv",
            "*odds*.csv",
            "*lines*.csv",
        ],
    )

    edges = _safe_read_csv(edges_path, "Edges") if edges_path else pd.DataFrame()
    scores = _safe_read_csv(scores_path, "Scores")
    odds = _safe_read_csv(odds_path, "Odds") if odds_path else pd.DataFrame()

    return edges, scores, odds


edges, scores, odds = load_data()

# ---------------------------
# Derived game metrics
# ---------------------------

if "home_score" in scores.columns and "away_score" in scores.columns:

    scores["margin_home"] = scores["home_score"] - scores["away_score"]
    scores["total_points"] = scores["home_score"] + scores["away_score"]

if "spread_home" in scores.columns:
    scores["home_cover"] = (scores["margin_home"] + scores["spread_home"]) > 0

if "total_close" in scores.columns:
    scores["over_result"] = scores["total_points"] > scores["total_close"]
scores["away_cover"] = scores["margin_home"] + scores["spread_home"] < 0
scores["under_result"] = scores["total_points"] < scores["total_close"]
# ------------------ Scores normalization ------------------
edges, scores, odds = load_data()

# ------------------ Scores normalization ------------------
if scores is None or scores.empty:
    st.error("No scores file could be loaded from exports/. Analytics Hub needs at least one scores CSV.")
    st.stop()

scores = _ensure_home_away_columns(scores)
scores = _ensure_score_columns(scores)
scores = _ensure_date_column(scores)
scores = _ensure_season_column(scores)
scores = _ensure_numeric_column(
    scores,
    "spread_home",
    "spread_home",
    "spread_close",
    "home_spread",
    "spread",
    "closing_spread_home",
)
scores = _ensure_numeric_column(
    scores,
    "total_close",
    "total_close",
    "closing_total",
    "total",
    "ou_line",
)

teams_home = sorted(scores["home"].dropna().astype(str).unique().tolist()) if "home" in scores.columns else []
teams_away = sorted(scores["away"].dropna().astype(str).unique().tolist()) if "away" in scores.columns else []
teams = sorted(set(teams_home).union(set(teams_away)))

if not teams:
    st.error("Scores file loaded, but no usable home/away team columns were found.")
    st.write("Detected columns:", list(scores.columns))
    st.dataframe(scores.head(10), use_container_width=True)
    st.stop()

all_seasons = _season_values_from_scores(scores)
if not all_seasons:
    all_seasons = ["All"]


def normalize_scores(sc: pd.DataFrame) -> pd.DataFrame:
    sc = sc.copy()

    # normalize date again if alternate lowercase source exists
    if "date" not in sc.columns and "_date_iso" in sc.columns:
        sc["date"] = pd.to_datetime(sc["_date_iso"], errors="coerce").dt.strftime("%Y-%m-%d")
    elif "date" in sc.columns:
        sc["date"] = pd.to_datetime(sc["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # base score math
    if {"home_score", "away_score"}.issubset(sc.columns):
        sc["margin_home"] = sc["home_score"] - sc["away_score"]
        sc["total_points"] = sc["home_score"] + sc["away_score"]

        # keep old internal names too, so existing downstream code still works
        sc["_margin_home"] = sc["margin_home"]
        sc["_total_points"] = sc["total_points"]

    # ATS / spread math
    if {"spread_home", "margin_home"}.issubset(sc.columns):
        sc["_home_spread_result"] = sc["margin_home"] + sc["spread_home"]

        sc["home_cover"] = np.where(
            sc["_home_spread_result"] > 0, 1,
            np.where(sc["_home_spread_result"] < 0, 0, 0.5)
        )

        # preserve old name for compatibility
        sc["_home_cover"] = sc["home_cover"]

    # Totals math
    if {"total_close", "total_points"}.issubset(sc.columns):
        sc["over_result"] = np.where(
            sc["total_points"] > sc["total_close"], 1,
            np.where(sc["total_points"] < sc["total_close"], 0, 0.5)
        )

        # preserve old name for compatibility
        sc["_total_over"] = sc["over_result"]

    return sc


scores = normalize_scores(scores)
st.write(scores[["spread_home","total_close"]].dropna().head())

# ------------------ Computation Blocks ------------------
def team_profile(sc: pd.DataFrame, team: str, seasons: list[int] | None = None) -> dict[str, pd.DataFrame | pd.Series]:
    df = sc.copy()

    df["home"] = df["home"].astype("string")
    df["away"] = df["away"].astype("string")

    mask = df["home"].fillna("").eq(team) | df["away"].fillna("").eq(team)
    df = df.loc[mask].copy()

    if "season" not in df.columns:
        df = _ensure_season_column(df)
    if seasons and "All" not in seasons and "season" in df.columns:
        df = df[df["season"].isin(seasons)]

    df["_is_home"] = df["home"].fillna("").eq(team).astype("int8")
    df["_pf"] = np.where(df["_is_home"] == 1, df["home_score"], df["away_score"])
    df["_pa"] = np.where(df["_is_home"] == 1, df["away_score"], df["home_score"])
    df["_mov"] = df["_pf"] - df["_pa"]
    df["_mol"] = np.where(df["_mov"] < 0, -df["_mov"], 0)

    if "spread_home" in df.columns:
        df["_team_spread"] = np.where(df["_is_home"] == 1, df["spread_home"], -df["spread_home"])
        team_result = df["_mov"] + df["_team_spread"]
        df["_ats_cover"] = np.where(team_result > 0, 1, np.where(team_result < 0, 0, 0.5))

    out: dict[str, pd.DataFrame | pd.Series] = {}

    agg = pd.Series(dtype=float)
    agg.loc["games"] = len(df)

    for col, name_mean, name_std in [
        ("_pf", "ppg_mean", "ppg_std"),
        ("_pa", "papg_mean", "papg_std"),
        ("_mov", "mov_mean", "mov_std"),
        ("_mol", "mol_mean", "mol_std"),
    ]:
        if col in df.columns and len(df) > 0:
            agg.loc[name_mean] = df[col].mean()
            agg.loc[name_std] = df[col].std(ddof=1)

    if "_ats_cover" in df.columns and len(df) > 0:
        no_push = df[df["_ats_cover"] != 0.5]
        agg.loc["ats_games_no_push"] = len(no_push)
        if len(no_push):
            agg.loc["ats_cover_pct"] = no_push["_ats_cover"].mean()
            team_won = df["_mov"] > 0
            covered = df["_ats_cover"] == 1
            lost = df["_mov"] < 0
            covered_win = (covered & team_won).sum()
            covered_loss = (covered & lost).sum()
            agg.loc["covered_win_pct"] = covered_win / len(no_push)
            agg.loc["covered_loss_pct"] = covered_loss / len(no_push)

    if "_total_over" in df.columns and len(df) > 0:
        no_push_tot = df[df["_total_over"] != 0.5]
        agg.loc["totals_games_no_push"] = len(no_push_tot)
        if len(no_push_tot):
            agg.loc["over_pct"] = (no_push_tot["_total_over"] == 1).mean()
            agg.loc["under_pct"] = (no_push_tot["_total_over"] == 0).mean()

    out["summary"] = agg

    split_tables = {}
    for label, filt in [
        ("Home", df["_is_home"] == 1),
        ("Away", df["_is_home"] == 0),
    ]:
        part = df.loc[filt]
        if len(part) == 0:
            continue

        row = {
            "games": len(part),
            "ppg_mean": part["_pf"].mean(),
            "ppg_std": part["_pf"].std(ddof=1),
            "papg_mean": part["_pa"].mean(),
            "papg_std": part["_pa"].std(ddof=1),
            "mov_mean": part["_mov"].mean(),
            "mov_std": part["_mov"].std(ddof=1),
        }

        if "_ats_cover" in part.columns:
            pnp = part[part["_ats_cover"] != 0.5]
            row["ats_games_no_push"] = len(pnp)
            row["ats_cover_pct"] = pnp["_ats_cover"].mean() if len(pnp) else np.nan

        split_tables[label] = row

    if split_tables:
        out["splits_home_away"] = pd.DataFrame(split_tables).T

    return out


def market_context(sc: pd.DataFrame, home: str, away: str, season: int | None, line: float | None, market: str = "spread") -> dict:
    df = sc.copy()

    if "season" not in df.columns:
        df = _ensure_season_column(df)

    if season is not None and season != "All" and "season" in df.columns:
        df = df[df["season"] == season]

    game = df[(df["home"] == home) & (df["away"] == away)]
    out = {"game_rows": len(game)}

    h2h = df[
        ((df["home"] == home) & (df["away"] == away)) |
        ((df["home"] == away) & (df["away"] == home))
    ]
    out["h2h_games"] = len(h2h)

    if market == "spread" and "spread_home" in df.columns and line is not None:
        cand = df[df["home"] == home].copy()
        cand = cand[np.isfinite(cand["spread_home"])]
        similar = cand[cand["spread_home"].between(line - 1.0, line + 1.0)]
        out["similar_spot_count"] = len(similar)

        if len(similar):
            spread_result = similar["margin_home"] + similar["spread_home"]
            no_push = spread_result[spread_result != 0]
            out["similar_cover_pct_home_side"] = (no_push > 0).mean() if len(no_push) else np.nan

    return out

# ------------------ UI ------------------
st.title("📊 Analytics Hub")
st.caption("Dual-mode analytics with tiered feature gating (Basic / Advanced / Premium)")

# TEMP entitlement control (replace with your login/role)
username = str(
    st.session_state.get("user")
    or st.session_state.get("username")
    or (st.session_state.get("auth", {}) or {}).get("username", "")
).strip().lower()

if username.startswith("beta") or username in {"murphey", "rmcoy2"}:
    tier = "premium"
    st.sidebar.success("Premium tier active")
else:
    tier = st.sidebar.selectbox("Tier", TIERS, index=1)
mode = st.radio("Mode", ["Team Profile", "Market / Matchup"], horizontal=True)

if mode == "Team Profile":
    col_sel, col_season = st.columns([2, 1])
    team = col_sel.selectbox("Team", teams, index=0)

    default_seasons = all_seasons[-3:] if len(all_seasons) >= 3 and all_seasons != ["All"] else all_seasons
    sel_seasons = col_season.multiselect("Seasons", all_seasons, default=default_seasons)

    res = team_profile(scores, team, sel_seasons)

    st.subheader(f"Summary — {team}")
    summary = res.get("summary", pd.Series(dtype=float))

    basic_cols = []
    if has_feature(tier, "team", "ppg_basic"):
        basic_cols += ["ppg_mean"]
    if has_feature(tier, "team", "pa_basic"):
        basic_cols += ["papg_mean"]
    if has_feature(tier, "team", "mov_basic"):
        basic_cols += ["mov_mean"]

    adv_cols = []
    if has_feature(tier, "team", "ats_overall"):
        adv_cols += ["ats_cover_pct"]
    if has_feature(tier, "team", "totals_over_under"):
        adv_cols += ["over_pct", "under_pct"]

    prem_cols = []
    if has_feature(tier, "team", "ppg_volatility"):
        prem_cols += ["ppg_std"]
    if has_feature(tier, "team", "pa_volatility"):
        prem_cols += ["papg_std"]
    if has_feature(tier, "team", "mov_volatility"):
        prem_cols += ["mov_std"]
    if has_feature(tier, "team", "mol_basic"):
        prem_cols += ["mol_mean", "mol_std"]
    if has_feature(tier, "team", "ats_covered_win_loss"):
        prem_cols += ["covered_win_pct", "covered_loss_pct"]

    cols_to_show = ["games"] + basic_cols + adv_cols + prem_cols

    show = summary[summary.index.isin(cols_to_show)].rename({
        "games": "Games",
        "ppg_mean": "PPG (mean)",
        "ppg_std": "PPG (std)",
        "papg_mean": "PA (mean)",
        "papg_std": "PA (std)",
        "mov_mean": "MOV (mean)",
        "mov_std": "MOV (std)",
        "mol_mean": "MOL (mean)",
        "mol_std": "MOL (std)",
        "ats_cover_pct": "ATS Cover %",
        "covered_win_pct": "Covered Win %",
        "covered_loss_pct": "Covered Loss %",
        "over_pct": "Over %",
        "under_pct": "Under %",
    })

    st.dataframe(show.to_frame("Value"), use_container_width=True)

    if has_feature(tier, "team", "splits_home_away") and "splits_home_away" in res:
        st.markdown("### Home / Away Splits (Premium)")
        st.dataframe(res["splits_home_away"], use_container_width=True)

    if has_feature(tier, "team", "csv_export"):
        csv = show.to_csv(index=True).encode("utf-8")
        st.download_button(
            "⬇️ Download summary CSV",
            csv,
            file_name=f"{team}_summary.csv",
            mime="text/csv",
        )

else:
    homes = sorted(scores["home"].dropna().astype(str).unique().tolist())
    c1, c2, c3 = st.columns(3)

    home = c1.selectbox("Home", homes, index=0)

    away_options = sorted(
        scores.loc[scores["home"] == home, "away"].dropna().astype(str).unique().tolist()
    )
    if not away_options:
        away_options = sorted(scores["away"].dropna().astype(str).unique().tolist())

    away = c2.selectbox("Away", away_options, index=0)
    season_choice = c3.selectbox("Season", ["All"] + all_seasons if all_seasons != ["All"] else ["All"], index=0)

    market = st.selectbox("Market", ["spread", "total"], index=0)

    if market == "spread":
        line_val = st.number_input("Home spread (negative = home favorite)", value=-3.0, step=0.5)
    else:
        line_val = st.number_input("Total (closing/target)", value=45.5, step=0.5)

    ctx = market_context(
        scores,
        home,
        away,
        season=None if season_choice == "All" else int(season_choice),
        line=line_val,
        market=market,
    )

    st.subheader("Matchup Context")
    st.write({
        "Rows (exact home/away matchup)": ctx.get("game_rows", 0),
        "Rows (all H2H in data)": ctx.get("h2h_games", 0),
    })

    if has_feature(tier, "market", "market_ats_simple") and market == "spread":
        st.markdown("#### Similar Spots")
        st.write({
            "Similar-spot rows (±1.0 around line)": ctx.get("similar_spot_count"),
            "Home side cover % in similar spots (no pushes)": ctx.get("similar_cover_pct_home_side"),
        })

    if has_feature(tier, "market", "distributions"):
        st.info("Premium: distribution plots of margins/total results would render here.")

    if has_feature(tier, "market", "alerts"):
        st.info("Premium: line-move / arb / middle alert configuration would render here.")