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
            username = None
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
    def begin_session():
        return None

    def touch_session():
        return None

    def session_duration_str():
        return ""

    def bump_usage(*args, **kwargs):
        return None

    def show_nudge(*args, **kwargs):
        return None
# ---- /recovered app shims ----

from pathlib import Path
import fnmatch

import numpy as np
import pandas as pd
import streamlit as st


# ------------------ Data Loading ------------------
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent] + list(here.parents)
    for p in candidates:
        if (p / "streamlit_app.py").exists() or (p / ".devcontainer").exists() or (p / "exports").exists():
            return p
    return Path.cwd()



def _exports_dir() -> Path:
    try:
        env_val = st.secrets.get("EDGE_EXPORTS_DIR", "")
    except Exception:
        env_val = ""

    if str(env_val).strip():
        p = Path(str(env_val).strip())
        if p.exists() and p.is_dir():
            return p

    repo = _repo_root()
    repo_exports = repo / "exports"
    return repo_exports



def _candidate_csv_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    files: list[Path] = []
    try:
        for p in root.rglob("*.csv"):
            if p.is_file():
                files.append(p)
    except Exception:
        for p in root.glob("*.csv"):
            if p.is_file():
                files.append(p)
    return sorted(set(files))



def _pick_latest_file(root: Path, preferred_names: list[str], glob_patterns: list[str]) -> Path | None:
    all_csvs = _candidate_csv_files(root)
    if not all_csvs:
        return None

    matches: list[Path] = []
    preferred_lower = {x.lower() for x in preferred_names}

    for p in all_csvs:
        if p.name.lower() in preferred_lower:
            matches.append(p)

    for pattern in glob_patterns:
        pat = pattern.lower()
        for p in all_csvs:
            if fnmatch.fnmatch(p.name.lower(), pat):
                matches.append(p)

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
            "edges_market_anchor.csv",
            "edges_master.csv",
            "edges.csv",
        ],
        glob_patterns=[
            "*edges*standardized*.csv",
            "*edges*normalized*.csv",
            "*edges*graded*.csv",
            "*edges*market*anchor*.csv",
            "*edges*master*.csv",
            "*edges*.csv",
        ],
    )

    scores_path = _pick_latest_file(
        root,
        preferred_names=[
            "fort_knox_market_joined_moneyline_scored_all_seasons.csv",
            "fort_knox_market_joined_moneyline_scored.csv",
            "games_master_template.csv",
            "scores_1966-2025.csv",
            "scores_1966-2025_merged.csv",
            "scores_normalized_std.csv",
            "scores_normalized.csv",
            "scores.csv",
        ],
        glob_patterns=[
            "*fort*knox*market*joined*moneyline*scored*all*seasons*.csv",
            "*fort*knox*market*joined*moneyline*scored*.csv",
            "*games_master_template*.csv",
            "*scores_1966-2025*.csv",
            "*scores*normalized*.csv",
            "*scores*.csv",
            "*scored*.csv",
        ],
    )

    odds_path = _pick_latest_file(
        root,
        preferred_names=[
            "closing_lines.csv",
            "games_with_odds.csv",
            "odds_lines_all.csv",
            "odds.csv",
            "lines.csv",
        ],
        glob_patterns=[
            "*closing*lines*.csv",
            "*games_with_odds*.csv",
            "*odds*.csv",
            "*lines*.csv",
        ],
    )

    edges = _safe_read_csv(edges_path, "Edges") if edges_path else pd.DataFrame()
    scores = _safe_read_csv(scores_path, "Scores")
    odds = _safe_read_csv(odds_path, "Odds") if odds_path else pd.DataFrame()

    return edges, scores, odds


# ------------------ Page Config ------------------
st.set_page_config(page_title="Analytics Hub — Team & Market", layout="wide")
require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})


# ------------------ Auth / Session ------------------
auth = login(required=False)
if not getattr(auth, "ok", True):
    st.stop()

show_logout()

if "user" not in st.session_state:
    st.session_state["user"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "auth" not in st.session_state:
    st.session_state["auth"] = {}

auth_dict = st.session_state.get("auth", {}) or {}
auth_username = getattr(auth, "username", None) or auth_dict.get("username")

if auth_username and not st.session_state.get("user"):
    st.session_state["user"] = auth_username
if auth_username and not st.session_state.get("username"):
    st.session_state["username"] = auth_username
if auth_username and "username" not in auth_dict:
    auth_dict["username"] = auth_username
    st.session_state["auth"] = auth_dict


# === Nudge+Session ===
begin_session()
touch_session()
if hasattr(st, "sidebar"):
    st.sidebar.caption(f"🕒 Session: {session_duration_str()}")
bump_usage("page_visit")
show_nudge(
    feature="analytics",
    metric="page_visit",
    threshold=10,
    period="1D",
    demo_unlock=True,
    location="inline",
)
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
            "splits_home_away", "splits_fav_dog", "csv_export",
        },
        "market": {
            "matchup_ctx_basic", "market_ats_simple", "market_similar_spots",
            "distributions", "alerts",
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
        "home", "home_team", "team_home", "home_team_bet",
        "HomeTeam", "Home_Team", "homeName", "Home", "HOME",
        "host", "home_name", "team1_home", "homeTeam",
    )
    away = _pick_ci(
        out,
        "away", "away_team", "team_away", "away_team_bet",
        "AwayTeam", "Away_Team", "awayName", "Away", "AWAY",
        "visitor", "away_name", "team1_away", "awayTeam",
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
        _pick_ci(out, "home_score", "HomeScore", "Home_Score", "score_home", "final_home", "home_points", "home_pts"),
        errors="coerce",
    )
    out["away_score"] = pd.to_numeric(
        _pick_ci(out, "away_score", "AwayScore", "Away_Score", "score_away", "final_away", "away_points", "away_pts"),
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
            out["date"] = pd.to_datetime(out[lower_map[key]], errors="coerce").dt.strftime("%Y-%m-%d")
            return out

    out["date"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
    return out



def _ensure_numeric_column(df: pd.DataFrame, target: str, *alternates: str) -> pd.DataFrame:
    out = df.copy()
    if target in out.columns:
        out[target] = pd.to_numeric(out[target], errors="coerce")
        return out
    out[target] = pd.to_numeric(_pick_ci(out, *alternates), errors="coerce")
    return out



def _ensure_season_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Season" in out.columns:
        out["season"] = pd.to_numeric(out["Season"], errors="coerce")
    elif "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce")
    elif "year" in out.columns:
        out["season"] = pd.to_numeric(out["year"], errors="coerce")
    elif "date" in out.columns:
        out["season"] = pd.to_numeric(out["date"].astype("string").str[:4], errors="coerce")
    else:
        out["season"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="float")
    return out



def _season_values_from_scores(df: pd.DataFrame) -> list[int]:
    if df is None or df.empty:
        return []
    years = pd.to_numeric(df.get("season", df.get("date", pd.Series(dtype="string")).astype("string").str[:4]), errors="coerce")
    return sorted(years.dropna().astype(int).unique().tolist())


edges, scores, odds = load_data()

st.sidebar.markdown("### Data Sources")
st.sidebar.caption(f"Using exports dir: {_exports_dir()}")
files = [str(p.relative_to(_exports_dir())) for p in _candidate_csv_files(_exports_dir())[:100]]
if files:
    st.sidebar.write("Files discovered:")
    st.sidebar.code("\n".join(files[:25]))

if scores is None or scores.empty:
    st.error("No scores file could be loaded from exports/. Analytics Hub needs at least one scores CSV.")
    st.stop()

scores = _ensure_home_away_columns(scores)
scores = _ensure_score_columns(scores)
scores = _ensure_date_column(scores)
scores = _ensure_season_column(scores)
scores = _ensure_numeric_column(scores, "spread_home", "spread_home", "spread_close", "home_spread", "spread", "closing_spread_home")
scores = _ensure_numeric_column(scores, "total_close", "total_close", "closing_total", "total", "ou_line")

if {"home_score", "away_score"}.issubset(scores.columns):
    scores["margin_home"] = scores["home_score"] - scores["away_score"]
    scores["total_points"] = scores["home_score"] + scores["away_score"]


if {"margin_home", "spread_home"}.issubset(scores.columns):
    scores["home_cover"] = (scores["margin_home"] + scores["spread_home"]) > 0
    scores["away_cover"] = (scores["margin_home"] + scores["spread_home"]) < 0

if {"total_points", "total_close"}.issubset(scores.columns):
    scores["over_result"] = scores["total_points"] > scores["total_close"]
    scores["under_result"] = scores["total_points"] < scores["total_close"]


def normalize_scores(sc: pd.DataFrame) -> pd.DataFrame:
    sc = sc.copy()
    if "date" not in sc.columns and "_date_iso" in sc.columns:
        sc["date"] = pd.to_datetime(sc["_date_iso"], errors="coerce").dt.strftime("%Y-%m-%d")
    elif "date" in sc.columns:
        sc["date"] = pd.to_datetime(sc["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if {"home_score", "away_score"}.issubset(sc.columns):
        sc["margin_home"] = sc["home_score"] - sc["away_score"]
        sc["total_points"] = sc["home_score"] + sc["away_score"]
        sc["_margin_home"] = sc["margin_home"]
        sc["_total_points"] = sc["total_points"]

    if {"spread_home", "margin_home"}.issubset(sc.columns):
        sc["_home_spread_result"] = sc["margin_home"] + sc["spread_home"]
        sc["home_cover"] = np.where(sc["_home_spread_result"] > 0, 1, np.where(sc["_home_spread_result"] < 0, 0, 0.5))
        sc["_home_cover"] = sc["home_cover"]

    if {"total_close", "total_points"}.issubset(sc.columns):
        sc["over_result"] = np.where(sc["total_points"] > sc["total_close"], 1, np.where(sc["total_points"] < sc["total_close"], 0, 0.5))
        sc["_total_over"] = sc["over_result"]
    return sc


scores = normalize_scores(scores)
teams = sorted(set(scores["home"].dropna().astype(str)).union(set(scores["away"].dropna().astype(str))))
all_seasons = _season_values_from_scores(scores) or ["All"]


def team_profile(sc: pd.DataFrame, team: str, seasons: list[int] | None = None) -> dict[str, pd.DataFrame | pd.Series]:
    df = sc.copy()
    mask = df["home"].astype("string").fillna("").eq(team) | df["away"].astype("string").fillna("").eq(team)
    df = df.loc[mask].copy()
    if seasons and "All" not in seasons and "season" in df.columns:
        df = df[df["season"].isin(seasons)]
    df["_is_home"] = df["home"].astype("string").fillna("").eq(team).astype("int8")
    df["_pf"] = np.where(df["_is_home"] == 1, df["home_score"], df["away_score"])
    df["_pa"] = np.where(df["_is_home"] == 1, df["away_score"], df["home_score"])
    df["_mov"] = df["_pf"] - df["_pa"]
    df["_mol"] = np.where(df["_mov"] < 0, -df["_mov"], 0)
    if "spread_home" in df.columns:
        df["_team_spread"] = np.where(df["_is_home"] == 1, df["spread_home"], -df["spread_home"])
        team_result = df["_mov"] + df["_team_spread"]
        df["_ats_cover"] = np.where(team_result > 0, 1, np.where(team_result < 0, 0, 0.5))

    agg = pd.Series(dtype=float)
    agg.loc["games"] = len(df)
    for col, name_mean, name_std in [("_pf", "ppg_mean", "ppg_std"), ("_pa", "papg_mean", "papg_std"), ("_mov", "mov_mean", "mov_std"), ("_mol", "mol_mean", "mol_std")]:
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
            agg.loc["covered_win_pct"] = (covered & team_won).sum() / len(no_push)
            agg.loc["covered_loss_pct"] = (covered & lost).sum() / len(no_push)
    if "_total_over" in df.columns and len(df) > 0:
        no_push_tot = df[df["_total_over"] != 0.5]
        agg.loc["totals_games_no_push"] = len(no_push_tot)
        if len(no_push_tot):
            agg.loc["over_pct"] = (no_push_tot["_total_over"] == 1).mean()
            agg.loc["under_pct"] = (no_push_tot["_total_over"] == 0).mean()
    out: dict[str, pd.DataFrame | pd.Series] = {"summary": agg}
    split_tables = {}
    for label, filt in [("Home", df["_is_home"] == 1), ("Away", df["_is_home"] == 0)]:
        part = df.loc[filt]
        if len(part) == 0:
            continue
        row = {"games": len(part), "ppg_mean": part["_pf"].mean(), "ppg_std": part["_pf"].std(ddof=1), "papg_mean": part["_pa"].mean(), "papg_std": part["_pa"].std(ddof=1), "mov_mean": part["_mov"].mean(), "mov_std": part["_mov"].std(ddof=1)}
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
    if season is not None and season != "All" and "season" in df.columns:
        df = df[df["season"] == season]
    game = df[(df["home"] == home) & (df["away"] == away)]
    out = {"game_rows": len(game)}
    h2h = df[((df["home"] == home) & (df["away"] == away)) | ((df["home"] == away) & (df["away"] == home))]
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


def load_team_market_outperformance(min_games: int = 6) -> pd.DataFrame:
    root = _exports_dir()
    market_probs_path = _pick_latest_file(
        root,
        preferred_names=[
            "games_master_with_market_probs.csv",
            "fort_knox_market_joined_moneyline_scored_all_seasons.csv",
            "fort_knox_market_joined_moneyline_scored.csv",
            "fort_knox_market_joined_moneyline.csv",
        ],
        glob_patterns=[
            "*market*probs*.csv",
            "*fort*knox*market*joined*moneyline*scored*all*seasons*.csv",
            "*fort*knox*market*joined*moneyline*scored*.csv",
            "*market_joined*moneyline*.csv",
            "*games_master*market*.csv",
        ],
    )
    df = _safe_read_csv(market_probs_path, "Market probability")
    if df.empty:
        return pd.DataFrame()
    df = _ensure_home_away_columns(df)
    df = _ensure_score_columns(df)
    df = _ensure_date_column(df)
    df = _ensure_season_column(df)
    df["market_prob_home"] = pd.to_numeric(_pick_ci(df, "market_prob_home", "home_market_prob", "implied_prob_home", "home_implied_prob"), errors="coerce")
    df["market_prob_away"] = pd.to_numeric(_pick_ci(df, "market_prob_away", "away_market_prob", "implied_prob_away", "away_implied_prob"), errors="coerce")
    needed = ["home", "away", "home_score", "away_score", "market_prob_home", "market_prob_away"]
    df = df.dropna(subset=needed).copy()
    if df.empty:
        return pd.DataFrame()
    df["game_date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["expected_win_home"] = df["market_prob_home"]
    df["expected_win_away"] = df["market_prob_away"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["away_win"] = 1 - df["home_win"]
    games = pd.concat([
        pd.DataFrame({"team": df["home"], "expected": df["expected_win_home"], "actual": df["home_win"], "game_date": df["game_date"]}),
        pd.DataFrame({"team": df["away"], "expected": df["expected_win_away"], "actual": df["away_win"], "game_date": df["game_date"]}),
    ], ignore_index=True)
    games = games.dropna(subset=["team", "expected", "actual"]).copy()
    games["team"] = games["team"].astype("string").str.strip()
    games = games[games["team"].notna() & (games["team"] != "")].copy()
    games = games.sort_values(["team", "game_date"], na_position="first")
    total = games.groupby("team").agg(games=("actual", "count"), actual=("actual", "sum"), expected=("expected", "sum"))
    total["diff"] = total["actual"] - total["expected"]
    total["diff_per_game"] = total["diff"] / total["games"].replace(0, np.nan)
    total["signal_strength"] = total["diff_per_game"].abs() * np.sqrt(total["games"])
    last8 = games.groupby("team", group_keys=False).tail(8).groupby("team").agg(actual=("actual", "sum"), expected=("expected", "sum"), games_last8=("actual", "count"))
    last8["diff_last8"] = last8["actual"] - last8["expected"]
    last8["diff_per_game_last8"] = last8["diff_last8"] / last8["games_last8"].replace(0, np.nan)
    last4 = games.groupby("team", group_keys=False).tail(4).groupby("team").agg(actual=("actual", "sum"), expected=("expected", "sum"), games_last4=("actual", "count"))
    last4["diff_last4"] = last4["actual"] - last4["expected"]
    last4["diff_per_game_last4"] = last4["diff_last4"] / last4["games_last4"].replace(0, np.nan)
    teams_df = total.join(last8[["games_last8", "diff_last8", "diff_per_game_last8"]], how="left").join(last4[["games_last4", "diff_last4", "diff_per_game_last4"]], how="left")
    teams_df = teams_df[teams_df["games"] >= int(min_games)].copy()
    if teams_df.empty:
        return pd.DataFrame()
    teams_df["tier"] = "Average"
    teams_df.loc[teams_df["diff_per_game"] >= 0.30, "tier"] = "Elite Overperformer"
    teams_df.loc[(teams_df["diff_per_game"] >= 0.10) & (teams_df["diff_per_game"] < 0.30), "tier"] = "Slight Overperformer"
    teams_df.loc[(teams_df["diff_per_game"] <= -0.10) & (teams_df["diff_per_game"] > -0.30), "tier"] = "Slight Underperformer"
    teams_df.loc[teams_df["diff_per_game"] <= -0.30, "tier"] = "Poor Underperformer"
    teams_df["regression_trap"] = np.where((teams_df["diff"] > 0) & (teams_df["diff_last4"] < 0), "Yes", "No")
    teams_df = teams_df.reset_index().rename(columns={"index": "team"})
    return teams_df.sort_values(["diff_per_game", "diff"], ascending=False).reset_index(drop=True)


def _safe_div(a, b):
    try:
        if b in [0, 0.0] or pd.isna(b):
            return np.nan
        return a / b
    except Exception:
        return np.nan


def _american_to_decimal(odds):
    try:
        x = float(odds)
    except Exception:
        return np.nan
    if x == 0:
        return np.nan
    if x > 0:
        return 1.0 + (x / 100.0)
    return 1.0 + (100.0 / abs(x))


def _best_series(df: pd.DataFrame, *names: str, numeric: bool = False) -> pd.Series:
    s = _pick_ci(df, *names)
    return pd.to_numeric(s, errors="coerce") if numeric else s


@st.cache_data(show_spinner=False)
def load_model_validation_dataset() -> pd.DataFrame:
    root = _exports_dir()
    path = _pick_latest_file(
        root,
        preferred_names=[
            'fort_knox_market_joined_moneyline_all_seasons.csv',
            'fort_knox_market_joined_moneyline_scored_all_seasons.csv',
            'fort_knox_market_joined_moneyline.csv',
            'games_master_with_market_probs.csv',
            'games_master_with_team_market_features.csv',
            'edges_standardized.csv',
            'edges_graded_full_normalized_std.csv',
            'edges_graded_full.csv',
            'edges_normalized.csv',
            'edges.csv',
        ],
        glob_patterns=[
            '*fort*knox*market*joined*moneyline*all*seasons*.csv',
            '*fort*knox*market*joined*moneyline*scored*all*seasons*.csv',
            '*fort*knox*market*joined*.csv',
            '*games_master*market*prob*.csv',
            '*games_master*team_market_features*.csv',
            '*edges*standardized*.csv',
            '*edges*graded*.csv',
            '*edges*normalized*.csv',
            '*edges*.csv',
            '*backtest*bets*moneyline*.csv',
        ],
    )
    return _safe_read_csv(path, 'Model Validation') if path else pd.DataFrame()


def prepare_model_validation_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work['model_prob'] = _best_series(work, 'model_prob', 'edge_prob_governed', 'prob_model', 'pred_prob', numeric=True)
    work['bet_prob'] = _best_series(work, 'implied_prob', 'bet_prob', 'market_prob_at_bet', 'prob_at_bet', 'open_prob', numeric=True)
    work['open_prob'] = _best_series(work, 'open_prob', 'market_prob_open', 'prob_open', numeric=True)
    work['mid_prob'] = _best_series(work, 'mid_prob', 'market_prob_mid', 'prob_mid', numeric=True)
    work['close_prob'] = _best_series(work, 'close_prob', 'market_prob_close', 'closing_prob', 'prob_close', numeric=True)
    if work['bet_prob'].isna().all() and not work['open_prob'].isna().all():
        work['bet_prob'] = work['open_prob']
    work['edge_vs_bet'] = work['model_prob'] - work['bet_prob']
    work['edge_vs_open'] = _best_series(work, 'model_edge_vs_open', 'edge_vs_open', numeric=True)
    work['edge_vs_mid'] = _best_series(work, 'model_edge_vs_mid', 'edge_vs_mid', numeric=True)
    work['edge_vs_close'] = _best_series(work, 'model_edge_vs_close', 'edge_vs_close', numeric=True)
    if work['edge_vs_open'].isna().all() and not work['open_prob'].isna().all():
        work['edge_vs_open'] = work['model_prob'] - work['open_prob']
    if work['edge_vs_mid'].isna().all() and not work['mid_prob'].isna().all():
        work['edge_vs_mid'] = work['model_prob'] - work['mid_prob']
    if work['edge_vs_close'].isna().all() and not work['close_prob'].isna().all():
        work['edge_vs_close'] = work['model_prob'] - work['close_prob']
    work['clv'] = work['close_prob'] - work['bet_prob']
    work['beat_close'] = np.where(work['clv'].notna(), work['clv'] > 0, np.nan)
    work['profit'] = _best_series(work, 'profit', 'pnl', 'net_profit', numeric=True)
    work['stake'] = _best_series(work, 'stake', 'risk', 'risk_units', numeric=True)
    work.loc[work['stake'].isna() | (work['stake'] <= 0), 'stake'] = 1.0
    hit = _best_series(work, 'hit', 'actual_win', 'win', 'won', numeric=True)
    hit = hit.where(hit.isin([0, 1]), np.nan)
    work['hit'] = pd.to_numeric(hit, errors='coerce')
    odds_candidates = _best_series(work, 'true_moneyline', 'closing_odds', 'close_price', 'open_price', 'mid_price', 'odds', 'price', numeric=True)
    work['decimal_odds'] = odds_candidates.apply(_american_to_decimal).where(lambda s: s > 1, np.nan)
    work['expected_profit_per_unit'] = (work['model_prob'] * (work['decimal_odds'] - 1.0)) - (1.0 - work['model_prob'])
    work['realized_roi'] = work['profit'] / work['stake']
    work['expected_roi'] = work['expected_profit_per_unit'] / work['stake']
    move = work['close_prob'] - work['bet_prob']
    model_dir = np.sign(work['model_prob'] - work['bet_prob'])
    move_dir = np.sign(move)
    work['line_move_agreement'] = np.where(model_dir != 0, model_dir == move_dir, np.nan)
    work['prob_bucket'] = pd.cut(work['model_prob'], bins=[0.0, 0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True, right=False, labels=['0.00-0.50', '0.50-0.60', '0.60-0.70', '0.70-0.80', '0.80-1.00'])
    work['timestamp'] = pd.to_datetime(_best_series(work, 'timestamp', 'game_date', 'date', 'Date'), errors='coerce')
    return work


def model_validation_payload(df: pd.DataFrame) -> dict[str, object]:
    work = prepare_model_validation_df(df)
    if work.empty:
        return {'work': pd.DataFrame(), 'metrics': {}}
    settled = work[work['hit'].notna()].copy()
    if settled.empty:
        settled = work.copy()
    metrics = {
        'sample_size': int(len(work)),
        'settled_bets': int(len(settled)),
        'avg_clv': float(work['clv'].mean()) if work['clv'].notna().any() else np.nan,
        'beat_close_rate': float(pd.Series(work['beat_close']).dropna().astype(float).mean()) if pd.Series(work['beat_close']).notna().any() else np.nan,
        'avg_edge_vs_bet': float(work['edge_vs_bet'].mean()) if work['edge_vs_bet'].notna().any() else np.nan,
        'avg_edge_vs_open': float(work['edge_vs_open'].mean()) if work['edge_vs_open'].notna().any() else np.nan,
        'avg_edge_vs_mid': float(work['edge_vs_mid'].mean()) if work['edge_vs_mid'].notna().any() else np.nan,
        'avg_edge_vs_close': float(work['edge_vs_close'].mean()) if work['edge_vs_close'].notna().any() else np.nan,
        'actual_roi': float(_safe_div(settled['profit'].sum(), settled['stake'].sum())) if {'profit', 'stake'}.issubset(settled.columns) else np.nan,
        'expected_roi': float(settled['expected_roi'].mean()) if settled['expected_roi'].notna().any() else np.nan,
        'line_move_agreement': float(pd.Series(work['line_move_agreement']).dropna().astype(float).mean()) if pd.Series(work['line_move_agreement']).notna().any() else np.nan,
        'filter_kept_rate': float((work['edge_vs_bet'] > 0).mean()) if work['edge_vs_bet'].notna().any() else np.nan,
    }
    cal = settled[['model_prob', 'hit']].dropna().copy()
    if not cal.empty:
        cal['bucket'] = pd.cut(cal['model_prob'], bins=np.linspace(0, 1, 11), include_lowest=True)
        calibration = cal.groupby('bucket', observed=False).agg(predicted=('model_prob', 'mean'), actual=('hit', 'mean'), bets=('hit', 'size')).reset_index()
        calibration['bucket_label'] = calibration['bucket'].astype(str)
    else:
        calibration = pd.DataFrame()
    roi_bucket = settled.groupby('prob_bucket', observed=False).agg(bets=('hit', 'size'), win_rate=('hit', 'mean'), roi=('realized_roi', 'mean'), expected_roi=('expected_roi', 'mean'), avg_edge=('edge_vs_bet', 'mean')).reset_index()
    market_stage = pd.DataFrame({'stage': ['Open', 'Mid', 'Close'], 'edge': [metrics['avg_edge_vs_open'], metrics['avg_edge_vs_mid'], metrics['avg_edge_vs_close']]})
    filter_funnel = pd.DataFrame({'stage': ['Initial rows', 'Rows with model prob', 'Positive edge rows', 'Settled rows'], 'rows': [len(work), int(work['model_prob'].notna().sum()), int((work['edge_vs_bet'] > 0).sum()) if work['edge_vs_bet'].notna().any() else 0, len(settled)]})
    return {'work': work, 'settled': settled, 'metrics': metrics, 'calibration': calibration, 'roi_bucket': roi_bucket, 'market_stage': market_stage, 'filter_funnel': filter_funnel}


# ------------------ UI ------------------
st.title("📊 Analytics Hub")
st.caption("Dual-mode analytics with tiered feature gating (Basic / Advanced / Premium)")

username = str(getattr(auth, "username", None) or st.session_state.get("user") or st.session_state.get("username") or (st.session_state.get("auth", {}) or {}).get("username", "")).strip().lower()
tier = "premium"
if premium_enabled() or username.startswith("beta") or username in {"murphey", "rmcoy2"}:
    st.sidebar.success("Premium tier active")
else:
    st.sidebar.success("Premium tier active")

mode = st.radio("Mode", ["Team Profile", "Market / Matchup", "Model Validation"], horizontal=True)

if mode == "Team Profile":
    col_sel, col_season = st.columns([2, 1])
    team = col_sel.selectbox("Team", teams, index=0)
    default_seasons = all_seasons[-3:] if len(all_seasons) >= 3 and all_seasons != ["All"] else all_seasons
    sel_seasons = col_season.multiselect("Seasons", all_seasons, default=default_seasons)
    res = team_profile(scores, team, sel_seasons)
    st.subheader(f"Summary — {team}")
    summary = res.get("summary", pd.Series(dtype=float))
    cols_to_show = ["games", "ppg_mean", "papg_mean", "mov_mean", "ats_cover_pct", "over_pct", "under_pct", "ppg_std", "papg_std", "mov_std", "mol_mean", "mol_std", "covered_win_pct", "covered_loss_pct"]
    show = summary[summary.index.isin(cols_to_show)].rename({"games": "Games", "ppg_mean": "PPG (mean)", "papg_mean": "PA (mean)", "mov_mean": "MOV (mean)", "ats_cover_pct": "ATS Cover %", "over_pct": "Over %", "under_pct": "Under %", "ppg_std": "PPG (std)", "papg_std": "PA (std)", "mov_std": "MOV (std)", "mol_mean": "MOL (mean)", "mol_std": "MOL (std)", "covered_win_pct": "Covered Win %", "covered_loss_pct": "Covered Loss %"})
    st.dataframe(show.to_frame("Value"), use_container_width=True)
    if "splits_home_away" in res:
        st.markdown("### Home / Away Splits")
        st.dataframe(res["splits_home_away"], use_container_width=True)
    st.markdown("### Team Over/Underperformance vs Market")
    min_games_market = st.slider("Minimum games for market table", min_value=1, max_value=17, value=6, step=1)
    market_perf = load_team_market_outperformance(min_games=min_games_market)
    if market_perf.empty:
        st.info("No market probability file was found, or the file is missing required columns for team over/underperformance.")
    else:
        display_cols = [c for c in ["team", "games", "actual", "expected", "diff", "diff_per_game", "diff_last8", "diff_per_game_last8", "diff_last4", "diff_per_game_last4", "signal_strength", "tier", "regression_trap"] if c in market_perf.columns]
        team_market_perf = market_perf.loc[market_perf["team"] == team, display_cols].copy()
        if team_market_perf.empty:
            st.info(f"No market-based team outperformance rows were found for {team} using the current minimum-games filter.")
        else:
            st.dataframe(team_market_perf, use_container_width=True, hide_index=True)
        with st.expander("View full league table"):
            st.dataframe(market_perf[display_cols], use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download market outperformance CSV", market_perf.to_csv(index=False).encode("utf-8"), file_name="team_market_outperformance.csv", mime="text/csv")

elif mode == "Market / Matchup":
    homes = sorted(scores["home"].dropna().astype(str).unique().tolist())
    c1, c2, c3 = st.columns(3)
    home = c1.selectbox("Home", homes, index=0)
    away_options = sorted(scores.loc[scores["home"] == home, "away"].dropna().astype(str).unique().tolist()) or sorted(scores["away"].dropna().astype(str).unique().tolist())
    away = c2.selectbox("Away", away_options, index=0)
    season_choice = c3.selectbox("Season", ["All"] + all_seasons if all_seasons != ["All"] else ["All"], index=0)
    market = st.selectbox("Market", ["spread", "total"], index=0)
    line_val = st.number_input("Home spread (negative = home favorite)", value=-3.0, step=0.5) if market == "spread" else st.number_input("Total (closing/target)", value=45.5, step=0.5)
    ctx = market_context(scores, home, away, season=None if season_choice == "All" else int(season_choice), line=line_val, market=market)
    st.subheader("Matchup Context")
    st.write({"Rows (exact home/away matchup)": ctx.get("game_rows", 0), "Rows (all H2H in data)": ctx.get("h2h_games", 0)})
    if market == "spread":
        st.markdown("#### Similar Spots")
        st.write({"Similar-spot rows (±1.0 around line)": ctx.get("similar_spot_count"), "Home side cover % in similar spots (no pushes)": ctx.get("similar_cover_pct_home_side")})

else:
    st.subheader("Model Validation & Market-Aware Architecture")
    mv_raw = load_model_validation_dataset()
    payload = model_validation_payload(mv_raw)
    metrics = payload.get('metrics', {})
    work = payload.get('work', pd.DataFrame())
    calibration = payload.get('calibration', pd.DataFrame())
    roi_bucket = payload.get('roi_bucket', pd.DataFrame())
    market_stage = payload.get('market_stage', pd.DataFrame())
    filter_funnel = payload.get('filter_funnel', pd.DataFrame())
    with st.expander("Why this model framework is different", expanded=True):
        st.markdown("**The market is the prior. The model is the adjustment.**")
    if work.empty:
        st.info("No usable backtest or market-validation dataset was found in exports/. Add an edges/backtest CSV or a games master with model and market probabilities to unlock this section.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg CLV", f"{metrics.get('avg_clv', np.nan)*100:,.2f}%" if pd.notna(metrics.get('avg_clv', np.nan)) else "—")
        m2.metric("Beat Close Rate", f"{metrics.get('beat_close_rate', np.nan)*100:,.1f}%" if pd.notna(metrics.get('beat_close_rate', np.nan)) else "—")
        m3.metric("Avg Edge vs Close", f"{metrics.get('avg_edge_vs_close', np.nan)*100:,.2f}%" if pd.notna(metrics.get('avg_edge_vs_close', np.nan)) else "—")
        m4.metric("Sample Size", f"{metrics.get('sample_size', 0):,}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Expected ROI", f"{metrics.get('expected_roi', np.nan)*100:,.2f}%" if pd.notna(metrics.get('expected_roi', np.nan)) else "—")
        m6.metric("Actual ROI", f"{metrics.get('actual_roi', np.nan)*100:,.2f}%" if pd.notna(metrics.get('actual_roi', np.nan)) else "—")
        m7.metric("Line Move Agreement", f"{metrics.get('line_move_agreement', np.nan)*100:,.1f}%" if pd.notna(metrics.get('line_move_agreement', np.nan)) else "—")
        m8.metric("Positive-Edge Kept Rate", f"{metrics.get('filter_kept_rate', np.nan)*100:,.1f}%" if pd.notna(metrics.get('filter_kept_rate', np.nan)) else "—")
        tabs = st.tabs(["Calibration", "ROI Buckets", "Market Stages", "Filter Funnel", "Raw Validation Data"])
        with tabs[0]:
            if not calibration.empty:
                st.dataframe(calibration[[c for c in ['bucket_label', 'bets', 'predicted', 'actual'] if c in calibration.columns]], use_container_width=True, hide_index=True)
        with tabs[1]:
            if not roi_bucket.empty:
                st.dataframe(roi_bucket, use_container_width=True, hide_index=True)
        with tabs[2]:
            st.dataframe(market_stage, use_container_width=True, hide_index=True)
        with tabs[3]:
            st.dataframe(filter_funnel, use_container_width=True, hide_index=True)
        with tabs[4]:
            preview_cols = [c for c in ['timestamp', 'model_prob', 'bet_prob', 'open_prob', 'mid_prob', 'close_prob', 'edge_vs_bet', 'edge_vs_open', 'edge_vs_mid', 'edge_vs_close', 'clv', 'profit', 'realized_roi', 'expected_roi', 'line_move_agreement', 'hit'] if c in work.columns]
            st.dataframe(work[preview_cols].head(500), use_container_width=True, hide_index=True)
            st.download_button('⬇️ Download model validation CSV', work.to_csv(index=False).encode('utf-8'), file_name='model_validation_metrics.csv', mime='text/csv')
