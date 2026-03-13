from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    from lib.compliance_gate import require_eligibility
except Exception:
    def require_eligibility(*args, **kwargs):
        return True

try:
    from lib.access import require_allowed_page, beta_banner
except Exception:
    def require_allowed_page(*args, **kwargs):
        return None

    def beta_banner(*args, **kwargs):
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


st.set_page_config(page_title="Settle Parlay", page_icon="✅", layout="wide")
require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})
require_allowed_page("pages/08_Settle_Parlay.py")
beta_banner()

auth = login(required=False)
if not getattr(auth, "ok", True):
    st.stop()
if not getattr(auth, "authenticated", True):
    st.info("You are in read-only mode.")
show_logout()

diag = mount_in_sidebar("08_Settle_Parlay")

st.title("✅ Settle Parlay")
st.caption("Load edge rows and score rows, then settle legs and summarize results.")


def exports_dir() -> Path:
    env = os.environ.get("EDGE_EXPORTS_DIR", "").strip()
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p

    here = Path(__file__).resolve()
    for up in [here.parent] + list(here.parents):
        if up.name.lower() == "edge-finder":
            p = up / "exports"
            p.mkdir(parents=True, exist_ok=True)
            return p

    p = Path.cwd() / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_existing(paths: list[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_").lower() for c in out.columns]
    return out


def first_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def to_date_key(df: pd.DataFrame) -> pd.Series:
    for c in ["_date_iso", "_date", "date", "game_date", "commence_time", "datetime", "timestamp"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            if s.notna().any():
                return s.dt.strftime("%Y-%m-%d").astype("string")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")


_ALIAS = {
    "REDSKINS": "COMMANDERS",
    "WASHINGTON": "COMMANDERS",
    "FOOTBALL": "COMMANDERS",
    "OAKLAND": "RAIDERS",
    "LAS_VEGAS": "RAIDERS",
    "VEGAS": "RAIDERS",
    "LV": "RAIDERS",
    "ST_LOUIS": "RAMS",
    "SAN_DIEGO": "CHARGERS",
    "NINERS": "49ERS",
    "TB": "BUCCANEERS",
    "KC": "CHIEFS",
    "NE": "PATRIOTS",
    "NO": "SAINTS",
    "JAX": "JAGUARS",
    "NYJ": "JETS",
    "NYG": "GIANTS",
}


def nickify(s: pd.Series) -> pd.Series:
    x = s.astype("string").fillna("").str.upper()
    x = x.str.replace(r"[^A-Z0-9 ]+", "", regex=True).str.strip()
    x = x.str.replace(r"\s+", "_", regex=True)
    x = x.replace(_ALIAS)
    return x


def market_norm(s: pd.Series) -> pd.Series:
    x = s.astype("string").fillna("").str.upper()
    x = x.replace({"SPREAD": "SPREADS", "TOTAL": "TOTALS", "ML": "H2H", "MONEYLINE": "H2H"})
    return x


def side_norm(s: pd.Series, home: pd.Series, away: pd.Series) -> pd.Series:
    x = s.astype("string").fillna("").str.upper().str.strip()
    x = x.str.replace(r"\s+", "_", regex=True)

    home_n = nickify(home)
    away_n = nickify(away)
    side_n = nickify(x)

    x = x.mask(side_n.eq(home_n), "HOME")
    x = x.mask(side_n.eq(away_n), "AWAY")
    x = x.mask(x.str.contains("OVER", na=False), "OVER")
    x = x.mask(x.str.contains("UNDER", na=False), "UNDER")
    return x


def american_to_profit_per_1(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(o > 0, o / 100.0, np.where(o < 0, 100.0 / np.abs(o), np.nan)),
        index=odds.index,
        dtype="float64",
    )


@st.cache_data(ttl=60)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Optional[Path]]]:
    exp = exports_dir()

    edge_candidates = [
        exp / "edges_graded_full_normalized_std.csv",
        exp / "edges_graded_full.csv",
        exp / "edges_standardized.csv",
        exp / "edges_normalized.csv",
        exp / "edges_master.csv",
        exp / "edges.csv",
    ]
    score_candidates = [
        exp / "scores_normalized_std.csv",
        exp / "scores_clean_std.csv",
        exp / "scores_normalized.csv",
        exp / "scores_clean.csv",
        exp / "scores_1966-2025.csv",
        exp / "scores.csv",
    ]

    edges_path = latest_existing(edge_candidates)
    scores_path = latest_existing(score_candidates)

    edges = pd.read_csv(edges_path, low_memory=False, encoding="utf-8-sig") if edges_path else pd.DataFrame()
    scores = pd.read_csv(scores_path, low_memory=False, encoding="utf-8-sig") if scores_path else pd.DataFrame()

    return edges, scores, {"edges": edges_path, "scores": scores_path}


edges, scores, src = load_data()

if edges.empty:
    st.error("No edges file found in exports folder.")
    st.stop()

if scores.empty:
    st.error("No scores file found in exports folder.")
    st.stop()

st.caption(f"Edges source: `{src['edges']}`")
st.caption(f"Scores source: `{src['scores']}`")

edges = clean_cols(edges)
scores = clean_cols(scores)

# Score prep
scores["home_score"] = pd.to_numeric(first_series(scores, ["home_score"]), errors="coerce")
scores["away_score"] = pd.to_numeric(first_series(scores, ["away_score"]), errors="coerce")
scores["_date_key"] = to_date_key(scores)
scores["_home_key"] = nickify(first_series(scores, ["_home_nick", "home_team", "home"]))
scores["_away_key"] = nickify(first_series(scores, ["_away_nick", "away_team", "away"]))

scores_small = scores[["_date_key", "_home_key", "_away_key", "home_score", "away_score"]].dropna(
    subset=["_date_key", "_home_key", "_away_key"]
).copy()

# Edge prep
edges["_date_key"] = to_date_key(edges)
edges["_home_key"] = nickify(first_series(edges, ["_home_nick", "home_team", "home"]))
edges["_away_key"] = nickify(first_series(edges, ["_away_nick", "away_team", "away"]))
edges["_market_norm"] = market_norm(first_series(edges, ["_market_norm", "market"]))
edges["side_norm"] = side_norm(
    first_series(edges, ["side", "selection", "team"]),
    first_series(edges, ["_home_nick", "home_team", "home"]),
    first_series(edges, ["_away_nick", "away_team", "away"]),
)
edges["line"] = pd.to_numeric(first_series(edges, ["line", "spread", "total"]), errors="coerce")
edges["odds"] = pd.to_numeric(first_series(edges, ["odds", "price", "american_odds", "american"]), errors="coerce")

merged = edges.merge(
    scores_small,
    how="left",
    on=["_date_key", "_home_key", "_away_key"],
    suffixes=("", "_sc"),
)

hs = pd.to_numeric(merged["home_score"], errors="coerce")
as_ = pd.to_numeric(merged["away_score"], errors="coerce")
total_pts = hs + as_

result = pd.Series(["UNSETTLED"] * len(merged), index=merged.index, dtype="string")

# H2H
mask_h2h = merged["_market_norm"].eq("H2H") & hs.notna() & as_.notna()
result.loc[mask_h2h & merged["side_norm"].eq("HOME")] = np.where(hs[mask_h2h] > as_[mask_h2h], "WIN", np.where(hs[mask_h2h] < as_[mask_h2h], "LOSE", "PUSH"))
result.loc[mask_h2h & merged["side_norm"].eq("AWAY")] = np.where(as_[mask_h2h] > hs[mask_h2h], "WIN", np.where(as_[mask_h2h] < hs[mask_h2h], "LOSE", "PUSH"))

# SPREADS
mask_sp = merged["_market_norm"].eq("SPREADS") & hs.notna() & as_.notna() & merged["line"].notna()
spread_home = hs + merged["line"]
spread_away = as_ + merged["line"]

result.loc[mask_sp & merged["side_norm"].eq("HOME")] = np.where(
    spread_home[mask_sp & merged["side_norm"].eq("HOME")] > as_[mask_sp & merged["side_norm"].eq("HOME")],
    "WIN",
    np.where(
        spread_home[mask_sp & merged["side_norm"].eq("HOME")] < as_[mask_sp & merged["side_norm"].eq("HOME")],
        "LOSE",
        "PUSH",
    ),
)

result.loc[mask_sp & merged["side_norm"].eq("AWAY")] = np.where(
    spread_away[mask_sp & merged["side_norm"].eq("AWAY")] > hs[mask_sp & merged["side_norm"].eq("AWAY")],
    "WIN",
    np.where(
        spread_away[mask_sp & merged["side_norm"].eq("AWAY")] < hs[mask_sp & merged["side_norm"].eq("AWAY")],
        "LOSE",
        "PUSH",
    ),
)

# TOTALS
mask_tot = merged["_market_norm"].eq("TOTALS") & total_pts.notna() & merged["line"].notna()
result.loc[mask_tot & merged["side_norm"].eq("OVER")] = np.where(
    total_pts[mask_tot & merged["side_norm"].eq("OVER")] > merged.loc[mask_tot & merged["side_norm"].eq("OVER"), "line"],
    "WIN",
    np.where(
        total_pts[mask_tot & merged["side_norm"].eq("OVER")] < merged.loc[mask_tot & merged["side_norm"].eq("OVER"), "line"],
        "LOSE",
        "PUSH",
    ),
)
result.loc[mask_tot & merged["side_norm"].eq("UNDER")] = np.where(
    total_pts[mask_tot & merged["side_norm"].eq("UNDER")] < merged.loc[mask_tot & merged["side_norm"].eq("UNDER"), "line"],
    "WIN",
    np.where(
        total_pts[mask_tot & merged["side_norm"].eq("UNDER")] > merged.loc[mask_tot & merged["side_norm"].eq("UNDER"), "line"],
        "LOSE",
        "PUSH",
    ),
)

merged["_result"] = result
merged["_settled"] = merged["_result"].isin(["WIN", "LOSE", "PUSH"])
merged["_profit_per_1"] = american_to_profit_per_1(merged["odds"])

stake = st.number_input("Stake per leg ($)", min_value=1.0, value=100.0, step=5.0)

merged["_pnl"] = np.where(
    merged["_result"].eq("WIN"),
    merged["_profit_per_1"] * stake,
    np.where(
        merged["_result"].eq("LOSE"),
        -stake,
        np.where(merged["_result"].eq("PUSH"), 0.0, np.nan),
    ),
)

settled = merged[merged["_settled"]].copy()
unsettled = merged[~merged["_settled"]].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(merged):,}")
c2.metric("Settled", f"{len(settled):,}")
c3.metric("Unsettled", f"{len(unsettled):,}")
c4.metric("Net P&L", f"${settled['_pnl'].sum():,.2f}" if not settled.empty else "$0.00")

st.subheader("Settlement Summary")
summary = (
    settled["_result"].value_counts(dropna=False)
    .rename_axis("result")
    .reset_index(name="rows")
    if not settled.empty
    else pd.DataFrame({"result": [], "rows": []})
)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Settled Rows")
show_cols = [
    c for c in [
        "_date_key",
        "_home_key",
        "_away_key",
        "_market_norm",
        "side_norm",
        "line",
        "odds",
        "home_score",
        "away_score",
        "_result",
        "_pnl",
    ]
    if c in settled.columns
]
st.dataframe(settled[show_cols].reset_index(drop=True), use_container_width=True, height=420)

with st.expander("Unsettled rows", expanded=False):
    show_cols2 = [
        c for c in [
            "_date_key",
            "_home_key",
            "_away_key",
            "_market_norm",
            "side_norm",
            "line",
            "odds",
        ]
        if c in unsettled.columns
    ]
    st.dataframe(unsettled[show_cols2].reset_index(drop=True), use_container_width=True, height=320)

st.download_button(
    "Download settled_parlay_rows.csv",
    settled.to_csv(index=False).encode("utf-8"),
    file_name="settled_parlay_rows.csv",
    mime="text/csv",
)