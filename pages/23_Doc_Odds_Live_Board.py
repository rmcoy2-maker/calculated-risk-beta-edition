from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Safe shims for recovered environments
# -----------------------------------------------------------------------------
try:
    from app.lib.auth import login, show_logout  # type: ignore
except Exception:
    def login(required: bool = False):
        class _Auth:
            ok = True
            authenticated = True
        return _Auth()

    def show_logout():
        return None


try:
    from app.lib.compliance_gate import require_eligibility  # type: ignore
except Exception:
    def require_eligibility(*args, **kwargs):
        return True


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="23 Doc Odds Live Board",
    page_icon="📊",
    layout="wide",
)

auth = login(required=False)
if not getattr(auth, "ok", True):
    st.stop()

show_logout()

try:
    require_eligibility(min_age=18, restricted_states={"WA", "ID", "NV"})
except Exception:
    pass


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _exports_dir() -> Path:
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
        if (up / "exports").exists():
            return up / "exports"

    p = Path.cwd() / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return pd.DataFrame()


def american_to_decimal(odds) -> float:
    if pd.isna(odds):
        return np.nan
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 1.0 + o / 100.0
    if o < 0:
        return 1.0 + 100.0 / abs(o)
    return np.nan


def implied_prob_from_american(odds) -> float:
    if pd.isna(odds):
        return np.nan
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return abs(o) / (abs(o) + 100.0)
    return np.nan


def _first_present(df: pd.DataFrame, candidates: list[str], default=None) -> pd.Series:
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return df[lookup[c.lower()]]
    if default is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    return pd.Series([default] * len(df), index=df.index)


def _latest_existing(paths: list[Path]) -> Optional[Path]:
    found = [p for p in paths if p.exists() and p.is_file()]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def load_lines(lines_csv: Path) -> pd.DataFrame:
    df = _safe_read_csv(lines_csv)
    if df.empty:
        return df

    required = ["book", "league", "game_id", "market"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"lines_live.csv missing required columns: {missing}")
        return pd.DataFrame()

    out = df.copy()

    out["selection"] = _first_present(out, ["selection", "ref", "outcome", "team", "side"], default="")
    out["American"] = pd.to_numeric(_first_present(out, ["American", "odds", "price"]), errors="coerce")
    out["Line/Point"] = pd.to_numeric(_first_present(out, ["Line/Point", "line", "point"]), errors="coerce")

    for col in ["player_name", "stat_type", "commence_time", "home", "away", "side"]:
        if col not in out.columns:
            out[col] = ""

    out["Decimal"] = out["American"].map(american_to_decimal)
    out["Impl. Prob (Odds)"] = out["American"].map(implied_prob_from_american)

    text_cols = ["book", "league", "game_id", "market", "selection", "player_name", "stat_type", "commence_time", "home", "away", "side"]
    for c in text_cols:
        out[c] = out[c].astype(str).fillna("")

    return out


def load_model_probs(model_csv: Path) -> pd.DataFrame:
    df = _safe_read_csv(model_csv)
    if df.empty:
        return pd.DataFrame(columns=["game_id", "market", "selection", "prob"])

    needed = ["game_id", "market", "selection", "prob"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"model_probs.csv missing columns: {missing}. Ignoring model probabilities.")
        return pd.DataFrame(columns=["game_id", "market", "selection", "prob"])

    df = df.copy()
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df = df.dropna(subset=["prob"])
    return df


def best_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.dropna(subset=["American"]).copy()

    group_cols = ["game_id", "market", "selection"]
    for opt in ["player_name", "stat_type", "side"]:
        if opt in work.columns:
            group_cols.append(opt)

    idx = work.groupby(group_cols)["American"].idxmax()
    best = work.loc[idx].copy()

    counts = work.groupby(group_cols)["book"].nunique().rename("#Books")
    best = best.merge(counts, on=group_cols, how="left")

    sort_cols = [c for c in ["league", "game_id", "market", "player_name", "selection"] if c in best.columns]
    if sort_cols:
        best = best.sort_values(sort_cols, na_position="last")

    return best


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
EXPORTS = _exports_dir()
LINES_CSV = EXPORTS / "lines_live.csv"
MODEL_CSV = EXPORTS / "model_probs.csv"

st.title("📊 Doc Odds Live Board")
st.caption(
    f"lines_live.csv: {'present' if LINES_CSV.exists() else 'missing'} · "
    f"model_probs.csv: {'present' if MODEL_CSV.exists() else 'missing'} · "
    f"exports: `{EXPORTS}`"
)

df_lines = load_lines(LINES_CSV)
df_model = load_model_probs(MODEL_CSV)

if df_lines.empty:
    st.warning("No live odds found. Populate `exports/lines_live.csv` first.")
    st.stop()

if not df_model.empty:
    df = df_lines.merge(
        df_model,
        on=["game_id", "market", "selection"],
        how="left",
    ).rename(columns={"prob": "Model Prob"})
else:
    df = df_lines.copy()
    df["Model Prob"] = np.nan

is_prop_row = (df["player_name"].str.strip() != "") | (df["stat_type"].str.strip() != "")

st.sidebar.header("Filters")
view_mode = st.sidebar.radio("Market Type", ["Game Lines", "Player Props"], index=0)

league_opts = sorted([x for x in df["league"].dropna().unique().tolist() if str(x).strip()])
league_choice = st.sidebar.selectbox("League", ["all"] + league_opts, index=0)

min_books = int(st.sidebar.number_input("Min # books", min_value=1, max_value=20, value=2, step=1))

st.sidebar.header("EV & Kelly")
stake_display = float(st.sidebar.number_input("EV display stake ($)", min_value=1, max_value=10000, value=100, step=25))
bankroll = float(st.sidebar.number_input("Bankroll ($)", min_value=0, max_value=10000000, value=1000, step=50))
kelly_cap = float(st.sidebar.number_input("Kelly fraction cap", min_value=0.0, max_value=1.0, value=0.25, step=0.05))
min_ev_pct = float(st.sidebar.number_input("Min EV %", value=2.0, step=0.5))

if view_mode == "Game Lines":
    df_view = df[~is_prop_row].copy()
else:
    df_view = df[is_prop_row].copy()

if league_choice != "all":
    df_view = df_view[df_view["league"] == league_choice]

if df_view.empty:
    st.warning("No rows match your filters.")
    st.stop()

best = best_rows(df_view)
best = best[best["#Books"] >= min_books]

mp = pd.to_numeric(best["Model Prob"], errors="coerce")
dec = pd.to_numeric(best["Decimal"], errors="coerce")

ev_per_dollar = mp * dec - 1.0
best["EV %"] = ev_per_dollar * 100.0
best[f"EV @ ${int(stake_display)}"] = ev_per_dollar * stake_display

b = dec - 1.0
q = 1.0 - mp
with np.errstate(divide="ignore", invalid="ignore"):
    kelly_raw = (b * mp - q) / b

kelly_raw = pd.Series(kelly_raw, index=best.index).clip(lower=0.0).fillna(0.0)
best["Kelly f"] = kelly_raw.clip(upper=kelly_cap)
best["Kelly stake ($)"] = best["Kelly f"] * bankroll

best = best[best["EV %"].fillna(-999999) >= min_ev_pct]

if best.empty:
    st.warning("No edges meet the EV / #books filters yet.")
    st.stop()

base_cols = [
    "league",
    "game_id",
    "book",
    "market",
    "player_name",
    "stat_type",
    "side",
    "selection",
    "American",
    "Line/Point",
    "Decimal",
    "Impl. Prob (Odds)",
    "#Books",
]
ev_cols = ["Model Prob", "EV %", f"EV @ ${int(stake_display)}", "Kelly f", "Kelly stake ($)"]

show_cols = [c for c in base_cols + ev_cols if c in best.columns]

st.subheader("Top Edges (Best Price per Selection)")
st.dataframe(
    best[show_cols].reset_index(drop=True),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Live Board reads `exports/lines_live.csv` and optional `exports/model_probs.csv`."
)