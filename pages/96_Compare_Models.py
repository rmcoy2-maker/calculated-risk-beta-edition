from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="96 Compare Models", page_icon="📈", layout="wide")

# -----------------------------
# Login guard
# -----------------------------
if not st.session_state.get("authenticated", False):
    st.switch_page("00_Home.py")
    st.stop()

st.sidebar.success(f"Logged in as {st.session_state.get('user', '')}")

st.title("📊 Compare Models")
BUILD_TAG = "compare-models-cloud-fix-v1"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def exports_dir() -> Path:
    exp = project_root() / "exports"
    return exp


def to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series([], dtype="float64")
        return pd.to_numeric(x.iloc[:, 0], errors="coerce")
    if np.isscalar(x) or x is None:
        return pd.Series([x], dtype="float64")
    return pd.to_numeric(pd.Series(x), errors="coerce")


def american_to_decimal(odds: Any) -> float | np.nan:
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
    out = df.copy()

    p_candidates = ["p_win", "prob", "p", "prob_win", "win_prob"]
    p = None
    for c in p_candidates:
        if c in out.columns:
            p = pd.to_numeric(out[c], errors="coerce").clip(0, 1)
            break
    out["p_win"] = p if p is not None else np.nan

    dec_candidates = ["_payout_decimal", "payout_decimal", "decimal_odds", "payout"]
    amer_candidates = ["american_odds", "odds", "american"]

    dec = None
    for c in dec_candidates:
        if c in out.columns:
            dec = pd.to_numeric(out[c], errors="coerce")
            break

    if dec is None:
        amer = None
        for c in amer_candidates:
            if c in out.columns:
                amer = out[c]
                break
        out["_payout_decimal"] = (
            pd.to_numeric(amer, errors="coerce").map(american_to_decimal)
            if amer is not None
            else np.nan
        )
    else:
        out["_payout_decimal"] = dec

    p = pd.to_numeric(out.get("p_win"), errors="coerce")
    d = pd.to_numeric(out.get("_payout_decimal"), errors="coerce")
    out["_ev_per_$1"] = p * d - (1.0 - p)
    return out


def ev_metrics(df_like: dict | pd.DataFrame) -> dict[str, Any]:
    get = df_like.get if hasattr(df_like, "get") else lambda k, default=None: getattr(df_like, k, default)

    p = to_series(get("p_win", np.nan))
    payout = to_series(get("_payout_decimal", get("payout_decimal", get("payout", np.nan))))
    ev = to_series(get("_ev_per_$1", get("ev", np.nan)))

    with_ev = int(ev.notna().sum())

    return {
        "rows": int(len(p)),
        "with_p": int(p.notna().sum()),
        "with_payout": int(payout.notna().sum()),
        "with_ev": with_ev,
        "ev_mean": float(ev.mean(skipna=True)) if with_ev else float("nan"),
        "ev_median": float(ev.median(skipna=True)) if with_ev else float("nan"),
        "ev_p95": float(ev.quantile(0.95)) if with_ev else float("nan"),
        "ev_p05": float(ev.quantile(0.05)) if with_ev else float("nan"),
        "ev_pos": int((ev > 0).sum()) if with_ev else 0,
        "ev_nonneg": int((ev >= 0).sum()) if with_ev else 0,
    }


def summarize_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if by not in df.columns:
        return pd.DataFrame(
            columns=[
                "model",
                "rows",
                "with_p",
                "with_payout",
                "with_ev",
                "ev_mean",
                "ev_median",
                "ev_p95",
                "ev_p05",
                "ev_pos",
                "ev_nonneg",
            ]
        )

    recs = []
    for model_name, g in df.groupby(by, dropna=False):
        recs.append({"model": str(model_name), **ev_metrics(g)})

    return pd.DataFrame(recs).sort_values(
        ["ev_mean", "with_ev", "rows"],
        ascending=[False, False, False],
        na_position="last",
    )


exp = exports_dir()
st.caption(f"Build: `{BUILD_TAG}` · exports: `{exp}`")

default_candidates = [
    exp / "edges_models.csv",
    exp / "edges_graded_plus.csv",
    exp / "edges_graded_full.csv",
    exp / "predictions.csv",
]

path_found = next((p for p in default_candidates if p.exists()), None)

left, right = st.columns([2, 1])

with left:
    st.write("**Source**")
    src_choice = st.radio(
        "Pick a source file",
        ["Auto-detected", "Upload CSV"],
        captions=["Use first existing default", "Upload a custom file"],
        label_visibility="collapsed",
        horizontal=True,
    )

    uploaded = None
    if src_choice == "Upload CSV":
        uploaded = st.file_uploader("Upload model predictions CSV", type=["csv"])
    elif path_found:
        st.caption(f"Auto: `{path_found.name}`")
    else:
        st.warning("No default file found in exports/. Upload a CSV.")

with right:
    st.write("**Model column**")
    model_col = st.text_input("Model name column", value="_model")
    st.write("**Probability column**")
    prob_col = st.text_input("p_win column", value="p_win")
    st.write("**Payout column**")
    payout_col = st.text_input("Decimal payout column", value="_payout_decimal")
    st.write("**American odds column (fallback)**")
    american_col = st.text_input("American odds column", value="american_odds")

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded, low_memory=False)
    elif path_found:
        df = pd.read_csv(path_found, low_memory=False)
    else:
        df = pd.DataFrame()
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

if df.empty:
    st.info("No data loaded yet.")
    st.stop()

df = add_probs_and_ev(df)

if prob_col in df.columns and prob_col != "p_win":
    df["p_win"] = pd.to_numeric(df[prob_col], errors="coerce").clip(0, 1)

if payout_col in df.columns and payout_col != "_payout_decimal":
    df["_payout_decimal"] = pd.to_numeric(df[payout_col], errors="coerce")

if american_col in df.columns and df["_payout_decimal"].isna().all():
    df["_payout_decimal"] = pd.to_numeric(df[american_col], errors="coerce").map(american_to_decimal)

if model_col not in df.columns:
    for alt in ["model", "model_name", "algo", "_model_name"]:
        if alt in df.columns:
            model_col = alt
            break
    else:
        df[model_col] = "unknown"

with st.expander("Filters", expanded=False):
    if "_league" in df.columns:
        leagues = sorted(df["_league"].dropna().astype(str).unique().tolist())
        chosen_leagues = st.multiselect("League(s)", leagues, default=leagues[:4] if leagues else [])
        if chosen_leagues:
            df = df[df["_league"].astype(str).isin(chosen_leagues)]

    min_p = st.slider("Min p_win", 0.0, 1.0, 0.0, 0.01)
    max_p = st.slider("Max p_win", 0.0, 1.0, 1.0, 0.01)
    df = df[df["p_win"].between(min_p, max_p, inclusive="both")]

    if "_ev_per_$1" in df.columns:
        min_ev = st.slider("Min EV per $1", -1.0, 2.0, -1.0, 0.01)
        df = df[df["_ev_per_$1"] >= min_ev]

overall = ev_metrics(df)

st.subheader("Overall")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Rows", f"{overall['rows']:,}")
c2.metric("With p", f"{overall['with_p']:,}")
c3.metric("With payout", f"{overall['with_payout']:,}")
c4.metric("With EV", f"{overall['with_ev']:,}")
c5.metric("EV mean", f"{overall['ev_mean']:.4f}" if not math.isnan(overall["ev_mean"]) else "—")
c6.metric("EV median", f"{overall['ev_median']:.4f}" if not math.isnan(overall["ev_median"]) else "—")
c7.metric("EV p95", f"{overall['ev_p95']:.4f}" if not math.isnan(overall["ev_p95"]) else "—")

st.subheader("By Model")
by_model = summarize_by(df, model_col)
st.dataframe(by_model, use_container_width=True, hide_index=True)

with st.expander("Sample Rows (cleaned view)", expanded=False):
    cols_show = [c for c in ["_league", model_col, "p_win", "_payout_decimal", "_ev_per_$1", "american_odds"] if c in df.columns]
    st.dataframe(df[cols_show].head(100), use_container_width=True, hide_index=True)

colA, colB = st.columns(2)
with colA:
    st.download_button(
        "Download per-model summary (CSV)",
        by_model.to_csv(index=False).encode("utf-8"),
        file_name="compare_models_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
with colB:
    st.download_button(
        "Download filtered rows (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="compare_models_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(f"✔ Ready · {len(df):,} rows after filters · model column: `{model_col}`")