from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Closing Line Intelligence",
    page_icon="🎯",
    layout="wide",
)


# -------------------------------------------------------------------
# CLV helpers
# -------------------------------------------------------------------
def implied_prob_from_american(odds: float | int | None) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    o = float(odds)
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def american_to_decimal(odds: float | int | None) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    o = float(odds)
    if o == 0:
        return None
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def calc_spread_clv(side: str, bet_line: float, close_line: float) -> float:
    if pd.isna(bet_line) or pd.isna(close_line):
        return float("nan")

    side = str(side).lower().strip()
    if side in {"home_spread", "home"}:
        return float(bet_line) - float(close_line)
    if side in {"away_spread", "away"}:
        return float(close_line) - float(bet_line)
    return float("nan")


def calc_total_clv(side: str, bet_line: float, close_line: float) -> float:
    if pd.isna(bet_line) or pd.isna(close_line):
        return float("nan")

    side = str(side).lower().strip()
    if side in {"over", "o"}:
        return float(close_line) - float(bet_line)
    if side in {"under", "u"}:
        return float(bet_line) - float(close_line)
    return float("nan")


def calc_ml_clv_odds(bet_odds: float, close_odds: float) -> float:
    if pd.isna(bet_odds) or pd.isna(close_odds):
        return float("nan")
    return float(bet_odds) - float(close_odds)


def calc_ml_clv_prob_delta(bet_odds: float, close_odds: float) -> float:
    if pd.isna(bet_odds) or pd.isna(close_odds):
        return float("nan")

    bet_prob = implied_prob_from_american(float(bet_odds))
    close_prob = implied_prob_from_american(float(close_odds))

    if bet_prob is None or close_prob is None:
        return float("nan")

    return float(close_prob) - float(bet_prob)


def classify_clv(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value > 0:
        return "Positive CLV"
    if value < 0:
        return "Negative CLV"
    return "Push CLV"


def add_clv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["clv_line"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["clv_odds"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["clv_implied_prob_delta"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["clv_result"] = "Unknown"
    out["beat_closing"] = pd.Series(np.nan, index=out.index, dtype=float)

    for idx, r in out.iterrows():
        market = str(r.get("market", "")).lower().strip()
        side = str(r.get("side", "")).lower().strip()

        bet_line = pd.to_numeric(
            pd.Series([r.get("line_at_pick", r.get("line"))]),
            errors="coerce",
        ).iloc[0]
        close_line = pd.to_numeric(
            pd.Series([r.get("closing_line")]),
            errors="coerce",
        ).iloc[0]
        bet_odds = pd.to_numeric(
            pd.Series([r.get("payout_odds", r.get("odds_at_pick", r.get("odds")))]),
            errors="coerce",
        ).iloc[0]
        close_odds = pd.to_numeric(
            pd.Series([r.get("closing_odds")]),
            errors="coerce",
        ).iloc[0]

        clv_value = float("nan")

        if "spread" in market:
            clv_value = calc_spread_clv(side, bet_line, close_line)
            out.at[idx, "clv_line"] = clv_value

        elif "total" in market or market.startswith("o/"):
            clv_value = calc_total_clv(side, bet_line, close_line)
            out.at[idx, "clv_line"] = clv_value

        elif "moneyline" in market or market == "ml" or market == "h2h":
            clv_odds = calc_ml_clv_odds(bet_odds, close_odds)
            clv_prob = calc_ml_clv_prob_delta(bet_odds, close_odds)
            out.at[idx, "clv_odds"] = clv_odds
            out.at[idx, "clv_implied_prob_delta"] = clv_prob
            clv_value = clv_prob

        out.at[idx, "clv_result"] = classify_clv(clv_value)
        if not pd.isna(clv_value):
            out.at[idx, "beat_closing"] = 1.0 if clv_value > 0 else 0.0

    return out


def roi_flat_stake(df: pd.DataFrame) -> float:
    w = df.copy()
    if "hit_bool" not in w.columns:
        return float("nan")

    w = w[pd.notna(w["hit_bool"])]
    if w.empty:
        return float("nan")

    if "decimal_odds" in w.columns and w["decimal_odds"].notna().any():
        dec = pd.to_numeric(w["decimal_odds"], errors="coerce")
    elif "payout_odds" in w.columns and w["payout_odds"].notna().any():
        dec = w["payout_odds"].apply(american_to_decimal).astype(float)
    else:
        dec = pd.Series([1.909] * len(w), index=w.index, dtype=float)

    returns = w["hit_bool"].astype(float) * (dec - 1.0)
    stake = len(w)
    total_return = returns.sum()
    return float((total_return - stake) / stake)


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
root = Path(__file__).resolve().parents[3]
exports = root / "exports"

candidates = [
    exports / "bets_log.csv",
    exports / "settled.csv",
]

source_path = None
for p in candidates:
    if p.exists():
        source_path = p
        break

st.title("🎯 Closing Line Intelligence")

if source_path is None:
    st.error("No suitable data file found in exports/. Expected bets_log.csv or settled.csv.")
    st.stop()

try:
    df = pd.read_csv(source_path, low_memory=False)
except Exception as exc:
    st.error(f"Could not read {source_path.name}: {exc}")
    st.stop()

try:
    df = add_clv_columns(df)
except Exception as exc:
    st.warning(f"CLV enrichment failed: {exc}")

# normalize hit_bool if possible
if "hit_bool" not in df.columns and "result" in df.columns:
    result_map = {"W": 1.0, "WIN": 1.0, "L": 0.0, "LOSS": 0.0}
    df["hit_bool"] = df["result"].astype(str).str.upper().map(result_map)

st.caption(f"Source: {source_path}")
st.caption(f"Rows loaded: {len(df):,}")


# -------------------------------------------------------------------
# Filters
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    if "market" in df.columns:
        markets = sorted(df["market"].dropna().astype(str).unique().tolist())
        sel_markets = st.multiselect("Market", markets, default=markets)
    else:
        sel_markets = []

    if "book" in df.columns:
        books = sorted(df["book"].dropna().astype(str).unique().tolist())
        sel_books = st.multiselect("Book", books, default=books)
    else:
        sel_books = []

    min_rows = st.slider("Minimum sample size", 1, 200, 10, 1)

view = df.copy()

if sel_markets and "market" in view.columns:
    view = view[view["market"].astype(str).isin(sel_markets)]

if sel_books and "book" in view.columns:
    view = view[view["book"].astype(str).isin(sel_books)]


# -------------------------------------------------------------------
# Topline metrics
# -------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Rows", f"{len(view):,}")

with m2:
    if "beat_closing" in view.columns and view["beat_closing"].notna().any():
        pct = view["beat_closing"].dropna().mean() * 100
        st.metric("Beat Closing %", f"{pct:0.1f}%")
    else:
        st.metric("Beat Closing %", "—")

with m3:
    if "clv_line" in view.columns and view["clv_line"].notna().any():
        st.metric("Avg CLV Line", f"{view['clv_line'].dropna().mean():0.3f}")
    else:
        st.metric("Avg CLV Line", "—")

with m4:
    if "clv_implied_prob_delta" in view.columns and view["clv_implied_prob_delta"].notna().any():
        st.metric("Avg CLV Prob Δ", f"{view['clv_implied_prob_delta'].dropna().mean() * 100:0.3f}%")
    else:
        st.metric("Avg CLV Prob Δ", "—")


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Book intelligence
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Book Intelligence")

if "book" in view.columns:
    book_source = view.copy()
    book_source = book_source[book_source["book"].notna()].copy()

    rows = []

    if not book_source.empty:
        for book, g in book_source.groupby("book", dropna=True):
            rows.append(
                {
                    "book": book,
                    "rows": len(g),
                    "positive_clv_pct": (
                        g["clv_result"].eq("Positive CLV").mean() * 100
                        if "clv_result" in g.columns
                        else np.nan
                    ),
                    "beat_closing_pct": (
                        g["beat_closing"].dropna().mean() * 100
                        if "beat_closing" in g.columns and g["beat_closing"].notna().any()
                        else np.nan
                    ),
                    "avg_clv_line": (
                        g["clv_line"].dropna().mean()
                        if "clv_line" in g.columns and g["clv_line"].notna().any()
                        else np.nan
                    ),
                    "avg_clv_odds": (
                        g["clv_odds"].dropna().mean()
                        if "clv_odds" in g.columns and g["clv_odds"].notna().any()
                        else np.nan
                    ),
                    "avg_clv_prob_delta_pct": (
                        g["clv_implied_prob_delta"].dropna().mean() * 100
                        if "clv_implied_prob_delta" in g.columns
                        and g["clv_implied_prob_delta"].notna().any()
                        else np.nan
                    ),
                    "roi_flat_pct": (
                        roi_flat_stake(g) * 100
                        if "hit_bool" in g.columns
                        else np.nan
                    ),
                }
            )

    if not rows:
        st.caption("No book-level CLV rows available after filters.")
    else:
        book_tbl = pd.DataFrame(rows)

        if "rows" in book_tbl.columns:
            book_tbl = book_tbl[book_tbl["rows"] >= min_rows].copy()

        if book_tbl.empty:
            st.caption("No books meet the current minimum sample filter.")
        else:
            book_tbl = book_tbl.sort_values(
                ["positive_clv_pct", "roi_flat_pct"],
                ascending=False,
                na_position="last",
            )
            st.dataframe(book_tbl, use_container_width=True, hide_index=True)
else:
    st.caption("Need column: book")


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Market intelligence
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Market Intelligence")

if "market" in view.columns:
    market_source = view.copy()
    market_source = market_source[market_source["market"].notna()].copy()

    rows = []

    if not market_source.empty:
        for market, g in market_source.groupby("market", dropna=True):
            rows.append(
                {
                    "market": market,
                    "rows": len(g),
                    "positive_clv_pct": (
                        g["clv_result"].eq("Positive CLV").mean() * 100
                        if "clv_result" in g.columns
                        else np.nan
                    ),
                    "beat_closing_pct": (
                        g["beat_closing"].dropna().mean() * 100
                        if "beat_closing" in g.columns and g["beat_closing"].notna().any()
                        else np.nan
                    ),
                    "avg_clv_line": (
                        g["clv_line"].dropna().mean()
                        if "clv_line" in g.columns and g["clv_line"].notna().any()
                        else np.nan
                    ),
                    "avg_clv_odds": (
                        g["clv_odds"].dropna().mean()
                        if "clv_odds" in g.columns and g["clv_odds"].notna().any()
                        else np.nan
                    ),
                    "avg_clv_prob_delta_pct": (
                        g["clv_implied_prob_delta"].dropna().mean() * 100
                        if "clv_implied_prob_delta" in g.columns
                        and g["clv_implied_prob_delta"].notna().any()
                        else np.nan
                    ),
                    "roi_flat_pct": (
                        roi_flat_stake(g) * 100
                        if "hit_bool" in g.columns
                        else np.nan
                    ),
                }
            )

    if not rows:
        st.caption("No market-level CLV rows available after filters.")
    else:
        market_tbl = pd.DataFrame(rows)

        if "rows" in market_tbl.columns:
            market_tbl = market_tbl[market_tbl["rows"] >= min_rows].copy()

        if market_tbl.empty:
            st.caption("No markets meet the current minimum sample filter.")
        else:
            market_tbl = market_tbl.sort_values(
                ["positive_clv_pct", "roi_flat_pct"],
                ascending=False,
                na_position="last",
            )
            st.dataframe(market_tbl, use_container_width=True, hide_index=True)
else:
    st.caption("Need column: market")


# -------------------------------------------------------------------
# Book x Market matrix
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Book × Market Matrix")

if "book" in view.columns and "market" in view.columns:
    tabs = st.tabs(["Beat Closing %", "Positive CLV %", "Avg CLV Line", "Avg CLV Prob Δ"])

    with tabs[0]:
        tmp = view[pd.notna(view["beat_closing"])].copy() if "beat_closing" in view.columns else pd.DataFrame()
        if not tmp.empty:
            pivot = tmp.pivot_table(index="book", columns="market", values="beat_closing", aggfunc="mean")
            st.dataframe((pivot * 100).round(1).fillna("—"), use_container_width=True)
        else:
            st.caption("No beat-closing data available.")

    with tabs[1]:
        tmp = view.copy()
        if "clv_result" in tmp.columns:
            tmp["positive_clv_flag"] = (tmp["clv_result"] == "Positive CLV").astype(float)
            pivot = tmp.pivot_table(index="book", columns="market", values="positive_clv_flag", aggfunc="mean")
            st.dataframe((pivot * 100).round(1).fillna("—"), use_container_width=True)
        else:
            st.caption("No clv_result data available.")

    with tabs[2]:
        tmp = view[pd.notna(view["clv_line"])].copy() if "clv_line" in view.columns else pd.DataFrame()
        if not tmp.empty:
            pivot = tmp.pivot_table(index="book", columns="market", values="clv_line", aggfunc="mean")
            st.dataframe(pivot.round(3).fillna("—"), use_container_width=True)
        else:
            st.caption("No clv_line data available.")

    with tabs[3]:
        tmp = view[pd.notna(view["clv_implied_prob_delta"])].copy() if "clv_implied_prob_delta" in view.columns else pd.DataFrame()
        if not tmp.empty:
            pivot = tmp.pivot_table(index="book", columns="market", values="clv_implied_prob_delta", aggfunc="mean")
            st.dataframe((pivot * 100).round(3).fillna("—"), use_container_width=True)
        else:
            st.caption("No clv_implied_prob_delta data available.")
else:
    st.caption("Need columns: book and market")


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Positive CLV vs ROI
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Positive CLV vs ROI")

if "book" in view.columns:
    compare_source = view.copy()
    compare_source = compare_source[compare_source["book"].notna()].copy()

    rows = []

    if not compare_source.empty:
        for book, g in compare_source.groupby("book", dropna=True):
            rows.append(
                {
                    "book": book,
                    "rows": len(g),
                    "positive_clv_pct": (
                        g["clv_result"].eq("Positive CLV").mean() * 100
                        if "clv_result" in g.columns
                        else np.nan
                    ),
                    "roi_flat_pct": (
                        roi_flat_stake(g) * 100
                        if "hit_bool" in g.columns
                        else np.nan
                    ),
                }
            )

    if not rows:
        st.caption("No comparison rows available after filters.")
    else:
        compare_tbl = pd.DataFrame(rows)

        if "rows" in compare_tbl.columns:
            compare_tbl = compare_tbl[compare_tbl["rows"] >= min_rows].copy()

        if compare_tbl.empty:
            st.caption("No books meet the current minimum sample filter.")
        else:
            compare_tbl = compare_tbl.sort_values(
                "positive_clv_pct",
                ascending=False,
                na_position="last",
            )
            st.dataframe(compare_tbl, use_container_width=True, hide_index=True)
else:
    st.caption("Need column: book")


# -------------------------------------------------------------------
# Export
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Export")

export_cols = [
    c for c in [
        "timestamp",
        "date",
        "book",
        "market",
        "selection",
        "side",
        "line_at_pick",
        "closing_line",
        "payout_odds",
        "closing_odds",
        "clv_line",
        "clv_odds",
        "clv_implied_prob_delta",
        "clv_result",
        "beat_closing",
        "result",
        "hit_bool",
    ]
    if c in view.columns
]

buf = io.StringIO()
view[export_cols].to_csv(buf, index=False)

st.download_button(
    "Download closing_line_intelligence.csv",
    data=buf.getvalue(),
    file_name="closing_line_intelligence.csv",
    mime="text/csv",
)