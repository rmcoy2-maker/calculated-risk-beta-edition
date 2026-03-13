from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


# -------------------------------------------------------------------
# Project root detection
# -------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PAGES_DIR = THIS_FILE.parent
APP_DIR = PAGES_DIR.parent

candidates = [
    APP_DIR.parent.parent,
    APP_DIR.parent,
]

ROOT = None
for candidate in candidates:
    if (candidate / "exports").exists() or (candidate / "tools").exists():
        ROOT = candidate
        break

if ROOT is None:
    ROOT = candidates[0]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPORTS_DIR = ROOT / "exports"


# -------------------------------------------------------------------
# File discovery / loading
# -------------------------------------------------------------------
def find_candidate_files() -> list[Path]:
    if not EXPORTS_DIR.exists():
        return []

    preferred: list[Path] = []
    fallback: list[Path] = []

    for p in EXPORTS_DIR.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".xlsx"}:
            continue

        name = p.name.lower()
        if any(
            key in name
            for key in [
                "games_master",
                "edge",
                "ev",
                "lines",
                "markets",
                "model",
                "picks",
                "scored",
                "tiers",
                "template",
            ]
        ):
            preferred.append(p)
        else:
            fallback.append(p)

    return sorted(preferred) + sorted(fallback)


@st.cache_data(show_spinner=False)
def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False)
    if p.suffix.lower() == ".xlsx":
        return pd.read_excel(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")


# -------------------------------------------------------------------
# Odds / EV math
# -------------------------------------------------------------------
def american_to_implied_prob(odds: float) -> float:
    if pd.isna(odds):
        return float("nan")
    odds = float(odds)
    if odds == 0:
        return float("nan")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds: float) -> float:
    if pd.isna(odds):
        return float("nan")
    odds = float(odds)
    if odds == 0:
        return float("nan")
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def prob_to_fair_american(prob: float) -> float:
    if pd.isna(prob) or prob <= 0 or prob >= 1:
        return float("nan")
    if prob >= 0.5:
        return -(prob / (1.0 - prob)) * 100.0
    return ((1.0 - prob) / prob) * 100.0


def ev_per_dollar(model_prob: float, american_odds: float) -> float:
    if pd.isna(model_prob) or pd.isna(american_odds):
        return float("nan")
    dec = american_to_decimal(float(american_odds))
    if pd.isna(dec):
        return float("nan")
    return (float(model_prob) * dec) - 1.0


def kelly_fraction(model_prob: float, american_odds: float) -> float:
    if pd.isna(model_prob) or pd.isna(american_odds):
        return float("nan")
    dec = american_to_decimal(float(american_odds))
    if pd.isna(dec) or dec <= 1:
        return float("nan")
    b = dec - 1.0
    p = float(model_prob)
    q = 1.0 - p
    k = ((b * p) - q) / b
    return max(0.0, k)


def recommended_stake(kelly: float, bankroll: float = 1000.0, tier: str = "Pro") -> float:
    if pd.isna(kelly):
        return 0.0

    tier_mult = {
        "Rookie": 0.10,
        "Pro": 0.25,
        "Elite": 0.50,
        "Normal": 0.20,
    }.get(tier, 0.25)

    return round(max(0.0, float(kelly)) * bankroll * tier_mult, 2)


# -------------------------------------------------------------------
# Normal-CDF approximation
# -------------------------------------------------------------------
def std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_from_line_delta(delta: float, sigma: float) -> float:
    if pd.isna(delta) or sigma <= 0:
        return float("nan")
    z = float(delta) / float(sigma)
    p = std_norm_cdf(z)
    return min(max(p, 0.001), 0.999)


# -------------------------------------------------------------------
# Column helpers
# -------------------------------------------------------------------
def first_existing_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def get_num(r: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in r.index:
            val = pd.to_numeric(pd.Series([r.get(name)]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val)
    return float("nan")


def get_text(r: pd.Series, names: list[str], default: str = "") -> str:
    for name in names:
        if name in r.index:
            val = r.get(name)
            if pd.notna(val):
                return str(val)
    return default


def classify_edge(ev: float) -> str:
    if pd.isna(ev):
        return "None"
    if ev >= 0.08:
        return "Max"
    if ev >= 0.05:
        return "Strong"
    if ev >= 0.02:
        return "Playable"
    if ev > 0:
        return "Lean"
    return "Negative"


def classify_confidence(prob: float) -> str:
    if pd.isna(prob):
        return "Unknown"
    if prob >= 0.60:
        return "Elite"
    if prob >= 0.56:
        return "High"
    if prob >= 0.53:
        return "Moderate"
    return "Low"


# -------------------------------------------------------------------
# Bet engine builder
# -------------------------------------------------------------------
def build_bet_engine_from_games(
    df: pd.DataFrame,
    tier_mode: str,
    bankroll: float,
    spread_sigma: float,
    total_sigma: float,
    ml_sigma: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for _, r in df.iterrows():
        home = get_text(r, ["home_team", "home"], "HOME")
        away = get_text(r, ["away_team", "away"], "AWAY")
        game = f"{away} @ {home}"

        game_id = r.get("game_id", None)
        season = r.get("season", None)
        week = r.get("week", None)
        game_date = r.get("game_date", r.get("date", None))
        book = r.get("book", None)

        spread_close = get_num(r, ["spread_close", "spread", "close_spread"])
        total_close = get_num(r, ["total_close", "total", "close_total"])
        ml_home = get_num(r, ["ml_home", "moneyline_home", "home_ml"])
        ml_away = get_num(r, ["ml_away", "moneyline_away", "away_ml"])

        exp_spread = get_num(r, ["exp_spread", "proj_spread", "pred_spread"])
        exp_total = get_num(r, ["exp_total", "proj_total", "pred_total"])
        home_edge = get_num(r, ["home_edge", "ml_edge_home", "moneyline_edge_home"])
        home_win_prob = get_num(r, ["home_win_prob", "home_prob", "ml_home_prob", "win_prob_home"])

        if pd.notna(spread_close) and pd.notna(exp_spread):
            home_cover_prob = prob_from_line_delta(exp_spread - spread_close, spread_sigma)
            away_cover_prob = 1.0 - home_cover_prob

            spread_rows = [
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "game_date": game_date,
                    "game": game,
                    "book": book,
                    "market": "spread",
                    "selection": home,
                    "side": "home_spread",
                    "line": spread_close,
                    "odds": -110.0,
                    "model_line": exp_spread,
                    "market_line": spread_close,
                    "model_prob": home_cover_prob,
                },
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "game_date": game_date,
                    "game": game,
                    "book": book,
                    "market": "spread",
                    "selection": away,
                    "side": "away_spread",
                    "line": -spread_close,
                    "odds": -110.0,
                    "model_line": -exp_spread,
                    "market_line": -spread_close,
                    "model_prob": away_cover_prob,
                },
            ]

            for row in spread_rows:
                row["implied_prob"] = american_to_implied_prob(row["odds"])
                row["fair_odds"] = prob_to_fair_american(row["model_prob"])
                row["edge"] = row["model_prob"] - row["implied_prob"]
                row["ev_per_1"] = ev_per_dollar(row["model_prob"], row["odds"])
                row["kelly"] = kelly_fraction(row["model_prob"], row["odds"])
                row["recommended_stake"] = recommended_stake(row["kelly"], bankroll, tier_mode)
                row["confidence_score"] = round(row["model_prob"] * 100.0, 1)
                row["confidence_band"] = classify_confidence(row["model_prob"])
                row["edge_tier"] = classify_edge(row["ev_per_1"])
                rows.append(row)

        if pd.notna(total_close) and pd.notna(exp_total):
            over_prob = prob_from_line_delta(exp_total - total_close, total_sigma)
            under_prob = 1.0 - over_prob

            total_rows = [
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "game_date": game_date,
                    "game": game,
                    "book": book,
                    "market": "total",
                    "selection": "Over",
                    "side": "over",
                    "line": total_close,
                    "odds": -110.0,
                    "model_line": exp_total,
                    "market_line": total_close,
                    "model_prob": over_prob,
                },
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "game_date": game_date,
                    "game": game,
                    "book": book,
                    "market": "total",
                    "selection": "Under",
                    "side": "under",
                    "line": total_close,
                    "odds": -110.0,
                    "model_line": exp_total,
                    "market_line": total_close,
                    "model_prob": under_prob,
                },
            ]

            for row in total_rows:
                row["implied_prob"] = american_to_implied_prob(row["odds"])
                row["fair_odds"] = prob_to_fair_american(row["model_prob"])
                row["edge"] = row["model_prob"] - row["implied_prob"]
                row["ev_per_1"] = ev_per_dollar(row["model_prob"], row["odds"])
                row["kelly"] = kelly_fraction(row["model_prob"], row["odds"])
                row["recommended_stake"] = recommended_stake(row["kelly"], bankroll, tier_mode)
                row["confidence_score"] = round(row["model_prob"] * 100.0, 1)
                row["confidence_band"] = classify_confidence(row["model_prob"])
                row["edge_tier"] = classify_edge(row["ev_per_1"])
                rows.append(row)

        ml_home_prob = float("nan")
        if pd.notna(home_win_prob):
            ml_home_prob = max(min(home_win_prob, 0.999), 0.001)
        elif pd.notna(home_edge):
            ml_home_prob = prob_from_line_delta(home_edge, ml_sigma)

        if pd.notna(ml_home_prob):
            ml_rows = []

            if pd.notna(ml_home):
                ml_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "game_date": game_date,
                        "game": game,
                        "book": book,
                        "market": "moneyline",
                        "selection": home,
                        "side": "home_ml",
                        "line": None,
                        "odds": ml_home,
                        "model_line": None,
                        "market_line": ml_home,
                        "model_prob": ml_home_prob,
                    }
                )

            if pd.notna(ml_away):
                ml_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "game_date": game_date,
                        "game": game,
                        "book": book,
                        "market": "moneyline",
                        "selection": away,
                        "side": "away_ml",
                        "line": None,
                        "odds": ml_away,
                        "model_line": None,
                        "market_line": ml_away,
                        "model_prob": 1.0 - ml_home_prob,
                    }
                )

            for row in ml_rows:
                row["implied_prob"] = american_to_implied_prob(row["odds"])
                row["fair_odds"] = prob_to_fair_american(row["model_prob"])
                row["edge"] = row["model_prob"] - row["implied_prob"]
                row["ev_per_1"] = ev_per_dollar(row["model_prob"], row["odds"])
                row["kelly"] = kelly_fraction(row["model_prob"], row["odds"])
                row["recommended_stake"] = recommended_stake(row["kelly"], bankroll, tier_mode)
                row["confidence_score"] = round(row["model_prob"] * 100.0, 1)
                row["confidence_band"] = classify_confidence(row["model_prob"])
                row["edge_tier"] = classify_edge(row["ev_per_1"])
                rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(
        by=["ev_per_1", "edge", "recommended_stake"],
        ascending=False,
    ).reset_index(drop=True)


# -------------------------------------------------------------------
# Market dislocation scanner
# -------------------------------------------------------------------
def build_market_dislocations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Works on raw market board data even without model outputs.

    Finds same game across multiple books and highlights:
    - ML home dislocation
    - ML away dislocation
    - Spread dislocation
    - Total dislocation
    """
    required = {"game_id", "home_team", "away_team"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    work = df.copy()

    if "game" not in work.columns:
        work["game"] = work["away_team"].astype(str) + " @ " + work["home_team"].astype(str)

    for col in ["ml_home", "ml_away", "spread_close", "total_close"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "book" not in work.columns:
        work["book"] = "Unknown"

    rows: list[dict] = []

    grouped = work.groupby(["game_id", "game"], dropna=False)

    for (game_id, game), g in grouped:
        unique_books = g["book"].dropna().astype(str).nunique()
        if unique_books < 2:
            continue

        # ML home
        if "ml_home" in g.columns and g["ml_home"].notna().sum() >= 2:
            best_idx = g["ml_home"].idxmax()
            worst_idx = g["ml_home"].idxmin()
            best_row = g.loc[best_idx]
            worst_row = g.loc[worst_idx]

            best_odds = pd.to_numeric(best_row["ml_home"], errors="coerce")
            worst_odds = pd.to_numeric(worst_row["ml_home"], errors="coerce")

            if pd.notna(best_odds) and pd.notna(worst_odds) and best_odds != worst_odds:
                rows.append(
                    {
                        "game_id": game_id,
                        "game": game,
                        "market": "moneyline",
                        "selection": str(best_row["home_team"]),
                        "best_book": str(best_row["book"]),
                        "best_price": best_odds,
                        "worst_book": str(worst_row["book"]),
                        "worst_price": worst_odds,
                        "price_gap": best_odds - worst_odds,
                        "best_implied_prob": american_to_implied_prob(best_odds),
                        "worst_implied_prob": american_to_implied_prob(worst_odds),
                        "implied_prob_gap": american_to_implied_prob(worst_odds)
                        - american_to_implied_prob(best_odds),
                    }
                )

        # ML away
        if "ml_away" in g.columns and g["ml_away"].notna().sum() >= 2:
            best_idx = g["ml_away"].idxmax()
            worst_idx = g["ml_away"].idxmin()
            best_row = g.loc[best_idx]
            worst_row = g.loc[worst_idx]

            best_odds = pd.to_numeric(best_row["ml_away"], errors="coerce")
            worst_odds = pd.to_numeric(worst_row["ml_away"], errors="coerce")

            if pd.notna(best_odds) and pd.notna(worst_odds) and best_odds != worst_odds:
                rows.append(
                    {
                        "game_id": game_id,
                        "game": game,
                        "market": "moneyline",
                        "selection": str(best_row["away_team"]),
                        "best_book": str(best_row["book"]),
                        "best_price": best_odds,
                        "worst_book": str(worst_row["book"]),
                        "worst_price": worst_odds,
                        "price_gap": best_odds - worst_odds,
                        "best_implied_prob": american_to_implied_prob(best_odds),
                        "worst_implied_prob": american_to_implied_prob(worst_odds),
                        "implied_prob_gap": american_to_implied_prob(worst_odds)
                        - american_to_implied_prob(best_odds),
                    }
                )

        # Spread
        if "spread_close" in g.columns and g["spread_close"].notna().sum() >= 2:
            max_idx = g["spread_close"].idxmax()
            min_idx = g["spread_close"].idxmin()
            max_row = g.loc[max_idx]
            min_row = g.loc[min_idx]

            max_line = pd.to_numeric(max_row["spread_close"], errors="coerce")
            min_line = pd.to_numeric(min_row["spread_close"], errors="coerce")

            if pd.notna(max_line) and pd.notna(min_line) and max_line != min_line:
                rows.append(
                    {
                        "game_id": game_id,
                        "game": game,
                        "market": "spread",
                        "selection": str(max_row["home_team"]),
                        "best_book": str(max_row["book"]),
                        "best_price": max_line,
                        "worst_book": str(min_row["book"]),
                        "worst_price": min_line,
                        "price_gap": max_line - min_line,
                        "best_implied_prob": None,
                        "worst_implied_prob": None,
                        "implied_prob_gap": None,
                    }
                )

        # Total
        if "total_close" in g.columns and g["total_close"].notna().sum() >= 2:
            max_idx = g["total_close"].idxmax()
            min_idx = g["total_close"].idxmin()
            max_row = g.loc[max_idx]
            min_row = g.loc[min_idx]

            max_line = pd.to_numeric(max_row["total_close"], errors="coerce")
            min_line = pd.to_numeric(min_row["total_close"], errors="coerce")

            if pd.notna(max_line) and pd.notna(min_line) and max_line != min_line:
                rows.append(
                    {
                        "game_id": game_id,
                        "game": game,
                        "market": "total",
                        "selection": "Total",
                        "best_book": str(min_row["book"]),
                        "best_price": min_line,
                        "worst_book": str(max_row["book"]),
                        "worst_price": max_line,
                        "price_gap": max_line - min_line,
                        "best_implied_prob": None,
                        "worst_implied_prob": None,
                        "implied_prob_gap": None,
                    }
                )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # Heuristic ranking score for dislocation importance
    out["dislocation_score"] = (
        out["price_gap"].abs().fillna(0)
        + (out["implied_prob_gap"].fillna(0) * 100.0)
    )

    return out.sort_values(
        by=["dislocation_score", "price_gap"],
        ascending=False,
    ).reset_index(drop=True)


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
def app() -> None:
    st.set_page_config(page_title="Bet Engine", page_icon="🧠", layout="wide")

    st.title("🧠 Bet Engine")
    st.caption("Recovered EV-first Bet Engine with Market Dislocation Scanner.")

    files = find_candidate_files()
    if not files:
        st.error(f"No CSV/XLSX files found under: {EXPORTS_DIR}")
        return

    with st.sidebar:
        st.subheader("Source")
        selected_path = st.selectbox("Select source file", [str(p) for p in files])

        st.subheader("Sizing")
        tier_mode = st.selectbox("Stake Profile", ["Rookie", "Pro", "Elite", "Normal"], index=1)
        bankroll = st.number_input(
            "Bankroll ($)",
            min_value=50.0,
            max_value=1_000_000.0,
            value=1000.0,
            step=50.0,
        )

        st.subheader("Bet Engine Filters")
        min_ev = st.slider("Min EV per $1", -0.20, 0.50, 0.00, 0.01)
        min_edge = st.slider("Min probability edge", -0.20, 0.20, 0.00, 0.005)
        min_conf = st.slider("Min confidence score", 0, 100, 50, 1)
        top_n = st.slider("Top plays", 5, 200, 40, 5)

        st.subheader("Model Calibration")
        spread_sigma = st.slider("Spread sigma", 1.0, 20.0, 13.5, 0.5)
        total_sigma = st.slider("Total sigma", 1.0, 30.0, 14.0, 0.5)
        ml_sigma = st.slider("Moneyline edge sigma", 0.01, 0.30, 0.10, 0.01)

        st.subheader("Dislocation Scanner")
        min_dislocation_gap = st.number_input(
            "Min dislocation gap",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=1.0,
        )
        top_dislocations = st.slider("Top dislocations", 5, 200, 25, 5)

    try:
        raw = load_table(selected_path)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return

    if raw.empty:
        st.warning("Selected file is empty.")
        return

    st.caption(f"Source file: `{selected_path}`")
    st.caption(f"Rows loaded: {len(raw):,}")

    tab1, tab2 = st.tabs(["Bet Engine", "Market Dislocations"])

    with tab1:
        bets = build_bet_engine_from_games(
            raw,
            tier_mode=tier_mode,
            bankroll=bankroll,
            spread_sigma=spread_sigma,
            total_sigma=total_sigma,
            ml_sigma=ml_sigma,
        )

        if bets.empty:
            st.error("No bets could be generated from this file.")
            st.write("Detected columns:")
            st.code(", ".join(raw.columns.astype(str).tolist()))
            st.info(
                "To generate plays, the file needs market columns plus at least one of: "
                "`exp_spread`, `exp_total`, `home_edge`, or `home_win_prob`."
            )
        else:
            plays = bets[
                (bets["ev_per_1"].fillna(-999) >= min_ev)
                & (bets["edge"].fillna(-999) >= min_edge)
                & (bets["confidence_score"].fillna(-999) >= min_conf)
            ].copy()

            market_options = sorted(plays["market"].dropna().astype(str).unique().tolist())
            if market_options:
                selected_markets = st.multiselect(
                    "Market filter",
                    market_options,
                    default=market_options,
                    key="bet_market_filter",
                )
                plays = plays[plays["market"].astype(str).isin(selected_markets)]

            if "book" in plays.columns:
                book_options = sorted(plays["book"].dropna().astype(str).unique().tolist())
                if book_options:
                    selected_books = st.multiselect(
                        "Book filter",
                        book_options,
                        default=book_options,
                        key="bet_book_filter",
                    )
                    plays = plays[plays["book"].astype(str).isin(selected_books)]

            plays = plays.sort_values(
                by=["ev_per_1", "edge", "recommended_stake"],
                ascending=False,
            ).head(top_n)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Generated Bets", f"{len(bets):,}")
            c2.metric("Filtered Plays", f"{len(plays):,}")
            c3.metric("Avg EV", f"{plays['ev_per_1'].mean():.3f}" if not plays.empty else "n/a")
            c4.metric(
                "Total Stake",
                f"${plays['recommended_stake'].sum():,.2f}" if not plays.empty else "$0.00",
            )

            if plays.empty:
                st.warning("No plays matched the current filters.")
            else:
                display_cols = [
                    "game_id",
                    "season",
                    "week",
                    "game_date",
                    "game",
                    "book",
                    "market",
                    "selection",
                    "side",
                    "line",
                    "odds",
                    "model_prob",
                    "implied_prob",
                    "fair_odds",
                    "edge",
                    "ev_per_1",
                    "kelly",
                    "recommended_stake",
                    "confidence_score",
                    "confidence_band",
                    "edge_tier",
                    "model_line",
                    "market_line",
                ]
                display_cols = [c for c in display_cols if c in plays.columns]

                st.dataframe(
                    plays[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download engine_plays.csv",
                    data=plays[display_cols].to_csv(index=False).encode("utf-8"),
                    file_name="engine_plays.csv",
                    mime="text/csv",
                )

    with tab2:
        dislocations = build_market_dislocations(raw)

        if dislocations.empty:
            st.info(
                "No market dislocations found. This usually means either only one book is present "
                "per game, or the selected file does not have comparable multi-book rows."
            )
        else:
            dislocations = dislocations[
                dislocations["price_gap"].abs().fillna(0) >= float(min_dislocation_gap)
            ].copy()

            market_options = sorted(dislocations["market"].dropna().astype(str).unique().tolist())
            if market_options:
                selected_dislocation_markets = st.multiselect(
                    "Dislocation market filter",
                    market_options,
                    default=market_options,
                    key="dislocation_market_filter",
                )
                dislocations = dislocations[
                    dislocations["market"].astype(str).isin(selected_dislocation_markets)
                ]

            dislocations = dislocations.head(top_dislocations)

            d1, d2, d3 = st.columns(3)
            d1.metric("Dislocations", f"{len(dislocations):,}")
            d2.metric(
                "Avg Price Gap",
                f"{dislocations['price_gap'].abs().mean():.2f}" if not dislocations.empty else "n/a",
            )
            d3.metric(
                "Max Price Gap",
                f"{dislocations['price_gap'].abs().max():.2f}" if not dislocations.empty else "n/a",
            )

            if dislocations.empty:
                st.warning("No dislocations matched the current filters.")
            else:
                st.dataframe(
                    dislocations,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download market_dislocations.csv",
                    data=dislocations.to_csv(index=False).encode("utf-8"),
                    file_name="market_dislocations.csv",
                    mime="text/csv",
                )

    with st.expander("Detected source columns"):
        st.code(", ".join(raw.columns.astype(str).tolist()))


if __name__ == "__main__":
    app()