from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_SPREAD_HOME = {"home", "home_spread", "favorite_home", "fav_home"}
_SPREAD_AWAY = {"away", "away_spread", "favorite_away", "fav_away"}
_OVER = {"over", "o"}
_UNDER = {"under", "u"}
_ML = {"h2h", "ml", "moneyline", "money line"}
_SPREAD = {"spread", "spreads"}
_TOTAL = {"total", "totals"}


@dataclass(frozen=True)
class CLVResult:
    clv: float
    beat_closing: float
    market_type: str


def _to_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def american_to_decimal(odds: Any) -> float:
    odds_f = _to_float(odds)
    if np.isnan(odds_f) or odds_f == 0:
        return float("nan")
    if odds_f > 0:
        return 1.0 + odds_f / 100.0
    return 1.0 + 100.0 / abs(odds_f)


def implied_prob_from_american(odds: Any) -> float:
    odds_f = _to_float(odds)
    if np.isnan(odds_f) or odds_f == 0:
        return float("nan")
    if odds_f > 0:
        return 100.0 / (odds_f + 100.0)
    return abs(odds_f) / (abs(odds_f) + 100.0)


def normalize_market(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _ML:
        return "moneyline"
    if text in _SPREAD or text.startswith("spread"):
        return "spread"
    if text in _TOTAL or text.startswith("total"):
        return "total"
    return text


def normalize_side(value: Any) -> str:
    return str(value or "").strip().lower()


def calc_spread_clv(side: Any, bet_line: Any, close_line: Any) -> float:
    side_n = normalize_side(side)
    bet = _to_float(bet_line)
    close = _to_float(close_line)
    if np.isnan(bet) or np.isnan(close):
        return float("nan")
    if side_n in _SPREAD_HOME:
        return bet - close
    if side_n in _SPREAD_AWAY:
        return close - bet
    return float("nan")


def calc_total_clv(side: Any, bet_line: Any, close_line: Any) -> float:
    side_n = normalize_side(side)
    bet = _to_float(bet_line)
    close = _to_float(close_line)
    if np.isnan(bet) or np.isnan(close):
        return float("nan")
    if side_n in _OVER:
        return close - bet
    if side_n in _UNDER:
        return bet - close
    return float("nan")


def calc_moneyline_clv_odds(bet_odds: Any, close_odds: Any) -> float:
    bet = _to_float(bet_odds)
    close = _to_float(close_odds)
    if np.isnan(bet) or np.isnan(close):
        return float("nan")
    return bet - close


def calc_moneyline_clv_prob_delta(bet_odds: Any, close_odds: Any) -> float:
    bet_p = implied_prob_from_american(bet_odds)
    close_p = implied_prob_from_american(close_odds)
    if np.isnan(bet_p) or np.isnan(close_p):
        return float("nan")
    return close_p - bet_p


def compute_clv(
    market: Any,
    side: Any,
    line_at_pick: Any = None,
    closing_line: Any = None,
    payout_odds: Any = None,
    closing_odds: Any = None,
) -> CLVResult:
    market_n = normalize_market(market)
    clv = float("nan")
    if market_n == "spread":
        clv = calc_spread_clv(side=side, bet_line=line_at_pick, close_line=closing_line)
    elif market_n == "total":
        clv = calc_total_clv(side=side, bet_line=line_at_pick, close_line=closing_line)
    elif market_n == "moneyline":
        clv = calc_moneyline_clv_prob_delta(bet_odds=payout_odds, close_odds=closing_odds)
        if np.isnan(clv):
            clv = calc_moneyline_clv_odds(bet_odds=payout_odds, close_odds=closing_odds)

    beat = float("nan")
    if not np.isnan(clv):
        if clv > 0:
            beat = 1.0
        elif clv < 0:
            beat = 0.0
        else:
            beat = 0.5
    return CLVResult(clv=clv, beat_closing=beat, market_type=market_n)


def compute_clv_from_row(row: pd.Series) -> CLVResult:
    return compute_clv(
        market=row.get("market"),
        side=row.get("side"),
        line_at_pick=row.get("line_at_pick", row.get("line")),
        closing_line=row.get("closing_line"),
        payout_odds=row.get("payout_odds", row.get("odds")),
        closing_odds=row.get("closing_odds"),
    )


def apply_clv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        out = pd.DataFrame() if df is None else df.copy()
        if "clv" not in out.columns:
            out["clv"] = pd.Series(dtype=float)
        if "beat_closing" not in out.columns:
            out["beat_closing"] = pd.Series(dtype=float)
        if "market_type" not in out.columns:
            out["market_type"] = pd.Series(dtype="string")
        return out

    out = df.copy()
    results = out.apply(compute_clv_from_row, axis=1)
    out["clv"] = [r.clv for r in results]
    out["beat_closing"] = [r.beat_closing for r in results]
    out["market_type"] = [r.market_type for r in results]
    return out
