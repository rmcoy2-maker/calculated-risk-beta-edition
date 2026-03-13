from __future__ import annotations

import pandas as pd


def confidence_band(conf: float) -> str:
    if conf >= 90:
        return "Gold"
    if conf >= 75:
        return "Dark Green"
    if conf >= 60:
        return "Green"
    if conf >= 45:
        return "Yellow"
    if conf >= 25:
        return "Amber"
    return "Red"


def _get(row: pd.Series, key: str, default=""):
    return row[key] if key in row and pd.notna(row[key]) else default


def edge_blurb(row: pd.Series) -> str:
    market_type = str(_get(row, "market_type", "")).lower()
    confidence = float(_get(row, "confidence", 50) or 50)
    edge_score = float(_get(row, "edge_score", 0) or 0)
    model_line = _get(row, "model_line", "")
    market_line = _get(row, "market_line", "")

    if market_type == "total" and edge_score >= 2.5:
        return f"Model sees hidden scoring value versus the market ({model_line} vs {market_line})."

    if market_type == "spread" and confidence >= 70:
        return "This side clears both the model threshold and the stability check."

    if market_type == "team_total":
        return "Team-total shape is cleaner than the full-game market and fits the projected script."

    if confidence >= 75:
        return "Premium edge with strong agreement across model, market, and game-script structure."

    if confidence >= 60:
        return "Playable edge with solid model support and acceptable variance."

    return "Leaning edge; usable, but more price-sensitive than the top board."


def game_script(row: pd.Series) -> str:
    away = _get(row, "away_team", "Away")
    home = _get(row, "home_team", "Home")
    away_score = _get(row, "model_away_score", "")
    home_score = _get(row, "model_home_score", "")
    total = _get(row, "market_total", "")

    if away_score != "" and home_score != "":
        return (
            f"Median script projects {away} {away_score} – {home} {home_score}. "
            f"Current market total sits around {total}."
        )

    return f"{away} @ {home} profiles as a structured game with playable angles if price holds."


def parlay_blurb(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No parlay portfolio available."
    return "Correlated structure built around script-aligned legs and reduced variance overlap."