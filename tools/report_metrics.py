from __future__ import annotations

import pandas as pd


def safe_col(df: pd.DataFrame, col: str, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def add_fort_knox_scores(edges: pd.DataFrame) -> pd.DataFrame:
    df = edges.copy()

    df["edge_score_num"] = safe_col(df, "edge_score", 0.0)
    df["confidence_num"] = safe_col(df, "confidence", 50.0)
    df["market_disagreement_num"] = safe_col(df, "market_disagreement", 0.0)
    df["parlay_synergy_num"] = safe_col(df, "parlay_synergy", 0.0)
    df["weather_adjustment_num"] = safe_col(df, "weather_adjustment", 0.0)
    df["stability_num"] = safe_col(df, "stability_score", 50.0)

    df["fort_knox_score"] = (
        0.35 * df["edge_score_num"]
        + 0.20 * df["confidence_num"]
        + 0.15 * df["market_disagreement_num"]
        + 0.10 * df["parlay_synergy_num"]
        + 0.10 * df["weather_adjustment_num"]
        + 0.10 * df["stability_num"]
    )

    return df


def top_edges(edges: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    df = add_fort_knox_scores(edges)
    return df.sort_values("fort_knox_score", ascending=False).head(n).reset_index(drop=True)


def heatmap_board(edges: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    df = add_fort_knox_scores(edges)
    return df.sort_values("fort_knox_score", ascending=False).head(n).reset_index(drop=True)


def top_parlay_legs(parlays: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    df = parlays.copy()
    sort_col = "parlay_score" if "parlay_score" in df.columns else df.columns[0]
    return df.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)