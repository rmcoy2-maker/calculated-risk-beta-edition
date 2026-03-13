from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def _normalize_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rename_map = {}

    if "away" in df.columns and "away_team" not in df.columns:
        rename_map["away"] = "away_team"
    if "home" in df.columns and "home_team" not in df.columns:
        rename_map["home"] = "home_team"

    if "visitor_team" in df.columns and "away_team" not in df.columns:
        rename_map["visitor_team"] = "away_team"
    if "team_favorite_id" in df.columns and "favorite_team" not in df.columns:
        rename_map["team_favorite_id"] = "favorite_team"

    if "spread" in df.columns and "market_spread" not in df.columns:
        rename_map["spread"] = "market_spread"
    if "total" in df.columns and "market_total" not in df.columns:
        rename_map["total"] = "market_total"

    if "away_score" in df.columns and "model_away_score" not in df.columns:
        rename_map["away_score"] = "model_away_score"
    if "home_score" in df.columns and "model_home_score" not in df.columns:
        rename_map["home_score"] = "model_home_score"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _add_matchup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "matchup" not in df.columns:
        away = df["away_team"] if "away_team" in df.columns else None
        home = df["home_team"] if "home_team" in df.columns else None
        if away is not None and home is not None:
            df = df.copy()
            df["matchup"] = away.astype(str) + " @ " + home.astype(str)

    return df


def _normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rename_map = {}

    if "edge" in df.columns and "edge_score" not in df.columns:
        rename_map["edge"] = "edge_score"
    if "ev" in df.columns and "edge_score" not in df.columns:
        rename_map["ev"] = "edge_score"
    if "conf" in df.columns and "confidence" not in df.columns:
        rename_map["conf"] = "confidence"
    if "bet" in df.columns and "label" not in df.columns:
        rename_map["bet"] = "label"
    if "play" in df.columns and "label" not in df.columns:
        rename_map["play"] = "label"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "confidence" not in df.columns:
        df["confidence"] = 50.0
    if "edge_score" not in df.columns:
        df["edge_score"] = 0.0

    return _add_matchup(_normalize_team_columns(df))


def _normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _normalize_team_columns(df)
    df = _add_matchup(df)
    return df


def _normalize_markets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _normalize_team_columns(df)
    df = _add_matchup(df)
    return df


def _normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _normalize_team_columns(df)
    df = _add_matchup(df)
    return df


def _normalize_parlays(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rename_map = {}
    if "score" in df.columns and "parlay_score" not in df.columns:
        rename_map["score"] = "parlay_score"
    if "play" in df.columns and "label" not in df.columns:
        rename_map["play"] = "label"
    if "leg" in df.columns and "label" not in df.columns:
        rename_map["leg"] = "label"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "parlay_score" not in df.columns:
        df["parlay_score"] = 0.0

    return _add_matchup(_normalize_team_columns(df))


def load_report_inputs(season: int, week: int, canonical_dir: str | Path) -> Dict[str, pd.DataFrame]:
    canonical_dir = Path(canonical_dir)

    games_path = canonical_dir / f"cr_games_{season}_w{week}.csv"
    edges_path = canonical_dir / f"cr_edges_{season}_w{week}.csv"
    markets_path = canonical_dir / f"cr_markets_{season}_w{week}.csv"
    parlays_path = canonical_dir / f"cr_parlay_scores_{season}_w{week}.csv"
    scores_path = canonical_dir / f"cr_scores_{season}_w{week}.csv"

    games = _normalize_games(_read_csv_if_exists(games_path))
    edges = _normalize_edges(_read_csv_if_exists(edges_path))
    markets = _normalize_markets(_read_csv_if_exists(markets_path))
    parlay_scores = _normalize_parlays(_read_csv_if_exists(parlays_path))
    scores = _normalize_scores(_read_csv_if_exists(scores_path))

    return {
        "games": games,
        "edges": edges,
        "markets": markets,
        "parlay_scores": parlay_scores,
        "scores": scores,
    }