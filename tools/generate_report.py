from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAVE_REPORTLAB = True
except Exception:
    HAVE_REPORTLAB = False

TZ = "America/New_York"
DEFAULT_SIM_BASIS = "50,000 Monte Carlo runs per game where model score inputs are available."
EDITION_TITLES = {
    "tnf": "THURSDAY NIGHT EDITION",
    "sunday_morning": "SUNDAY MORNING EDITION",
    "sunday_afternoon": "SUNDAY AFTERNOON EDITION",
    "snf": "SUNDAY NIGHT EDITION",
    "monday": "MONDAY EDITION",
    "tuesday": "TUESDAY WRAP EDITION",
}


@dataclass
class ReportPaths:
    exports_dir: Path
    reports_dir: Path
    games_path: Path
    edges_path: Path
    markets_path: Path
    scores_path: Path
    parlay_path: Path
    json_path: Path
    pdf_path: Path


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for up in [here.parent] + list(here.parents):
        if (up / "exports").exists():
            return up

    for env_name in ["EDGE_FINDER_ROOT", "EDGE_EXPORTS_DIR"]:
        env = os.environ.get(env_name, "").strip()
        if not env:
            continue
        candidate = Path(env)
        if env_name == "EDGE_EXPORTS_DIR":
            if candidate.exists() and candidate.name.lower() == "exports":
                return candidate.parent
        elif (candidate / "exports").exists():
            return candidate

    return Path.cwd()


def _pick_input_base(exports: Path) -> Path:
    canonical = exports / "canonical"
    if canonical.exists():
        expected = [
            "cr_games_",
            "cr_edges_",
            "cr_markets_",
            "cr_scores_",
            "cr_parlay_scores_",
        ]
        names = {p.name for p in canonical.glob("*.csv")}
        if any(any(name.startswith(prefix) for name in names) for prefix in expected):
            return canonical
    return exports


def resolve_paths(season: int, week: int, edition: str, reports_dir_arg: str | None = None) -> ReportPaths:
    root = _find_root()
    exports = root / "exports"
    input_base = _pick_input_base(exports)

    if reports_dir_arg:
        reports = Path(reports_dir_arg)
        if not reports.is_absolute():
            reports = root / reports
    else:
        reports = exports / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    return ReportPaths(
        exports_dir=input_base,
        reports_dir=reports,
        games_path=input_base / f"cr_games_{season}_w{week}.csv",
        edges_path=input_base / f"cr_edges_{season}_w{week}.csv",
        markets_path=input_base / f"cr_markets_{season}_w{week}.csv",
        scores_path=input_base / f"cr_scores_{season}_w{week}.csv",
        parlay_path=input_base / f"cr_parlay_scores_{season}_w{week}.csv",
        json_path=reports / f"{season}_week{week}_{edition}_3v1.json"
        pdf_path=reports / f"{season}_week{week}_{edition}_3v1.pdf"
    )

def load_game_projection_master(paths, season: int, week: int) -> pd.DataFrame:
    candidates = [
        paths.exports_dir / f"game_projection_master_{season}_w{week}.csv",
        paths.reports_dir.parent / "canonical" / f"game_projection_master_{season}_w{week}.csv",
        Path(__file__).resolve().parents[1] / "exports" / "canonical" / f"game_projection_master_{season}_w{week}.csv",
    ]

    for path in candidates:
        if path.exists():
            df = load_csv(path)
            if not df.empty:
                return df

    return pd.DataFrame()
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, low_memory=False)


def lc_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    low = lc_map(df)
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def first_existing(df: pd.DataFrame, candidates: list[str], default: Any = np.nan) -> pd.Series:
    col = find_col(df, candidates)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def to_num(s: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if np.isnan(default):
        return out
    return out.fillna(default)


def clip(series: pd.Series | float, lo: float, hi: float):
    if isinstance(series, pd.Series):
        return series.clip(lo, hi)
    return max(lo, min(hi, series))


def clean_team_name(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    aliases = {
        "football team": "Commanders",
        "washington football team": "Commanders",
        "wsn": "Commanders",
        "tb": "Buccaneers",
        "sf": "49ers",
        "ne": "Patriots",
        "kc": "Chiefs",
        "gb": "Packers",
        "det": "Lions",
        "ind": "Colts",
        "hou": "Texans",
        "mia": "Dolphins",
        "lar": "Rams",
        "la": "Rams",
        "lv": "Raiders",
        "no": "Saints",
        "nyj": "Jets",
        "nyg": "Giants",
        "phi": "Eagles",
        "pit": "Steelers",
        "bal": "Ravens",
        "cin": "Bengals",
        "dal": "Cowboys",
        "ari": "Cardinals",
        "atl": "Falcons",
        "car": "Panthers",
        "chi": "Bears",
        "cle": "Browns",
        "den": "Broncos",
        "jax": "Jaguars",
        "ten": "Titans",
        "sea": "Seahawks",
        "min": "Vikings",
        "lac": "Chargers",
    }
    return aliases.get(s.lower(), s)


def fmt_line(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    if abs(v - round(v)) < 1e-9:
        return f"{v:+.0f}"
    return f"{v:+.1f}"


def fmt_total(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.1f}".rstrip("0").rstrip(".")


def implied_prob_from_american(o: Any) -> float:
    try:
        v = float(o)
    except Exception:
        return math.nan
    if v > 0:
        return 100.0 / (v + 100.0)
    if v < 0:
        return abs(v) / (abs(v) + 100.0)
    return math.nan


def build_game_lookup(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["game_id", "game_label", "home_team", "away_team"])

    g = games.copy()
    g["home_team"] = first_existing(g, ["home_team", "home", "HomeTeam", "team_home"]).map(clean_team_name)
    g["away_team"] = first_existing(g, ["away_team", "away", "AwayTeam", "team_away"]).map(clean_team_name)

    fallback_label = first_existing(g, ["game_label", "matchup", "title"], default="").astype(str)
    g["game_label"] = np.where(
        g["away_team"].astype(str).str.len() > 0,
        g["away_team"].astype(str) + " @ " + g["home_team"].astype(str),
        fallback_label,
    )
    g["game_id"] = first_existing(g, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
    return g[["game_id", "game_label", "home_team", "away_team"]].drop_duplicates()


def normalize_market_name(x: Any) -> str:
    s = str(x or "").strip().lower()
    mapping = {
        "h2h": "moneyline",
        "ml": "moneyline",
        "moneyline": "moneyline",
        "money line": "moneyline",
        "spread": "spread",
        "spreads": "spread",
        "ats": "spread",
        "total": "total",
        "totals": "total",
        "game total": "total",
        "team total": "team_total",
        "team totals": "team_total",
    }
    return mapping.get(s, s)


def format_play_label(row: pd.Series) -> str:
    market = str(row.get("market_norm", row.get("market", ""))).strip().lower()
    side = clean_team_name(row.get("side_team", row.get("side", row.get("selection", ""))))
    line = row.get("line")
    total = row.get("total_line", row.get("line"))
    direction = str(row.get("direction", row.get("bet_direction", ""))).strip().title()
    player = str(row.get("player_name", row.get("player", ""))).strip()
    stat = str(row.get("prop_stat", row.get("stat_type", ""))).strip()

    if market in {"moneyline", "ml", "h2h"}:
        return f"{side} ML".strip()
    if market in {"spread", "spreads", "ats"}:
        return f"{side} {fmt_line(line)}".strip() if side else f"Spread {fmt_line(line)}".strip()
    if market in {"total", "totals", "game_total"}:
        if not direction:
            direction = "Over"
        return f"{direction} {fmt_total(total)}".strip()
    if market in {"team_total", "team totals", "team_total_points"}:
        if not direction:
            direction = "Over"
        base = f"{side} Team Total".strip()
        return f"{base} {direction} {fmt_total(total)}".strip()
    if market in {"player_prop", "prop", "player_total"}:
        bits = [player, direction, fmt_total(total), stat]
        return " ".join([b for b in bits if b]).strip()
    if side:
        return side
    return str(row.get("selection", row.get("label", "Unknown play")))


def enrich_edges(edges: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()

    out = edges.copy()
    out["game_id"] = first_existing(out, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
    out["market_norm"] = first_existing(out, ["market_norm", "market_type", "market"]).map(normalize_market_name)
    out["line"] = to_num(first_existing(out, ["line", "spread_line", "points", "handicap"]))
    out["total_line"] = to_num(first_existing(out, ["total_line", "points_line", "line"]))
    out["side_team"] = first_existing(out, ["team", "side_team", "selection_team", "pick_team", "side", "selection"]).map(clean_team_name)
    out["score"] = to_num(first_existing(out, ["score", "edge_score", "fort_knox_score", "fk_score", "rank_score","fort_knox", "edge_rank_score", "model_score", "composite_score"]))
    out["score"] = out["score"].fillna(out["confidence"])
    out["confidence"] = to_num(first_existing(out, ["confidence", "confidence_score", "conf", "grade_confidence"]))
    out["p_win"] = to_num(first_existing(out, ["p_win", "win_prob", "prob", "model_prob"]))
    out["market_odds"] = to_num(first_existing(out, ["odds", "price", "american_odds"]))
    out["implied_prob"] = out["market_odds"].map(implied_prob_from_american)
    out["edge_prob"] = out["p_win"] - out["implied_prob"]

    lookup = build_game_lookup(games)
    existing_game_label = "game_label" in out.columns
    existing_home = "home_team" in out.columns
    existing_away = "away_team" in out.columns

    if not lookup.empty:
        merge_cols = ["game_id", "game_label", "home_team", "away_team"]
        out = out.merge(lookup[merge_cols], on="game_id", how="left", suffixes=("", "_lk"))

        if existing_game_label:
            out["game_label"] = first_existing(out, ["game_label", "game_label_lk"], default="").astype(str)
        else:
            out["game_label"] = first_existing(out, ["game_label_lk"], default="").astype(str)

        if existing_home:
            out["home_team"] = first_existing(out, ["home_team", "home_team_lk"], default="").map(clean_team_name)
        else:
            out["home_team"] = first_existing(out, ["home_team_lk"], default="").map(clean_team_name)

        if existing_away:
            out["away_team"] = first_existing(out, ["away_team", "away_team_lk"], default="").map(clean_team_name)
        else:
            out["away_team"] = first_existing(out, ["away_team_lk"], default="").map(clean_team_name)

        drop_cols = [c for c in ["game_label_lk", "home_team_lk", "away_team_lk"] if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
    else:
        if "game_label" not in out.columns:
            out["game_label"] = first_existing(out, ["game_label", "matchup", "event_name"], default="").astype(str)
        if "home_team" not in out.columns:
            out["home_team"] = first_existing(out, ["home_team", "home"], default="").map(clean_team_name)
        if "away_team" not in out.columns:
            out["away_team"] = first_existing(out, ["away_team", "away"], default="").map(clean_team_name)

    out["play_label"] = out.apply(format_play_label, axis=1)
    out["board_label"] = np.where(
        out["game_label"].astype(str).str.len() > 0,
        out["game_label"].astype(str) + " | " + out["play_label"].astype(str),
        out["play_label"].astype(str),
    )
    return out


def _coalesce_num(df: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    return to_num(first_existing(df, candidates), default=default)

def enrich_games(games: pd.DataFrame, markets: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games.copy()

    g = games.copy()
    g["game_id"] = first_existing(g, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
    g["home_team"] = first_existing(g, ["home_team", "home", "HomeTeam", "team_home"]).map(clean_team_name)
    g["away_team"] = first_existing(g, ["away_team", "away", "AwayTeam", "team_away"]).map(clean_team_name)
    g["game_label"] = np.where(
        g["away_team"].astype(str).str.len() > 0,
        g["away_team"] + " @ " + g["home_team"],
        first_existing(g, ["game_label", "matchup"], default="").astype(str),
    )

    # defaults
    g["market_spread_raw"] = np.nan
    g["market_total"] = np.nan

    # -----------------------------
    # Pull explicit market columns
    # -----------------------------
    if not markets.empty:
        m = markets.copy()
        m["game_id"] = first_existing(m, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)

        spread_df = pd.DataFrame({
            "game_id": m["game_id"],
            "market_spread_raw": to_num(first_existing(m, ["market_spread", "spread", "spread_home", "closing_spread_home"]))
        }).dropna(subset=["market_spread_raw"]).drop_duplicates("game_id")

        total_df = pd.DataFrame({
            "game_id": m["game_id"],
            "market_total": to_num(first_existing(m, ["market_total", "total", "total_line", "closing_total"]))
        }).dropna(subset=["market_total"]).drop_duplicates("game_id")

        if not spread_df.empty:
            g = g.merge(spread_df, on="game_id", how="left", suffixes=("", "_m"))
            if "market_spread_raw_m" in g.columns:
                g["market_spread_raw"] = g["market_spread_raw"].fillna(g["market_spread_raw_m"])
                g = g.drop(columns=["market_spread_raw_m"])

        if not total_df.empty:
            g = g.merge(total_df, on="game_id", how="left", suffixes=("", "_m"))
            if "market_total_m" in g.columns:
                g["market_total"] = g["market_total"].fillna(g["market_total_m"])
                g = g.drop(columns=["market_total_m"])

    # -----------------------------
    # Optional upstream projected scores
    # -----------------------------
    g["home_score_mean"] = np.nan
    g["away_score_mean"] = np.nan

    if not scores.empty:
        s = scores.copy()
        s["game_id"] = first_existing(s, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)

        home_mean = to_num(first_existing(s, ["home_score_mean", "home_points_mean", "home_proj_score", "proj_home_score"]))
        away_mean = to_num(first_existing(s, ["away_score_mean", "away_points_mean", "away_proj_score", "proj_away_score"]))

        mean_df = pd.DataFrame({
            "game_id": s["game_id"],
            "home_score_mean": home_mean,
            "away_score_mean": away_mean,
        }).drop_duplicates("game_id")

        g = g.merge(mean_df, on="game_id", how="left", suffixes=("", "_s"))
        if "home_score_mean_s" in g.columns:
            g["home_score_mean"] = g["home_score_mean"].fillna(g["home_score_mean_s"])
            g = g.drop(columns=["home_score_mean_s"])
        if "away_score_mean_s" in g.columns:
            g["away_score_mean"] = g["away_score_mean"].fillna(g["away_score_mean_s"])
            g = g.drop(columns=["away_score_mean_s"])

    # -----------------------------
    # Stable fallback features
    # -----------------------------
    power_diff = _coalesce_num(g, ["power_diff", "team_power_diff", "elo_diff", "rating_diff", "home_power_minus_away_power"])
    recent_form_diff = _coalesce_num(g, ["recent_form_diff", "form_diff", "last5_diff", "rolling_form_diff", "home_form_minus_away_form"])
    qb_diff = _coalesce_num(g, ["qb_diff", "qb_edge", "home_qb_minus_away_qb"])
    pace_diff = _coalesce_num(g, ["pace_diff", "plays_diff", "tempo_diff"])
    home_field_pts = _coalesce_num(g, ["home_field_pts", "hfa_pts", "home_field_advantage"], default=1.75)
    home_field_pts = home_field_pts.where(home_field_pts != 0, 1.75)

    market_spread = to_num(g["market_spread_raw"])
    market_total = to_num(g["market_total"])

    # -----------------------------
    # Margin model
    # positive = home edge
    # -----------------------------
    raw_margin = (
        0.70 * market_spread.fillna(0.0)
        + 0.45 * power_diff
        + 0.30 * recent_form_diff
        + 0.20 * qb_diff
        + 0.08 * pace_diff
        + home_field_pts
    )

    shrunk_margin = np.where(
        market_spread.notna(),
        0.75 * market_spread + 0.25 * raw_margin,
        raw_margin,
    )
    g["raw_proj_margin"] = raw_margin
    g["shrunk_proj_margin"] = pd.Series(shrunk_margin, index=g.index).clip(-17.0, 17.0)

    # -----------------------------
    # Total model
    # -----------------------------
    derived_total = (
        market_total.fillna(41.5)
        + 0.60 * pace_diff
        + 0.12 * recent_form_diff.abs()
    )

    if g["home_score_mean"].notna().any() and g["away_score_mean"].notna().any():
        mean_total = g["home_score_mean"].fillna(21.0) + g["away_score_mean"].fillna(20.0)
        raw_total = 0.55 * market_total.fillna(41.5) + 0.45 * mean_total
    else:
        raw_total = derived_total

    g["raw_proj_total"] = raw_total
    g["shrunk_proj_total"] = pd.Series(
        np.where(
            market_total.notna(),
            0.78 * market_total + 0.22 * raw_total,
            raw_total
        ),
        index=g.index
    ).clip(30.0, 62.0)

    # -----------------------------
    # Convert total + margin to scores
    # -----------------------------
    g["proj_home_points"] = ((g["shrunk_proj_total"] + g["shrunk_proj_margin"]) / 2.0).clip(lower=10.0)
    g["proj_away_points"] = ((g["shrunk_proj_total"] - g["shrunk_proj_margin"]) / 2.0).clip(lower=10.0)

    # small asymmetry to prevent every game rounding to 21-20
    g["proj_home_points"] = g["proj_home_points"] + 0.20 * home_field_pts + 0.10 * qb_diff.clip(lower=0)
    g["proj_away_points"] = g["proj_away_points"] - 0.05 * home_field_pts - 0.10 * qb_diff.clip(upper=0).abs()

    g["proj_home_points"] = g["proj_home_points"].round(1)
    g["proj_away_points"] = g["proj_away_points"].round(1)
    g["median_home_points"] = g["proj_home_points"].round().astype(int)
    g["median_away_points"] = g["proj_away_points"].round().astype(int)

    return g

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
def _safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _fmt_spread_for_home(home_team: str, spread_val: Any) -> str:
    v = _safe_float(spread_val)
    if pd.isna(v):
        return ""
    return f"{home_team} {fmt_line(v)}"


def _best_edge_per_game(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()

    tmp = edges.copy()
    sort_cols = [c for c in ["score", "confidence"] if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols, ascending=False)
    if "game_id" not in tmp.columns:
        return tmp.head(0)
    return tmp.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)


def _edge_reason_text(r: pd.Series) -> str:
    market = str(r.get("market_norm", "")).lower()
    play_label = str(r.get("play_label", r.get("board_label", "Top play")))
    conf = _safe_float(r.get("confidence", np.nan))
    score = _safe_float(r.get("score", np.nan))
    edge_prob = _safe_float(r.get("edge_prob", np.nan))

    if pd.notna(edge_prob):
        prob_text = f"model edge vs implied price is {edge_prob * 100:+.1f} pts"
    else:
        prob_text = "model support is positive"

    if market == "moneyline":
        why = "straight-up win probability is stronger than the market price"
    elif market == "spread":
        why = "projected margin clears the number"
    elif market == "total":
        why = "projected scoring environment points the same way as the play"
    elif market == "team_total":
        why = "projected team scoring sits on the favorable side of the posted number"
    else:
        why = "the model stack ranks this as the strongest game angle"

    conf_txt = f"Confidence {conf:.0f}" if pd.notna(conf) else "Confidence unavailable"
    score_txt = f"Fort Knox Score {score:.2f}" if pd.notna(score) else "Fort Knox Score unavailable"

    return f"Simulation Value Play: {play_label}. {conf_txt}; {score_txt}. {why}, and {prob_text}."


def _game_projection_text(g: pd.Series) -> str:
    home = str(g.get("home_team", "Home"))
    away = str(g.get("away_team", "Away"))

    median_home = g.get("median_home_points", np.nan)
    median_away = g.get("median_away_points", np.nan)
    proj_margin = _safe_float(g.get("shrunk_proj_margin", np.nan))
    proj_total = _safe_float(g.get("shrunk_proj_total", np.nan))

    parts: list[str] = []

    if pd.notna(median_home) and pd.notna(median_away):
        parts.append(f"Model median: {home} {int(median_home)}, {away} {int(median_away)}.")

    if pd.notna(proj_margin):
        favorite = home if proj_margin > 0 else away
        margin_abs = abs(proj_margin)
        if margin_abs >= 10:
            parts.append(f"Projected script: {favorite} in clear control.")
        elif margin_abs >= 7:
            parts.append(f"Projected script: {favorite} by about a touchdown.")
        elif margin_abs >= 4:
            parts.append(f"Projected script: {favorite} by more than one possession.")
        elif margin_abs >= 2:
            parts.append(f"Projected script: {favorite} in a one-score control game.")
        else:
            parts.append(f"Projected script: tight game with a late lean.")

    if pd.notna(proj_total):
        parts.append(f"Projected total: {proj_total:.1f}.")

    return " ".join(parts).strip()


def _parlay_pick_label_from_edge(r: pd.Series) -> str:
    play_label = str(r.get("play_label", "")).strip()
    if play_label:
        return play_label

    board_label = str(r.get("board_label", "")).strip()
    if " | " in board_label:
        return board_label.split(" | ", 1)[1].strip()
    if board_label:
        return board_label

    return str(r.get("label", "Pick unavailable")).strip() or "Pick unavailable"

def build_game_script(game_row: pd.Series, edges_for_game: pd.DataFrame) -> str:
    home = str(game_row.get("home_team", "Home"))
    away = str(game_row.get("away_team", "Away"))

    market_spread = _safe_float(game_row.get("market_spread_home", game_row.get("market_spread_raw", np.nan)))
    market_total = _safe_float(game_row.get("market_total", np.nan))

    proj_home = _safe_float(game_row.get("proj_home_points", np.nan))
    proj_away = _safe_float(game_row.get("proj_away_points", np.nan))
    proj_margin = _safe_float(game_row.get("proj_margin_home", game_row.get("shrunk_proj_margin", np.nan)))
    proj_total = _safe_float(game_row.get("proj_total", game_row.get("shrunk_proj_total", np.nan)))

    intro = f"{away} @ {home}."

    market_bits = []
    if pd.notna(market_spread):
        market_bits.append(f"Spread: {home} {fmt_line(market_spread)}")
    if pd.notna(market_total):
        market_bits.append(f"Total: {fmt_total(market_total)}")

    market_text = ""
    if market_bits:
        market_text = "Market: " + " | ".join(market_bits) + "."

    # Only trust projection text if it is not obviously a flat fallback.
    projection_is_trustworthy = True

    if pd.notna(proj_home) and pd.notna(proj_away):
        if abs(proj_home - 21.6) < 0.11 and abs(proj_away - 19.9) < 0.11:
            projection_is_trustworthy = False

    if pd.notna(proj_total):
        if abs(proj_total - 41.5) < 0.11:
            projection_is_trustworthy = False

    projection_text = ""
    if projection_is_trustworthy:
        proj_bits = []

        if pd.notna(proj_home) and pd.notna(proj_away):
            proj_bits.append(f"Model projection: {home} {proj_home:.1f}, {away} {proj_away:.1f}.")

        if pd.notna(proj_margin):
            fav = home if proj_margin > 0 else away
            margin_abs = abs(proj_margin)
            if margin_abs >= 10:
                proj_bits.append(f"Projected script: {fav} in clear control.")
            elif margin_abs >= 7:
                proj_bits.append(f"Projected script: {fav} by about a touchdown.")
            elif margin_abs >= 4:
                proj_bits.append(f"Projected script: {fav} by more than one possession.")
            elif margin_abs >= 2:
                proj_bits.append(f"Projected script: {fav} in a one-score control game.")
            else:
                proj_bits.append("Projected script: tight game with a late lean.")

        if pd.notna(proj_total):
            proj_bits.append(f"Projected total: {proj_total:.1f}.")

        projection_text = " ".join(proj_bits).strip()

    if edges_for_game.empty:
        reason = "No ranked edge survived filtering for this matchup."
    else:
        sort_cols = [c for c in ["score", "confidence"] if c in edges_for_game.columns]
        top = edges_for_game.sort_values(sort_cols, ascending=False).iloc[0] if sort_cols else edges_for_game.iloc[0]

        play_label = str(top.get("play_label", top.get("board_label", "Top play"))).strip()
        market = str(top.get("market_norm", "")).lower()
        conf = _safe_float(top.get("confidence", np.nan))
        score = _safe_float(top.get("score", np.nan))
        edge_prob = _safe_float(top.get("edge_prob", np.nan))

        conf_text = f"Confidence {conf:.0f}" if pd.notna(conf) else "Confidence unavailable"
        score_text = f"Fort Knox Score {score:.2f}" if pd.notna(score) else "Fort Knox Score unavailable"
        edge_text = f"Model edge vs implied price: {edge_prob * 100:+.1f} pts." if pd.notna(edge_prob) else "Model edge vs implied price unavailable."

        if market == "moneyline":
            why = "Straight-up win profile is the strongest angle on this game."
        elif market == "spread":
            why = "Spread is the strongest ranked angle on this game."
        elif market == "total":
            why = "Total is the strongest ranked angle on this game."
        elif market == "team_total":
            why = "Team total is the strongest ranked angle on this game."
        else:
            why = "This is the strongest ranked angle on this game."

        reason = f"Simulation Value Play: {play_label}. {conf_text}; {score_text}. {why} {edge_text}"

    parts = [intro, market_text, projection_text, reason]
    return " ".join([p for p in parts if p]).strip()

    # If favorite conflicts with medians, nudge medians to match the actual top edge.
    if favorite == home and median_home <= median_away:
        median_home = median_away + 1
    elif favorite == away and median_away <= median_home:
        median_away = median_home + 1

    favorite = home if proj_margin > 0 else away
    margin_abs = abs(proj_margin)

    if margin_abs >= 7:
        margin_text = f"{favorite} by about a touchdown or better"
    elif margin_abs >= 4:
        margin_text = f"{favorite} by more than one clean possession"
    elif margin_abs >= 2:
        margin_text = f"{favorite} in a one-score control script"
    else:
        margin_text = "a tight game that leans late toward the stronger side"

    if pd.notna(market_total):
        total_delta = proj_total - float(market_total)
    else:
        total_delta = 0.0

    if total_delta >= 2.0:
        total_text = "The scoring environment leans over the market."
    elif total_delta <= -2.0:
        total_text = "The scoring environment leans under the market."
    else:
        total_text = "Total projection is close to the market, so side quality matters more than total drift."

    if pd.notna(market_spread):
        line_text = f"{home} {fmt_line(market_spread)}"
    else:
        line_text = "market spread unavailable"

    if pd.notna(market_total):
        total_line_text = fmt_total(market_total)
    else:
        total_line_text = "market total unavailable"

    if edges_for_game.empty:
        angle_text = "Closest look exists, but no tracked edge row survived ranking for this matchup."
    else:
        sort_cols = [c for c in ["score", "confidence"] if c in edges_for_game.columns]
        top = edges_for_game.sort_values(sort_cols, ascending=False).iloc[0] if sort_cols else edges_for_game.iloc[0]
        play_label = str(top.get("play_label", "Top angle"))
        eprob = top.get("edge_prob", np.nan)
        score = top.get("score", np.nan)
        conf = top.get("confidence", np.nan)
        market = str(top.get("market_norm", "")).lower()

        if pd.notna(eprob):
            prob_text = f"model edge of {round(float(eprob) * 100.0, 1):+.1f} percentage points versus implied price"
        else:
            prob_text = "support from the model stack"

        if market == "moneyline":
            angle_text = f"Best straight angle is {play_label} on a {prob_text}; Fort Knox Score {float(score):.2f}, confidence {float(conf):.0f}."
        elif market == "spread":
            angle_text = f"Best spread look is {play_label}; projected margin and recent-form inputs clear the number with a {prob_text}."
        elif market == "total":
            angle_text = f"Best total look is {play_label}; pace, scoring environment, and distribution shape all point the same way."
        elif market == "team_total":
            angle_text = f"Best derivative is {play_label}; projected team scoring sits on the favorable side of the posted number."
        else:
            angle_text = f"Best tracked angle is {play_label} with {prob_text}."

    return (
        f"{away} @ {home}. Line: {line_text} | Total {total_line_text}. "
        f"Model medians: {home} {median_home}, {away} {median_away}. "
        f"Projected script points to {margin_text}. {total_text} {angle_text}"
    )
def normalize_matchup_text(x: Any) -> str:
    s = str(x or "").strip().upper()
    s = s.replace("VS.", "@").replace("VS", "@")
    s = " ".join(s.split())
    return s


def build_matchup_edge_lookup(edges: pd.DataFrame, games: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    if edges.empty or games.empty:
        return out

    g = build_game_lookup(games).copy()
    if g.empty:
        return out

    # full-name map
    g["matchup_full"] = g["game_label"].astype(str).map(normalize_matchup_text)

    # abbreviation map
    def abbrev(name: str) -> str:
        name = clean_team_name(name)
        mapping = {
            "49ers": "SF",
            "Buccaneers": "TB",
            "Patriots": "NE",
            "Chiefs": "KC",
            "Packers": "GB",
            "Lions": "DET",
            "Colts": "IND",
            "Texans": "HOU",
            "Dolphins": "MIA",
            "Rams": "LAR",
            "Raiders": "LV",
            "Saints": "NO",
            "Jets": "NYJ",
            "Giants": "NYG",
            "Eagles": "PHI",
            "Steelers": "PIT",
            "Ravens": "BAL",
            "Bengals": "CIN",
            "Cowboys": "DAL",
            "Cardinals": "ARI",
            "Falcons": "ATL",
            "Panthers": "CAR",
            "Bears": "CHI",
            "Browns": "CLE",
            "Broncos": "DEN",
            "Jaguars": "JAX",
            "Titans": "TEN",
            "Seahawks": "SEA",
            "Vikings": "MIN",
            "Chargers": "LAC",
            "Bills": "BUF",
            "Commanders": "WAS",
        }
        return mapping.get(name, name[:3].upper())

    g["matchup_abbr"] = g["away_team"].map(abbrev) + " @ " + g["home_team"].map(abbrev)
    g["matchup_abbr"] = g["matchup_abbr"].map(normalize_matchup_text)

    e = edges.copy()
    sort_cols = [c for c in ["score", "confidence"] if c in e.columns]
    e = e.sort_values(sort_cols, ascending=False) if sort_cols else e

    merged = e.merge(g[["game_id", "matchup_full", "matchup_abbr"]], on="game_id", how="left")
    for _, row in merged.iterrows():
        label = str(row.get("play_label", "")).strip()
        if not label:
            continue
        full = str(row.get("matchup_full", "")).strip()
        abbr = str(row.get("matchup_abbr", "")).strip()
        if full and full not in out:
            out[full] = label
        if abbr and abbr not in out:
            out[abbr] = label

    return out
def enrich_parlays(parlays: pd.DataFrame, edges: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if parlays.empty:
        return parlays.copy()
    matchup_edge_lookup = build_matchup_edge_lookup(edges, games)

    p = parlays.copy()
    p["parlay_score"] = to_num(first_existing(p, ["parlay_score", "score", "portfolio_score", "fk_score"]), default=np.nan)
    p["p_win"] = to_num(first_existing(p, ["p_win", "win_prob", "prob", "model_prob"]), default=np.nan)
    p["confidence_raw"] = to_num(first_existing(p, ["confidence", "confidence_score", "conf"]), default=np.nan)

    p["parlay_score"] = np.where(
        p["parlay_score"].notna(),
        p["parlay_score"],
        np.where(p["p_win"].notna(), (p["p_win"] * 100.0).round(2), 55.0),
    )

    game_lookup = build_game_lookup(games)

    edge_lookup: dict[str, list[pd.Series]] = {}
    if not edges.empty:
        tmp = edges.copy()
        gid_col = find_col(tmp, ["game_id"])
        if gid_col is not None:
            sort_cols = [c for c in ["score", "confidence"] if c in tmp.columns]
            tmp = tmp.sort_values(sort_cols, ascending=False) if sort_cols else tmp
            for _, r in tmp.iterrows():
                edge_lookup.setdefault(str(r[gid_col]), []).append(r)

    leg_texts = []
    leg_counts = []
    correlations = []
    confidences = []

    leg_game_cols = [c for c in p.columns if c.lower() in {
        "game_id_1", "game_id_2", "game_id_3",
        "leg1_game_id", "leg2_game_id", "leg3_game_id"
    }]

    leg_label_cols = [c for c in p.columns if c.lower() in {
        "leg_1_label", "leg_2_label", "leg_3_label",
        "leg1_label", "leg2_label", "leg3_label"
    }]

    generic_label_col = find_col(p, ["label"])
    matchup_col = find_col(p, ["matchup"])
    market_col = find_col(p, ["market"])
    side_col = find_col(p, ["side"])
    line_col = find_col(p, ["line"])

    def build_single_row_label(row: pd.Series) -> str:
        if generic_label_col is not None:
            v = str(row.get(generic_label_col, "")).strip()
            if v and v.lower() != "nan":
                return v

        matchup = str(row.get(matchup_col, "")).strip() if matchup_col else ""
        market = str(row.get(market_col, "")).strip().lower() if market_col else ""
        side = clean_team_name(row.get(side_col, "")) if side_col else ""
        line = row.get(line_col, np.nan) if line_col else np.nan

        if market in {"moneyline", "ml", "h2h"} and side:
            return f"{matchup} | {side} ML" if matchup else f"{side} ML"
        if market in {"spread", "spreads", "ats"} and side:
            return f"{matchup} | {side} {fmt_line(line)}" if matchup else f"{side} {fmt_line(line)}"
        if market in {"total", "totals", "game_total"} and side:
            side_txt = str(side).title()
            return f"{matchup} | {side_txt} {fmt_total(line)}" if matchup else f"{side_txt} {fmt_total(line)}"

        if matchup:
            return f"{matchup} | Parlay leg"
        return "Parlay leg"

    for _, row in p.iterrows():
        labels: list[str] = []

        # 1. Explicit leg label columns
        for c in leg_label_cols:
            v = str(row.get(c, "")).strip()
            if v and v.lower() != "nan":
                labels.append(v)

        # 2. Linked game ids -> use strongest edge for that game
        if not labels:
            for c in leg_game_cols:
                gid = str(row.get(c, "")).strip()
                if not gid or gid.lower() == "nan":
                    continue

                picks = edge_lookup.get(gid, [])
                if picks:
                    best = picks[0]
                    label = str(best.get("play_label", "")).strip()
                    if not label:
                        label = str(best.get("board_label", "")).strip()
                    if label:
                        labels.append(label)
                        continue

                hit = game_lookup.loc[game_lookup["game_id"] == gid]
                if not hit.empty:
                    game_label = str(hit.iloc[0]["game_label"]).strip()
                    if game_label:
                        labels.append(f"{game_label} ML")

        # 3. If parlay file is one row per leg, build from row fields directly
        if not labels:
            matchup = str(row.get(matchup_col, "")).strip() if matchup_col else ""
            matchup_norm = normalize_matchup_text(matchup)

            if matchup_norm in matchup_edge_lookup:
                labels.append(matchup_edge_lookup[matchup_norm])
            else:
                market = str(row.get(market_col, "")).strip().lower() if market_col else ""
                side = clean_team_name(row.get(side_col, "")) if side_col else ""
                line = row.get(line_col, np.nan) if line_col else np.nan

                if market in {"moneyline", "ml", "h2h"} and side:
                    labels.append(f"{side} ML")
                elif market in {"spread", "spreads", "ats"} and side:
                    labels.append(f"{side} {fmt_line(line)}")
                elif market in {"total", "totals", "game_total"} and side:
                    labels.append(f"{str(side).title()} {fmt_total(line)}")
                elif matchup:
                    labels.append(matchup)

        # 4. Raw text fallback
        if not labels:
            for idx in range(1, 7):
                for base in [f"leg_{idx}", f"leg{idx}", f"pick_{idx}", f"pick{idx}"]:
                    if base in p.columns:
                        v = str(row.get(base, "")).strip()
                        if v and v.lower() != "nan":
                            labels.append(v)

        labels = [x for x in labels if x and x.lower() != "nan"]

        if not labels:
            labels = ["Pick unavailable"]

        leg_counts.append(len(labels))
        leg_texts.append(labels)

        same_game = len({lbl.split(" | ")[0] for lbl in labels}) < len(labels) if labels else False
        if same_game:
            correlations.append("same-game or tightly linked structure")
            corr_penalty = 4.0
        else:
            correlations.append("cross-game mix with limited structural overlap")
            corr_penalty = 0.0

        score = float(row.get("parlay_score", np.nan)) if pd.notna(row.get("parlay_score", np.nan)) else np.nan
        pwin = float(row.get("p_win", np.nan)) if pd.notna(row.get("p_win", np.nan)) else np.nan
        raw_conf = float(row.get("confidence_raw", np.nan)) if pd.notna(row.get("confidence_raw", np.nan)) else np.nan

        if pd.notna(score) and pd.notna(pwin):
            conf = 35.0 + 0.55 * score + 28.0 * pwin - corr_penalty
        elif pd.notna(score):
            conf = 38.0 + 0.60 * score - corr_penalty
        elif pd.notna(pwin):
            conf = 30.0 + 40.0 * pwin - corr_penalty
        elif pd.notna(raw_conf):
            conf = raw_conf
        else:
            conf = 52.0 - corr_penalty

        confidences.append(float(clip(conf, 25.0, 92.0)))
    p["leg_labels"] = leg_texts
    p["leg_count"] = leg_counts
    p["correlation_note"] = correlations
    p["confidence"] = confidences
    return p
def build_edge_summary(r: pd.Series) -> str:
    label = str(r.get("board_label", r.get("play_label", "Play")))
    conf = float(r.get("confidence", 0.0) or 0.0)
    score = float(r.get("score", 0.0) or 0.0)
    band = confidence_band(conf)
    market = str(r.get("market_norm", "")).lower()
    edge_prob = r.get("edge_prob", np.nan)

    if pd.notna(edge_prob):
        prob_phrase = f"Model edge vs implied price: {float(edge_prob) * 100:+.1f} pts."
    else:
        prob_phrase = "Model support is positive but incomplete on price-vs-prob columns."

    if market == "moneyline":
        why = "Straight-up win profile grades stronger than the market price."
    elif market == "spread":
        why = "Projected margin clears the number with support from form and home/road context."
    elif market == "total":
        why = "Scoring environment and pace point in the same direction."
    elif market == "team_total":
        why = "Projected team scoring sits on the favorable side of the posted derivative."
    else:
        why = "Board support is stronger than the average available angle."

    return f"{label}. Confidence {conf:.0f} — {band}. {why} {prob_phrase} Fort Knox Score {score:.2f}."


def section_top_edges(edges: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if edges.empty:
        return []
    sort_cols = [c for c in ["score", "confidence"] if c in edges.columns]
    tmp = edges.sort_values(sort_cols, ascending=False).head(limit).copy() if sort_cols else edges.head(limit).copy()
    rows = []
    for _, r in tmp.iterrows():
        rows.append(
            {
                "rank": len(rows) + 1,
                "game_label": str(r.get("game_label", "")),
                "play_label": str(r.get("play_label", "")),
                "board_label": str(r.get("board_label", "")),
                "confidence": round(float(r.get("confidence", 0.0)), 2),
                "band": confidence_band(float(r.get("confidence", 0.0))),
                "score": round(float(r.get("score", 0.0)), 2),
                "summary": build_edge_summary(r),
            }
        )
    return rows


def section_heatmap(edges: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    if edges.empty:
        return []
    sort_cols = [c for c in ["score", "confidence"] if c in edges.columns]
    tmp = edges.sort_values(sort_cols, ascending=False).head(limit).copy() if sort_cols else edges.head(limit).copy()
    return [
        {
            "board_label": str(r.get("board_label", "")),
            "band": confidence_band(float(r.get("confidence", 0.0))),
            "score": round(float(r.get("score", 0.0)), 2),
        }
        for _, r in tmp.iterrows()
    ]


def section_scripts(games: pd.DataFrame, edges: pd.DataFrame, proj_master: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    if games.empty:
        return []

    rows = []
    sort_key = "game_label" if "game_label" in games.columns else "game_id"

    edge_map: dict[str, pd.DataFrame] = {}
    if not edges.empty and "game_id" in edges.columns:
        for gid, sub in edges.groupby("game_id"):
            edge_map[str(gid)] = sub.copy()

    proj_map: dict[str, pd.Series] = {}
    if proj_master is not None and not proj_master.empty and "game_id" in proj_master.columns:
        for _, r in proj_master.iterrows():
            proj_map[str(r.get("game_id", ""))] = r

    for _, g in games.sort_values(sort_key).iterrows():
        gid = str(g.get("game_id", ""))
        e = edge_map.get(gid, pd.DataFrame())

        # prefer projection-master row if present
        src = proj_map.get(gid, g)

        rows.append(
            {
                "game_label": str(
                    src.get(
                        "matchup",
                        g.get("game_label", f"{g.get('away_team', '')} @ {g.get('home_team', '')}")
                    )
                ).strip(),
                "script": build_game_script(src, e),
            }
        )

    return rows


def build_parlay_summary(legs: list[str], conf: float, pwin: Any, score: Any, corr: str) -> str:
    score_txt = f"Composite score {float(score):.2f}" if pd.notna(score) else "Composite score unavailable"
    pwin_txt = f"joint win estimate {float(pwin) * 100:.1f}%" if pd.notna(pwin) else "joint win estimate unavailable"
    legs_txt = "; ".join(legs[:3]) if legs else "legs unavailable"
    return (
        f"{len(legs)}-leg suggested mix. {score_txt}; {pwin_txt}; {corr}. "
        f"Legs: {legs_txt}."
    )


def section_parlays(parlays: pd.DataFrame, limit: int = 3, edges: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Preferred path: build suggested parlays directly from top explicit edge labels.
    if edges is not None and not edges.empty:
        e = edges.copy()
        sort_cols = [c for c in ["score", "confidence"] if c in e.columns]
        e = e.sort_values(sort_cols, ascending=False) if sort_cols else e

        # Keep strongest distinct picks by game.
        if "game_id" in e.columns:
            e = e.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)

        top = e.head(9).copy()
        picks = []
        for _, r in top.iterrows():
            label = _parlay_pick_label_from_edge(r)
            if label and label.lower() != "pick unavailable":
                picks.append(
                    {
                        "label": label,
                        "confidence": _safe_float(r.get("confidence", np.nan)),
                        "score": _safe_float(r.get("score", np.nan)),
                        "p_win": _safe_float(r.get("p_win", np.nan)),
                    }
                )

        # Build 3 suggested mixes from top explicit plays.
        recipes = [
            picks[0:2],
            picks[0:3],
            picks[1:3],
        ]

        rank = 1
        for combo in recipes:
            combo = [x for x in combo if x]
            if not combo:
                continue

            legs = [x["label"] for x in combo]
            conf_vals = [x["confidence"] for x in combo if pd.notna(x["confidence"])]
            score_vals = [x["score"] for x in combo if pd.notna(x["score"])]
            pwin_vals = [x["p_win"] for x in combo if pd.notna(x["p_win"])]

            conf = float(np.mean(conf_vals)) if conf_vals else 55.0
            score = float(np.mean(score_vals)) if score_vals else np.nan

            if pwin_vals:
                joint = 1.0
                for p in pwin_vals:
                    joint *= max(0.01, min(0.99, float(p)))
                pwin = joint
            else:
                pwin = np.nan

            corr = "cross-game mix built from the highest-ranked explicit plays"
            rows.append(
                {
                    "rank": rank,
                    "legs": legs,
                    "confidence": round(conf, 2),
                    "band": confidence_band(conf),
                    "summary": build_parlay_summary(legs, conf, pwin, score, corr),
                }
            )
            rank += 1
            if rank > limit:
                break

        if rows:
            return rows

    # Fallback path: use existing parlay rows if explicit legs already exist.
    if parlays.empty:
        return []

    tmp = parlays.copy()
    sort_cols = [c for c in ["confidence", "parlay_score", "p_win"] if c in tmp.columns]
    tmp = tmp.sort_values(sort_cols, ascending=False).head(limit).copy() if sort_cols else tmp.head(limit).copy()

    for i, (_, r) in enumerate(tmp.iterrows(), start=1):
        legs = list(r.get("leg_labels", []))
        if not legs:
            raw = str(r.get("label", "")).strip()
            legs = [raw] if raw else ["Pick unavailable"]

        conf = float(r.get("confidence", 55.0) or 55.0)
        rows.append(
            {
                "rank": i,
                "legs": legs,
                "confidence": round(conf, 2),
                "band": confidence_band(conf),
                "summary": build_parlay_summary(
                    legs,
                    conf,
                    r.get("p_win", np.nan),
                    r.get("parlay_score", np.nan),
                    "fallback portfolio output",
                ),
            }
        )
    return rows


def build_report_payload(season: int, week: int, edition: str, as_of: Optional[str], edges: pd.DataFrame, games: pd.DataFrame, parlays: pd.DataFrame) -> dict[str, Any]:
    return {
        "title": f"CALCULATED RISK™ · EDGE FACTOR™ · 3v1™\nFORT KNOX — WEEK {week} {EDITION_TITLES.get(edition, edition.upper())}",
        "meta": {
            "week": week,
            "season": season,
            "edition": edition,
            "as_of": as_of,
            "prepared_by": "Doc Odds — Educational & Entertainment Use Only",
            "confidence_bands": [
                "Gold (90-95): highest-confidence board only",
                "Dark Green (75-89): strong edge with solid support",
                "Green (60-74): playable edge",
                "Yellow (45-59): lean / context play",
                "Amber (25-44): weak support",
                "Red (<25): avoid / informational only",
            ],
            "simulation_basis": DEFAULT_SIM_BASIS,
            "section_notes": {
                "top_edges": "Ranked by Fort Knox Score; confidence reflects edge quality plus stability.",
                "heatmap": "Color shows confidence band, not stake size.",
                "parlays": "Parlay confidence reflects joint profile and structure quality, not guaranteed hit rate.",
            },
        },
        "sections": {
            "top_edges": section_top_edges(edges),
            "heatmap": section_heatmap(edges),
            "game_scripts": section_scripts(games, edges, proj_master=parlays if False else None),
            "parlays": section_parlays(parlays, edges=edges),
        },
        "appendix": {
            "edge_count": int(len(edges)),
            "game_count": int(len(games)),
            "parlay_count": int(len(parlays)),
            "notes": [
                "Top board is ranked by Fort Knox Score, blending edge, confidence, disagreement, synergy, weather, and stability.",
                "Narratives are generated from canonical week files but now key off explicit play labels and projected margin/total logic.",
                "This report is for educational and entertainment purposes only.",
            ],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def wrap_text(text: str, max_chars: int = 105) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        test = cur + " " + w
        if len(test) <= max_chars:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def render_pdf(path: Path, payload: dict[str, Any]) -> None:
    if not HAVE_REPORTLAB:
        raise RuntimeError("reportlab is not installed, so PDF output could not be rendered")

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    left = 42
    top = height - 42
    y = top

    def new_page():
        nonlocal y
        c.showPage()
        y = top

    def draw_line(text: str, size: int = 10, bold: bool = False, gap: int = 14):
        nonlocal y
        font = "Helvetica-Bold" if bold else "Helvetica"
        if y < 54:
            new_page()
        c.setFont(font, size)
        c.drawString(left, y, text)
        y -= gap

    for idx, line in enumerate(payload["title"].split("\n")):
        draw_line(line, size=13 if idx == 0 else 14, bold=True, gap=16)

    meta = payload["meta"]
    for line in [
        f"Week: {meta['week']}",
        f"Edition: {meta['edition']}",
        f"As Of: {meta['as_of']}",
        f"Prepared by: {meta['prepared_by']}",
        "",
        "HOW TO READ THE CONFIDENCE BANDS",
    ]:
        draw_line(line, size=10, bold=(line == "HOW TO READ THE CONFIDENCE BANDS"), gap=14)

    for line in meta["confidence_bands"] + [f"Simulation basis: {meta['simulation_basis']}"]:
        for chunk in wrap_text(line):
            draw_line(chunk, size=10, gap=12)

    draw_line("", gap=10)
    draw_line("SECTION 1 — TOP EDGES", size=11, bold=True, gap=16)
    for row in payload["sections"]["top_edges"]:
        draw_line(f"{row['rank']}) {row['board_label']} | Confidence: {row['confidence']} | Score: {row['score']}", size=10, gap=13)
        for chunk in wrap_text(row["summary"], max_chars=100):
            draw_line(chunk, size=10, gap=12)
        draw_line("", gap=8)

    draw_line("SECTION 2 — MODEL PROJECTION BOARD", size=11, bold=True, gap=16)
    for row in payload["sections"]["heatmap"]:
        draw_line(f"{row['board_label']} | {row['band']} | Score {row['score']}", size=10, gap=12)

    draw_line("", gap=10)
    draw_line("SECTION 3 — SIMULATION VALUE BOARD", size=11, bold=True, gap=16)
    for row in payload["sections"]["game_scripts"]:
        draw_line(row["game_label"], size=10, bold=True, gap=12)
        for chunk in wrap_text(row["script"], max_chars=100):
            draw_line(chunk, size=10, gap=12)
        draw_line("", gap=8)

    draw_line("SECTION 4 — SUGGESTED PARLAYS", size=11, bold=True, gap=16)
    for row in payload["sections"]["parlays"]:
        draw_line(f"Parlay #{row['rank']}", size=10, bold=True, gap=12)
        for leg in row["legs"]:
            draw_line(f"- {leg}", size=10, gap=12)
        draw_line(f"Confidence: {row['confidence']} ({row['band']})", size=10, gap=12)
        for chunk in wrap_text(row["summary"], max_chars=100):
            draw_line(chunk, size=10, gap=12)
        draw_line("", gap=8)

    draw_line("APPENDIX", size=11, bold=True, gap=16)
    for key, value in payload["appendix"].items():
        if key == "notes":
            continue
        draw_line(f"{key}: {value}", size=10, gap=12)
    for note in payload["appendix"]["notes"]:
        for chunk in wrap_text(f"- {note}", max_chars=100):
            draw_line(chunk, size=10, gap=12)

    c.save()


def detect_as_of(games: pd.DataFrame, edges: pd.DataFrame) -> Optional[str]:
    for df in [games, edges]:
        for col in ["as_of", "snapshot_time", "updated_at", "created_at", "run_ts", "timestamp"]:
            if col in df.columns:
                vals = df[col].dropna().astype(str)
                if not vals.empty:
                    return vals.iloc[0]
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuilt Fort Knox 3v1 report generator with explicit play labels and better scripts.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--edition", type=str, default="monday")
    parser.add_argument("--reports-dir", type=str, default=None, help=r"Optional override for output report folder, e.g. exports\reports_test8")
    parser.add_argument("--write-back-labels", action="store_true", help="Also stamp play_label/board_label back into the weekly edges and parlay CSVs.")
    parser.add_argument("--asof", type=str, default=None)    
    args = parser.parse_args()

    paths = resolve_paths(args.season, args.week, args.edition, args.reports_dir)
    proj_master = load_game_projection_master(paths, args.season, args.week)
    games = load_csv(paths.games_path)
    edges = load_csv(paths.edges_path)
    markets = load_csv(paths.markets_path)
    scores = load_csv(paths.scores_path)
    parlays = load_csv(paths.parlay_path)

    if games.empty:
        raise SystemExit(f"Missing or empty games file: {paths.games_path}")
    if edges.empty:
        raise SystemExit(f"Missing or empty edges file: {paths.edges_path}")

    edges2 = enrich_edges(edges, games)
    games2 = enrich_games(games, markets, scores)
    parlays2 = enrich_parlays(parlays, edges2, games2)
    as_of = args.asof or detect_as_of(games2, edges2)
    payload = build_report_payload(
        args.season,
        args.week,
        args.edition,
        as_of,
        edges2,
        games2 if proj_master.empty else proj_master,
        parlays2,
    )

    write_json(paths.json_path, payload)
    print(f"[OK] wrote JSON: {paths.json_path}")

    if args.write_back_labels:
        edges_back = edges.copy()
        for col in ["play_label", "board_label", "game_label", "market_norm", "confidence", "score", "p_win", "edge_prob"]:
            if col in edges2.columns:
                edges_back[col] = edges2[col].values
        edges_back.to_csv(paths.edges_path, index=False)
        print(f"[OK] stamped edge labels: {paths.edges_path}")

        if not parlays.empty:
            parlays_back = parlays.copy()
            if "confidence" in parlays2.columns:
                parlays_back["confidence"] = parlays2["confidence"].values
            leg_series = parlays2.get("leg_labels", pd.Series([[]] * len(parlays2)))
            for i in range(3):
                col = f"leg_{i+1}_label"
                vals = [legs[i] if len(legs) > i else "" for legs in leg_series]
                parlays_back[col] = vals
            parlays_back.to_csv(paths.parlay_path, index=False)
            print(f"[OK] stamped parlay labels/confidence: {paths.parlay_path}")

    try:
        render_pdf(paths.pdf_path, payload)
        print(f"[OK] wrote PDF: {paths.pdf_path}")
    except Exception as e:
        print(f"[WARN] PDF not written: {e}")


if __name__ == "__main__":
    main()
