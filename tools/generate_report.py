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
    historical_odds_path: Path
    merged_odds_path: Path
    open_mid_close_path: Path
    json_path: Path
    pdf_path: Path


def _find_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _existing_first(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def resolve_paths(
    season: int,
    week: int,
    edition: str,
    reports_dir_arg: str | None = None,
) -> ReportPaths:
    root = _find_root()
    exports = root / "exports"

    historical_odds = _existing_first(
        [
            exports / "historical_odds" / "nfl_historical_odds_2020_2025_master.parquet",
            exports / "historical_odds" / "nfl_historical_odds_2020_2025_master.csv",
            root / "tools" / "exports" / "historical_odds" / "nfl_historical_odds_2020_2025_master.parquet",
            root / "tools" / "exports" / "historical_odds" / "nfl_historical_odds_2020_2025_master.csv",
        ]
    )

    merged_odds = _existing_first(
        [
            exports / "nfl_odds_full_merged.parquet",
            exports / "nfl_odds_full_merged.csv",
            exports / "historical_odds" / "nfl_odds_full_merged.parquet",
            exports / "historical_odds" / "nfl_odds_full_merged.csv",
            root / "tools" / "exports" / "historical_odds" / "nfl_odds_full_merged.parquet",
            root / "tools" / "exports" / "historical_odds" / "nfl_odds_full_merged.csv",
        ]
    )

    open_mid_close = _existing_first(
        [
            exports / "historical_odds" / "nfl_open_mid_close_odds.parquet",
            exports / "historical_odds" / "nfl_open_mid_close_odds.csv",
            root / "tools" / "exports" / "historical_odds" / "nfl_open_mid_close_odds.parquet",
            root / "tools" / "exports" / "historical_odds" / "nfl_open_mid_close_odds.csv",
        ]
    )

    return ReportPaths(
        exports_dir=exports,
        reports_dir=reports,
        historical_odds_path=historical_odds,
        merged_odds_path=merged_odds,
        open_mid_close_path=open_mid_close,
        json_path=reports / f"{season}_week{week}_{edition}_3v1.json",
        pdf_path=reports / f"{season}_week{week}_{edition}_3v1.pdf",
    )


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, low_memory=False)


def lc_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    low = lc_map(df)
    for cand in candidates:
        key = cand.lower()
        if key in low:
            return low[key]
    return None


def filter_season_week(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    season_col = find_col(out, ["season", "Season", "year", "YEAR"])
    week_col = find_col(out, ["week", "Week", "WEEK"])
    if season_col is not None:
        out = out[pd.to_numeric(out[season_col], errors="coerce") == season]
    if week_col is not None:
        out = out[pd.to_numeric(out[week_col], errors="coerce") == week]
    return out.reset_index(drop=True)


def load_game_projection_master(paths: ReportPaths, season: int, week: int) -> pd.DataFrame:
    candidates = [
        paths.exports_dir / "canonical" / f"game_projection_master_{season}_w{week}.parquet",
        paths.exports_dir / "canonical" / f"game_projection_master_{season}_w{week}.csv",
        paths.exports_dir / f"game_projection_master_{season}_w{week}.parquet",
        paths.exports_dir / f"game_projection_master_{season}_w{week}.csv",
    ]
    for path in candidates:
        if path.exists():
            df = load_table(path)
            if not df.empty:
                return df
    return pd.DataFrame()


def first_existing(df: pd.DataFrame, candidates: list[str], default: Any = np.nan) -> pd.Series:
    col = find_col(df, candidates)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def to_num(s: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out if np.isnan(default) else out.fillna(default)


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
        "buf": "Bills",
        "was": "Commanders",
    }
    return aliases.get(s.lower(), s)


def fmt_line(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:+.0f}" if abs(v - round(v)) < 1e-9 else f"{v:+.1f}"


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
    mapping = {"h2h": "moneyline", "ml": "moneyline", "money line": "moneyline", "spreads": "spread", "ats": "spread", "totals": "total", "game total": "total", "team totals": "team_total"}
    return mapping.get(s, s)


def format_play_label(row: pd.Series) -> str:
    market = str(row.get("market_norm", row.get("market", ""))).strip().lower()
    side = clean_team_name(row.get("side_team", row.get("side", row.get("selection", ""))))
    line = row.get("line")
    total = row.get("total_line", row.get("line"))
    direction = str(row.get("direction", row.get("bet_direction", ""))).strip().title()
    if market in {"moneyline", "ml", "h2h"}:
        return f"{side} ML".strip()
    if market in {"spread", "spreads", "ats"}:
        return f"{side} {fmt_line(line)}".strip() if side else f"Spread {fmt_line(line)}".strip()
    if market in {"total", "totals", "game_total"}:
        return f"{direction or 'Over'} {fmt_total(total)}".strip()
    return side or str(row.get("selection", row.get("label", "Unknown play")))


def enrich_edges(edges: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    out = edges.copy()
    out["game_id"] = first_existing(out, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
    out["market_norm"] = first_existing(out, ["market_norm", "market_type", "market"]).map(normalize_market_name)
    out["line"] = to_num(first_existing(out, ["line", "spread_line", "points", "handicap"]))
    out["total_line"] = to_num(first_existing(out, ["total_line", "points_line", "line"]))
    out["side_team"] = first_existing(out, ["team", "side_team", "selection_team", "pick_team", "side", "selection"]).map(clean_team_name)
    out["confidence"] = to_num(first_existing(out, ["confidence", "confidence_score", "conf", "grade_confidence"]))
    out["score"] = to_num(first_existing(out, ["score", "edge_score", "fort_knox_score", "fk_score", "rank_score", "fort_knox", "composite_score"]))
    out["score"] = out["score"].fillna(out["confidence"])
    out["p_win"] = to_num(first_existing(out, ["p_win", "win_prob", "prob", "model_prob"]))
    out["market_odds"] = to_num(first_existing(out, ["odds", "price", "american_odds"]))
    out["implied_prob"] = out["market_odds"].map(implied_prob_from_american)
    out["edge_prob"] = out["p_win"] - out["implied_prob"]
    lookup = build_game_lookup(games)
    if not lookup.empty:
        out = out.merge(lookup, on="game_id", how="left", suffixes=("", "_lk"))
    out["game_label"] = out.get("game_label", first_existing(out, ["matchup", "event_name"], default="").astype(str))
    out["play_label"] = out.apply(format_play_label, axis=1)
    out["board_label"] = np.where(out["game_label"].astype(str).str.len() > 0, out["game_label"].astype(str) + " | " + out["play_label"].astype(str), out["play_label"].astype(str))
    return out


def enrich_games(games: pd.DataFrame, markets: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games.copy()
    g = games.copy()
    g["game_id"] = first_existing(g, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
    g["home_team"] = first_existing(g, ["home_team", "home", "HomeTeam", "team_home"]).map(clean_team_name)
    g["away_team"] = first_existing(g, ["away_team", "away", "AwayTeam", "team_away"]).map(clean_team_name)
    g["game_label"] = np.where(g["away_team"].astype(str).str.len() > 0, g["away_team"] + " @ " + g["home_team"], first_existing(g, ["game_label", "matchup"], default="").astype(str))
    g["market_spread_raw"] = np.nan
    g["market_total"] = np.nan

    if not markets.empty:
        m = markets.copy()
        m["game_id"] = first_existing(m, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
        spread_df = pd.DataFrame({"game_id": m["game_id"], "market_spread_raw": to_num(first_existing(m, ["market_spread", "spread", "spread_home", "closing_spread_home"]))}).dropna(subset=["market_spread_raw"]).drop_duplicates("game_id")
        total_df = pd.DataFrame({"game_id": m["game_id"], "market_total": to_num(first_existing(m, ["market_total", "total", "total_line", "closing_total"]))}).dropna(subset=["market_total"]).drop_duplicates("game_id")
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

    g["home_score_mean"] = np.nan
    g["away_score_mean"] = np.nan
    if not scores.empty:
        s = scores.copy()
        s["game_id"] = first_existing(s, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
        mean_df = pd.DataFrame({"game_id": s["game_id"], "home_score_mean": to_num(first_existing(s, ["home_score_mean", "home_points_mean", "home_proj_score", "proj_home_score"])), "away_score_mean": to_num(first_existing(s, ["away_score_mean", "away_points_mean", "away_proj_score", "proj_away_score"]))}).drop_duplicates("game_id")
        g = g.merge(mean_df, on="game_id", how="left", suffixes=("", "_s"))
        for col in ["home_score_mean", "away_score_mean"]:
            alt = f"{col}_s"
            if alt in g.columns:
                g[col] = g[col].fillna(g[alt])
                g = g.drop(columns=[alt])

    power_diff = to_num(first_existing(g, ["power_diff", "team_power_diff", "elo_diff", "rating_diff", "home_power_minus_away_power"]), default=0.0)
    recent_form_diff = to_num(first_existing(g, ["recent_form_diff", "form_diff", "last5_diff", "rolling_form_diff", "home_form_minus_away_form"]), default=0.0)
    qb_diff = to_num(first_existing(g, ["qb_diff", "qb_edge", "home_qb_minus_away_qb"]), default=0.0)
    pace_diff = to_num(first_existing(g, ["pace_diff", "plays_diff", "tempo_diff"]), default=0.0)
    home_field_pts = to_num(first_existing(g, ["home_field_pts", "hfa_pts", "home_field_advantage"]), default=1.75)
    home_field_pts = home_field_pts.where(home_field_pts != 0, 1.75)
    market_spread = to_num(g["market_spread_raw"])
    market_total = to_num(g["market_total"])

    raw_margin = 0.70 * market_spread.fillna(0.0) + 0.45 * power_diff + 0.30 * recent_form_diff + 0.20 * qb_diff + 0.08 * pace_diff + home_field_pts
    shrunk_margin = np.where(market_spread.notna(), 0.75 * market_spread + 0.25 * raw_margin, raw_margin)
    g["shrunk_proj_margin"] = pd.Series(shrunk_margin, index=g.index).clip(-17.0, 17.0)
    raw_total = market_total.fillna(41.5) + 0.60 * pace_diff + 0.12 * recent_form_diff.abs()
    g["shrunk_proj_total"] = pd.Series(np.where(market_total.notna(), 0.78 * market_total + 0.22 * raw_total, raw_total), index=g.index).clip(30.0, 62.0)
    g["proj_home_points"] = ((g["shrunk_proj_total"] + g["shrunk_proj_margin"]) / 2.0).clip(lower=10.0).round(1)
    g["proj_away_points"] = ((g["shrunk_proj_total"] - g["shrunk_proj_margin"]) / 2.0).clip(lower=10.0).round(1)
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


def build_game_script(game_row: pd.Series, edges_for_game: pd.DataFrame) -> str:
    home = str(game_row.get("home_team", "Home"))
    away = str(game_row.get("away_team", "Away"))
    proj_home = _safe_float(game_row.get("proj_home_points", np.nan))
    proj_away = _safe_float(game_row.get("proj_away_points", np.nan))
    proj_margin = _safe_float(game_row.get("shrunk_proj_margin", np.nan))
    proj_total = _safe_float(game_row.get("shrunk_proj_total", np.nan))
    intro = f"{away} @ {home}."
    projection_text = ""
    if pd.notna(proj_home) and pd.notna(proj_away):
        projection_text = f"Model projection: {home} {proj_home:.1f}, {away} {proj_away:.1f}. Projected margin: {proj_margin:+.1f}. Projected total: {proj_total:.1f}."
    if edges_for_game.empty:
        reason = "No ranked edge survived filtering for this matchup."
    else:
        sort_cols = [c for c in ["score", "confidence"] if c in edges_for_game.columns]
        top = edges_for_game.sort_values(sort_cols, ascending=False).iloc[0] if sort_cols else edges_for_game.iloc[0]
        play_label = str(top.get("play_label", top.get("board_label", "Top play"))).strip()
        conf = _safe_float(top.get("confidence", np.nan))
        score = _safe_float(top.get("score", np.nan))
        edge_prob = _safe_float(top.get("edge_prob", np.nan))
        reason = f"Simulation Value Play: {play_label}. Confidence {conf:.0f}; Fort Knox Score {score:.2f}. Model edge vs implied price: {edge_prob * 100:+.1f} pts."
    return " ".join([p for p in [intro, projection_text, reason] if p]).strip()


def build_edge_summary(r: pd.Series) -> str:
    label = str(r.get("board_label", r.get("play_label", "Play")))
    conf = float(r.get("confidence", 0.0) or 0.0)
    score = float(r.get("score", 0.0) or 0.0)
    edge_prob = r.get("edge_prob", np.nan)
    prob_phrase = f"Model edge vs implied price: {float(edge_prob) * 100:+.1f} pts." if pd.notna(edge_prob) else "Model support is positive but incomplete on price-vs-prob columns."
    return f"{label}. Confidence {conf:.0f} — {confidence_band(conf)}. {prob_phrase} Fort Knox Score {score:.2f}."


def section_top_edges(edges: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if edges.empty:
        return []
    sort_cols = [c for c in ["score", "confidence"] if c in edges.columns]
    tmp = edges.sort_values(sort_cols, ascending=False).head(limit).copy() if sort_cols else edges.head(limit).copy()
    rows = []
    for _, r in tmp.iterrows():
        rows.append({"rank": len(rows) + 1, "board_label": str(r.get("board_label", "")), "confidence": round(float(r.get("confidence", 0.0)), 2), "band": confidence_band(float(r.get("confidence", 0.0))), "score": round(float(r.get("score", 0.0)), 2), "summary": build_edge_summary(r)})
    return rows


def section_heatmap(edges: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    if edges.empty:
        return []
    sort_cols = [c for c in ["score", "confidence"] if c in edges.columns]
    tmp = edges.sort_values(sort_cols, ascending=False).head(limit).copy() if sort_cols else edges.head(limit).copy()
    return [{"board_label": str(r.get("board_label", "")), "band": confidence_band(float(r.get("confidence", 0.0))), "score": round(float(r.get("score", 0.0)), 2)} for _, r in tmp.iterrows()]


def section_scripts(games: pd.DataFrame, edges: pd.DataFrame, proj_master: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    if games.empty:
        return []
    rows = []
    edge_map = {str(gid): sub.copy() for gid, sub in edges.groupby("game_id")} if (not edges.empty and "game_id" in edges.columns) else {}
    proj_map = {str(r.get("game_id", "")): r for _, r in proj_master.iterrows()} if (proj_master is not None and not proj_master.empty and "game_id" in proj_master.columns) else {}
    sort_key = "game_label" if "game_label" in games.columns else "game_id"
    for _, g in games.sort_values(sort_key).iterrows():
        gid = str(g.get("game_id", ""))
        e = edge_map.get(gid, pd.DataFrame())
        src = proj_map.get(gid, g)
        rows.append({"game_label": str(src.get("matchup", g.get("game_label", ""))).strip(), "script": build_game_script(src, e)})
    return rows


def section_parlays(parlays: pd.DataFrame, limit: int = 3, edges: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    return []


def build_report_payload(
    season: int,
    week: int,
    edition: str,
    as_of: Optional[str],
    edges: pd.DataFrame,
    games: pd.DataFrame,
    parlays: pd.DataFrame,
    proj_master: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return {
        "title": (
            f"CALCULATED RISK™ · EDGE FACTOR™ · 3v1™\n"
            f"FORT KNOX — WEEK {week} {EDITION_TITLES.get(edition, edition.upper())}"
        ),
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
                "Red (<25): avoid / informational only.",
            ],
            "simulation_basis": DEFAULT_SIM_BASIS,
        },
        "sections": {
            "top_edges": section_top_edges(edges),
            "heatmap": section_heatmap(edges),
            "game_scripts": section_scripts(games, edges, proj_master=proj_master),
            "parlays": section_parlays(parlays, edges=edges),
        },
        "appendix": {
            "edge_count": int(len(edges)),
            "game_count": int(len(games)),
            "parlay_count": int(len(parlays)),
            "notes": [
                "This report is for educational and entertainment purposes only."
            ],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def wrap_text(text: str, max_chars: int = 105) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
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
        raise RuntimeError("reportlab is not installed")
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    _, height = letter
    left, top, y = 42, height - 42, height - 42

    def new_page():
        nonlocal y
        c.showPage()
        y = top

    def draw_line(text: str, size: int = 10, bold: bool = False, gap: int = 14):
        nonlocal y
        if y < 54:
            new_page()
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(left, y, text)
        y -= gap

    for idx, line in enumerate(payload["title"].split("\n")):
        draw_line(line, size=13 if idx == 0 else 14, bold=True, gap=16)
    for row in payload["sections"]["top_edges"]:
        draw_line(f"{row['rank']}) {row['board_label']} | Confidence: {row['confidence']} | Score: {row['score']}", size=10, gap=13)
        for chunk in wrap_text(row["summary"], max_chars=100):
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
    parser = argparse.ArgumentParser(description="Fort Knox 3v1 report generator.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--edition", type=str, default="monday")
    parser.add_argument("--reports-dir", type=str, default=None)
    parser.add_argument("--write-back-labels", action="store_true")
    parser.add_argument("--asof", type=str, default=None)
    args = parser.parse_args()

    paths = resolve_paths(args.season, args.week, args.edition, args.reports_dir)
    proj_master = load_game_projection_master(paths, args.season, args.week)
    historical = load_table(paths.historical_odds_path)
    merged = load_table(paths.merged_odds_path)
    open_mid_close = load_table(paths.open_mid_close_path)
    games = filter_season_week(merged, args.season, args.week)
    edges = filter_season_week(merged, args.season, args.week)
    markets = filter_season_week(open_mid_close, args.season, args.week)
    scores = filter_season_week(historical, args.season, args.week)
    parlays = pd.DataFrame()

    print(f"[INFO] root={_find_root()}")
    print(f"[INFO] merged source: {paths.merged_odds_path}")
    print(f"[INFO] historical source: {paths.historical_odds_path}")
    print(f"[INFO] open_mid_close source: {paths.open_mid_close_path}")
    print(f"[INFO] season={args.season} week={args.week}")
    print(f"[INFO] merged rows: {len(merged):,}")
    print(f"[INFO] historical rows: {len(historical):,}")
    print(f"[INFO] open_mid_close rows: {len(open_mid_close):,}")
    print(f"[INFO] games rows after filter: {len(games):,}")
    print(f"[INFO] edges rows after filter: {len(edges):,}")
    print(f"[INFO] markets rows after filter: {len(markets):,}")
    print(f"[INFO] scores rows after filter: {len(scores):,}")
    if games.empty:
        raise SystemExit(f"No rows found for season={args.season}, week={args.week} in {paths.merged_odds_path}")

    edges2 = enrich_edges(edges, games)
    games2 = enrich_games(games, markets, scores)
    as_of = args.asof or detect_as_of(games2, edges2)
    payload = build_report_payload(args.season, args.week, args.edition, as_of, edges2, games2, parlays, proj_master=proj_master)
    write_json(paths.json_path, payload)
    print(f"[OK] wrote JSON: {paths.json_path}")
    try:
        render_pdf(paths.pdf_path, payload)
        print(f"[OK] wrote PDF: {paths.pdf_path}")
    except Exception as e:
        print(f"[WARN] PDF not written: {e}")


if __name__ == "__main__":
    main()
