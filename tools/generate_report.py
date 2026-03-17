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
DEFAULT_SIM_BASIS = (
    "50,000 Monte Carlo runs per game where model score inputs are available."
)
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


def resolve_paths(
    season: int, week: int, edition: str, reports_dir_arg: str | None = None
) -> ReportPaths:
    root = _find_root()
    exports = root / "exports"
    historical_odds = exports / "historical_odds"

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
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def filter_season_week(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    season_col = find_col(out, ["season", "Season", "YEAR", "year"])
    week_col = find_col(out, ["week", "Week", "WEEK"])

    if season_col is not None:
        out = out[pd.to_numeric(out[season_col], errors="coerce") == season]

    if week_col is not None:
        out = out[pd.to_numeric(out[week_col], errors="coerce") == week]

    return out.reset_index(drop=True)


def load_game_projection_master(
    paths: ReportPaths, season: int, week: int
) -> pd.DataFrame:
    merged = load_table(paths.merged_odds_path)
    return filter_season_week(merged, season, week)


def first_existing(
    df: pd.DataFrame, candidates: list[str], default: Any = np.nan
) -> pd.Series:
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
    g["home_team"] = first_existing(
        g, ["home_team", "home", "HomeTeam", "team_home"]
    ).map(clean_team_name)
    g["away_team"] = first_existing(
        g, ["away_team", "away", "AwayTeam", "team_away"]
    ).map(clean_team_name)

    fallback_label = first_existing(
        g, ["game_label", "matchup", "title"], default=""
    ).astype(str)
    g["game_label"] = np.where(
        g["away_team"].astype(str).str.len() > 0,
        g["away_team"].astype(str) + " @ " + g["home_team"].astype(str),
        fallback_label,
    )
    g["game_id"] = first_existing(
        g, ["game_id", "event_id", "id", "gid", "GameID"], default=""
    ).astype(str)
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
    side = clean_team_name(
        row.get("side_team", row.get("side", row.get("selection", "")))
    )
    line = row.get("line")
    total = row.get("total_line", row.get("line"))
    direction = str(row.get("direction", row.get("bet_direction", ""))).strip().title()
    player = str(row.get("player_name", row.get("player", ""))).strip()
    stat = str(row.get("prop_stat", row.get("stat_type", ""))).strip()

    if market in {"moneyline", "ml", "h2h"}:
        return f"{side} ML".strip()
    if market in {"spread", "spreads", "ats"}:
        return (
            f"{side} {fmt_line(line)}".strip()
            if side
            else f"Spread {fmt_line(line)}".strip()
        )
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
    out["game_id"] = first_existing(
        out, ["game_id", "event_id", "id", "gid", "GameID"], default=""
    ).astype(str)
    out["market_norm"] = first_existing(
        out, ["market_norm", "market_type", "market"]
    ).map(normalize_market_name)
    out["line"] = to_num(
        first_existing(out, ["line", "spread_line", "points", "handicap"])
    )
    out["total_line"] = to_num(
        first_existing(out, ["total_line", "points_line", "line"])
    )
    out["side_team"] = first_existing(
        out,
        ["team", "side_team", "selection_team", "pick_team", "side", "selection"],
    ).map(clean_team_name)

    out["confidence"] = to_num(
        first_existing(
            out, ["confidence", "confidence_score", "conf", "grade_confidence"]
        )
    )
    out["score"] = to_num(
        first_existing(
            out,
            [
                "score",
                "edge_score",
                "fort_knox_score",
                "fk_score",
                "rank_score",
                "fort_knox",
                "edge_rank_score",
                "model_score",
                "composite_score",
            ],
        )
    )
    out["score"] = out["score"].fillna(out["confidence"])
    out["p_win"] = to_num(
        first_existing(out, ["p_win", "win_prob", "prob", "model_prob"])
    )
    out["market_odds"] = to_num(first_existing(out, ["odds", "price", "american_odds"]))
    out["implied_prob"] = out["market_odds"].map(implied_prob_from_american)
    out["edge_prob"] = out["p_win"] - out["implied_prob"]

    lookup = build_game_lookup(games)
    if not lookup.empty:
        out = out.merge(lookup, on="game_id", how="left", suffixes=("", "_lk"))

    if "game_label" not in out.columns:
        out["game_label"] = first_existing(
            out, ["matchup", "event_name"], default=""
        ).astype(str)
    if "home_team" not in out.columns:
        out["home_team"] = first_existing(out, ["home_team", "home"], default="").map(
            clean_team_name
        )
    if "away_team" not in out.columns:
        out["away_team"] = first_existing(out, ["away_team", "away"], default="").map(
            clean_team_name
        )

    out["play_label"] = out.apply(format_play_label, axis=1)
    out["board_label"] = np.where(
        out["game_label"].astype(str).str.len() > 0,
        out["game_label"].astype(str) + " | " + out["play_label"].astype(str),
        out["play_label"].astype(str),
    )
    return out


def _coalesce_num(
    df: pd.DataFrame, candidates: list[str], default: float = 0.0
) -> pd.Series:
    return to_num(first_existing(df, candidates), default=default)


def enrich_games(
    games: pd.DataFrame, markets: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    if games.empty:
        return games.copy()

    g = games.copy()
    g["game_id"] = first_existing(
        g, ["game_id", "event_id", "id", "gid", "GameID"], default=""
    ).astype(str)
    g["home_team"] = first_existing(
        g, ["home_team", "home", "HomeTeam", "team_home"]
    ).map(clean_team_name)
    g["away_team"] = first_existing(
        g, ["away_team", "away", "AwayTeam", "team_away"]
    ).map(clean_team_name)
    g["game_label"] = np.where(
        g["away_team"].astype(str).str.len() > 0,
        g["away_team"] + " @ " + g["home_team"],
        first_existing(g, ["game_label", "matchup"], default="").astype(str),
    )

    g["market_spread_raw"] = np.nan
    g["market_total"] = np.nan

    if not markets.empty:
        m = markets.copy()
        m["game_id"] = first_existing(
            m, ["game_id", "event_id", "id", "gid", "GameID"], default=""
        ).astype(str)

        spread_df = (
            pd.DataFrame(
                {
                    "game_id": m["game_id"],
                    "market_spread_raw": to_num(
                        first_existing(
                            m,
                            [
                                "market_spread",
                                "spread",
                                "spread_home",
                                "closing_spread_home",
                            ],
                        )
                    ),
                }
            )
            .dropna(subset=["market_spread_raw"])
            .drop_duplicates("game_id")
        )

        total_df = (
            pd.DataFrame(
                {
                    "game_id": m["game_id"],
                    "market_total": to_num(
                        first_existing(
                            m, ["market_total", "total", "total_line", "closing_total"]
                        )
                    ),
                }
            )
            .dropna(subset=["market_total"])
            .drop_duplicates("game_id")
        )

        if not spread_df.empty:
            g = g.merge(spread_df, on="game_id", how="left", suffixes=("", "_m"))
            if "market_spread_raw_m" in g.columns:
                g["market_spread_raw"] = g["market_spread_raw"].fillna(
                    g["market_spread_raw_m"]
                )
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
        s["game_id"] = first_existing(
            s, ["game_id", "event_id", "id", "gid", "GameID"], default=""
        ).astype(str)

        home_mean = to_num(
            first_existing(
                s,
                [
                    "home_score_mean",
                    "home_points_mean",
                    "home_proj_score",
                    "proj_home_score",
                ],
            )
        )
        away_mean = to_num(
            first_existing(
                s,
                [
                    "away_score_mean",
                    "away_points_mean",
                    "away_proj_score",
                    "proj_away_score",
                ],
            )
        )

        mean_df = pd.DataFrame(
            {
                "game_id": s["game_id"],
                "home_score_mean": home_mean,
                "away_score_mean": away_mean,
            }
        ).drop_duplicates("game_id")

        g = g.merge(mean_df, on="game_id", how="left", suffixes=("", "_s"))
        if "home_score_mean_s" in g.columns:
            g["home_score_mean"] = g["home_score_mean"].fillna(g["home_score_mean_s"])
            g = g.drop(columns=["home_score_mean_s"])
        if "away_score_mean_s" in g.columns:
            g["away_score_mean"] = g["away_score_mean"].fillna(g["away_score_mean_s"])
            g = g.drop(columns=["away_score_mean_s"])

    power_diff = _coalesce_num(
        g,
        [
            "power_diff",
            "team_power_diff",
            "elo_diff",
            "rating_diff",
            "home_power_minus_away_power",
        ],
    )
    recent_form_diff = _coalesce_num(
        g,
        [
            "recent_form_diff",
            "form_diff",
            "last5_diff",
            "rolling_form_diff",
            "home_form_minus_away_form",
        ],
    )
    qb_diff = _coalesce_num(g, ["qb_diff", "qb_edge", "home_qb_minus_away_qb"])
    pace_diff = _coalesce_num(g, ["pace_diff", "plays_diff", "tempo_diff"])
    home_field_pts = _coalesce_num(
        g, ["home_field_pts", "hfa_pts", "home_field_advantage"], default=1.75
    )
    home_field_pts = home_field_pts.where(home_field_pts != 0, 1.75)

    market_spread = to_num(g["market_spread_raw"])
    market_total = to_num(g["market_total"])

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

    derived_total = (
        market_total.fillna(41.5) + 0.60 * pace_diff + 0.12 * recent_form_diff.abs()
    )

    if g["home_score_mean"].notna().any() and g["away_score_mean"].notna().any():
        mean_total = g["home_score_mean"].fillna(21.0) + g["away_score_mean"].fillna(
            20.0
        )
        raw_total = 0.55 * market_total.fillna(41.5) + 0.45 * mean_total
    else:
        raw_total = derived_total

    g["raw_proj_total"] = raw_total
    g["shrunk_proj_total"] = pd.Series(
        np.where(
            market_total.notna(),
            0.78 * market_total + 0.22 * raw_total,
            raw_total,
        ),
        index=g.index,
    ).clip(30.0, 62.0)

    g["proj_home_points"] = (
        (g["shrunk_proj_total"] + g["shrunk_proj_margin"]) / 2.0
    ).clip(lower=10.0)
    g["proj_away_points"] = (
        (g["shrunk_proj_total"] - g["shrunk_proj_margin"]) / 2.0
    ).clip(lower=10.0)

    g["proj_home_points"] = (
        g["proj_home_points"] + 0.20 * home_field_pts + 0.10 * qb_diff.clip(lower=0)
    )
    g["proj_away_points"] = (
        g["proj_away_points"]
        - 0.05 * home_field_pts
        - 0.10 * qb_diff.clip(upper=0).abs()
    )

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

    market_spread = _safe_float(
        game_row.get("market_spread_home", game_row.get("market_spread_raw", np.nan))
    )
    market_total = _safe_float(game_row.get("market_total", np.nan))

    proj_home = _safe_float(game_row.get("proj_home_points", np.nan))
    proj_away = _safe_float(game_row.get("proj_away_points", np.nan))
    proj_margin = _safe_float(
        game_row.get("proj_margin_home", game_row.get("shrunk_proj_margin", np.nan))
    )
    proj_total = _safe_float(
        game_row.get("proj_total", game_row.get("shrunk_proj_total", np.nan))
    )

    intro = f"{away} @ {home}."

    market_bits = []
    if pd.notna(market_spread):
        market_bits.append(f"Spread: {home} {fmt_line(market_spread)}")
    if pd.notna(market_total):
        market_bits.append(f"Total: {fmt_total(market_total)}")

    market_text = "Market: " + " | ".join(market_bits) + "." if market_bits else ""

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
            proj_bits.append(
                f"Model projection: {home} {proj_home:.1f}, {away} {proj_away:.1f}."
            )

        if pd.notna(proj_margin):
            fav = home if proj_margin > 0 else away
            margin_abs = abs(proj_margin)
            if margin_abs >= 10:
                proj_bits.append(f"Projected script: {fav} in clear control.")
            elif margin_abs >= 7:
                proj_bits.append(f"Projected script: {fav} by about a touchdown.")
            elif margin_abs >= 4:
                proj_bits.append(
                    f"Projected script: {fav} by more than one possession."
                )
            elif margin_abs >= 2:
                proj_bits.append(
                    f"Projected script: {fav} in a one-score control game."
                )
            else:
                proj_bits.append("Projected script: tight game with a late lean.")

        if pd.notna(proj_total):
            proj_bits.append(f"Projected total: {proj_total:.1f}.")

        projection_text = " ".join(proj_bits).strip()

    if edges_for_game.empty:
        reason = "No ranked edge survived filtering for this matchup."
    else:
        sort_cols = [c for c in ["score", "confidence"] if c in edges_for_game.columns]
        top = (
            edges_for_game.sort_values(sort_cols, ascending=False).iloc[0]
            if sort_cols
            else edges_for_game.iloc[0]
        )

        play_label = str(
            top.get("play_label", top.get("board_label", "Top play"))
        ).strip()
        market = str(top.get("market_norm", "")).lower()
        conf = _safe_float(top.get("confidence", np.nan))
        score = _safe_float(top.get("score", np.nan))
        edge_prob = _safe_float(top.get("edge_prob", np.nan))

        conf_text = (
            f"Confidence {conf:.0f}" if pd.notna(conf) else "Confidence unavailable"
        )
        score_text = (
            f"Fort Knox Score {score:.2f}"
            if pd.notna(score)
            else "Fort Knox Score unavailable"
        )
        edge_text = (
            f"Model edge vs implied price: {edge_prob * 100:+.1f} pts."
            if pd.notna(edge_prob)
            else "Model edge vs implied price unavailable."
        )

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


def normalize_matchup_text(x: Any) -> str:
    s = str(x or "").strip().upper()
    s = s.replace("VS.", "@").replace("VS", "@")
    s = " ".join(s.split())
    return s


def build_matchup_edge_lookup(
    edges: pd.DataFrame, games: pd.DataFrame
) -> dict[str, str]:
    out: dict[str, str] = {}
    if edges.empty or games.empty:
        return out

    g = build_game_lookup(games).copy()
    if g.empty:
        return out

    g["matchup_full"] = g["game_label"].astype(str).map(normalize_matchup_text)

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

    merged = e.merge(
        g[["game_id", "matchup_full", "matchup_abbr"]], on="game_id", how="left"
    )
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


def enrich_parlays(
    parlays: pd.DataFrame, edges: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    if parlays.empty:
        return parlays.copy()

    matchup_edge_lookup = build_matchup_edge_lookup(edges, games)

    p = parlays.copy()
    p["parlay_score"] = to_num(
        first_existing(p, ["parlay_score", "score", "portfolio_score", "fk_score"]),
        default=np.nan,
    )
    p["p_win"] = to_num(
        first_existing(p, ["p_win", "win_prob", "prob", "model_prob"]), default=np.nan
    )
    p["confidence_raw"] = to_num(
        first_existing(p, ["confidence", "confidence_score", "conf"]), default=np.nan
    )

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

    leg_game_cols = [
        c
        for c in p.columns
        if c.lower()
        in {
            "game_id_1",
            "game_id_2",
            "game_id_3",
            "leg1_game_id",
            "leg2_game_id",
            "leg3_game_id",
        }
    ]
    leg_label_cols = [
        c
        for c in p.columns
        if c.lower()
        in {
            "leg_1_label",
            "leg_2_label",
            "leg_3_label",
            "leg1_label",
            "leg2_label",
            "leg3_label",
        }
    ]

    generic_label_col = find_col(p, ["label"])
    matchup_col = find_col(p, ["matchup"])
    market_col = find_col(p, ["market"])
    side_col = find_col(p, ["side"])
    line_col = find_col(p, ["line"])

    for _, row in p.iterrows():
        labels: list[str] = []

        for c in leg_label_cols:
            v = str(row.get(c, "")).strip()
            if v and v.lower() != "nan":
                labels.append(v)

        if not labels:
            for c in leg_game_cols:
                gid = str(row.get(c, "")).strip()
                if not gid or gid.lower() == "nan":
                    continue

                picks = edge_lookup.get(gid, [])
                if picks:
                    best = picks[0]
                    label = (
                        str(best.get("play_label", "")).strip()
                        or str(best.get("board_label", "")).strip()
                    )
                    if label:
                        labels.append(label)
                        continue

                hit = game_lookup.loc[game_lookup["game_id"] == gid]
                if not hit.empty:
                    game_label = str(hit.iloc[0]["game_label"]).strip()
                    if game_label:
                        labels.append(f"{game_label} ML")

        if not labels:
            matchup = str(row.get(matchup_col, "")).strip() if matchup_col else ""
            matchup_norm = normalize_matchup_text(matchup)

            if matchup_norm in matchup_edge_lookup:
                labels.append(matchup_edge_lookup[matchup_norm])
            else:
                market = (
                    str(row.get(market_col, "")).strip().lower() if market_col else ""
                )
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

        if not labels and generic_label_col is not None:
            raw = str(row.get(generic_label_col, "")).strip()
            if raw and raw.lower() != "nan":
                labels = [raw]

        if not labels:
            labels = ["Pick unavailable"]

        leg_counts.append(len(labels))
        leg_texts.append(labels)

        same_game = (
            len({lbl.split(" | ")[0] for lbl in labels}) < len(labels)
            if labels
            else False
        )
        if same_game:
            correlations.append("same-game or tightly linked structure")
            corr_penalty = 4.0
        else:
            correlations.append("cross-game mix with limited structural overlap")
            corr_penalty = 0.0

        score = (
            float(row.get("parlay_score", np.nan))
            if pd.notna(row.get("parlay_score", np.nan))
            else np.nan
        )
        pwin = (
            float(row.get("p_win", np.nan))
            if pd.notna(row.get("p_win", np.nan))
            else np.nan
        )
        raw_conf = (
            float(row.get("confidence_raw", np.nan))
            if pd.notna(row.get("confidence_raw", np.nan))
            else np.nan
        )

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
        prob_phrase = (
            "Model support is positive but incomplete on price-vs-prob columns."
        )

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
    tmp = (
        edges.sort_values(sort_cols, ascending=False).head(limit).copy()
        if sort_cols
        else edges.head(limit).copy()
    )
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
    tmp = (
        edges.sort_values(sort_cols, ascending=False).head(limit).copy()
        if sort_cols
        else edges.head(limit).copy()
    )
    return [
        {
            "board_label": str(r.get("board_label", "")),
            "band": confidence_band(float(r.get("confidence", 0.0))),
            "score": round(float(r.get("score", 0.0)), 2),
        }
        for _, r in tmp.iterrows()
    ]


def section_scripts(
    games: pd.DataFrame,
    edges: pd.DataFrame,
    proj_master: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if games.empty:
        return []

    rows = []
    sort_key = "game_label" if "game_label" in games.columns else "game_id"

    edge_map: dict[str, pd.DataFrame] = {}
    if not edges.empty and "game_id" in edges.columns:
        for gid, sub in edges.groupby("game_id"):
            edge_map[str(gid)] = sub.copy()

    proj_map: dict[str, pd.Series] = {}
    if (
        proj_master is not None
        and not proj_master.empty
        and "game_id" in proj_master.columns
    ):
        for _, r in proj_master.iterrows():
            proj_map[str(r.get("game_id", ""))] = r

    for _, g in games.sort_values(sort_key).iterrows():
        gid = str(g.get("game_id", ""))
        e = edge_map.get(gid, pd.DataFrame())
        src = proj_map.get(gid, g)

        rows.append(
            {
                "game_label": str(
                    src.get(
                        "matchup",
                        g.get(
                            "game_label",
                            f"{g.get('away_team', '')} @ {g.get('home_team', '')}",
                        ),
                    )
                ).strip(),
                "script": build_game_script(src, e),
            }
        )

    return rows


def build_parlay_summary(
    legs: list[str], conf: float, pwin: Any, score: Any, corr: str
) -> str:
    score_txt = (
        f"Composite score {float(score):.2f}"
        if pd.notna(score)
        else "Composite score unavailable"
    )
    pwin_txt = (
        f"joint win estimate {float(pwin) * 100:.1f}%"
        if pd.notna(pwin)
        else "joint win estimate unavailable"
    )
    legs_txt = "; ".join(legs[:3]) if legs else "legs unavailable"
    return f"{len(legs)}-leg suggested mix. {score_txt}; {pwin_txt}; {corr}. Legs: {legs_txt}."


def section_parlays(
    parlays: pd.DataFrame, limit: int = 3, edges: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if edges is not None and not edges.empty:
        e = edges.copy()
        sort_cols = [c for c in ["score", "confidence"] if c in e.columns]
        e = e.sort_values(sort_cols, ascending=False) if sort_cols else e

        if "game_id" in e.columns:
            e = e.drop_duplicates(subset=["game_id"], keep="first").reset_index(
                drop=True
            )

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

        recipes = [picks[0:2], picks[0:3], picks[1:3]]

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

    if parlays.empty:
        return []

    tmp = parlays.copy()
    sort_cols = [c for c in ["confidence", "parlay_score", "p_win"] if c in tmp.columns]
    tmp = (
        tmp.sort_values(sort_cols, ascending=False).head(limit).copy()
        if sort_cols
        else tmp.head(limit).copy()
    )

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
            "game_scripts": section_scripts(games, edges, proj_master=proj_master),
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
        raise RuntimeError(
            "reportlab is not installed, so PDF output could not be rendered"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    _, height = letter
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
        draw_line(
            line, size=10, bold=(line == "HOW TO READ THE CONFIDENCE BANDS"), gap=14
        )

    for line in meta["confidence_bands"] + [
        f"Simulation basis: {meta['simulation_basis']}"
    ]:
        for chunk in wrap_text(line):
            draw_line(chunk, size=10, gap=12)

    draw_line("", gap=10)
    draw_line("SECTION 1 — TOP EDGES", size=11, bold=True, gap=16)
    for row in payload["sections"]["top_edges"]:
        draw_line(
            f"{row['rank']}) {row['board_label']} | Confidence: {row['confidence']} | Score: {row['score']}",
            size=10,
            gap=13,
        )
        for chunk in wrap_text(row["summary"], max_chars=100):
            draw_line(chunk, size=10, gap=12)
        draw_line("", gap=8)

    draw_line("SECTION 2 — MODEL PROJECTION BOARD", size=11, bold=True, gap=16)
    for row in payload["sections"]["heatmap"]:
        draw_line(
            f"{row['board_label']} | {row['band']} | Score {row['score']}",
            size=10,
            gap=12,
        )

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
        for col in [
            "as_of",
            "snapshot_time",
            "updated_at",
            "created_at",
            "run_ts",
            "timestamp",
        ]:
            if col in df.columns:
                vals = df[col].dropna().astype(str)
                if not vals.empty:
                    return vals.iloc[0]
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuilt Fort Knox 3v1 report generator with explicit play labels and better scripts."
    )
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--edition", type=str, default="monday")
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help=r"Optional override for output report folder, e.g. exports\reports_test8",
    )
    parser.add_argument(
        "--write-back-labels",
        action="store_true",
        help="Disabled in master-CSV/parquet mode; retained only for CLI compatibility.",
    )
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

    print(f"[INFO] season={args.season} week={args.week}")
    print(f"[INFO] merged rows: {len(merged):,}")
    print(f"[INFO] historical rows: {len(historical):,}")
    print(f"[INFO] open_mid_close rows: {len(open_mid_close):,}")
    print(f"[INFO] games rows after filter: {len(games):,}")
    print(f"[INFO] edges rows after filter: {len(edges):,}")
    print(f"[INFO] markets rows after filter: {len(markets):,}")
    print(f"[INFO] scores rows after filter: {len(scores):,}")

    if games.empty:
        raise SystemExit(
            f"No rows found for season={args.season}, week={args.week} in {paths.merged_odds_path}"
        )

    if edges.empty:
        print(
            f"[WARN] No edge rows found for season={args.season}, week={args.week} in {paths.merged_odds_path}"
        )

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
        games2,
        parlays2,
        proj_master=proj_master,
    )

    write_json(paths.json_path, payload)
    print(f"[OK] wrote JSON: {paths.json_path}")

    if args.write_back_labels:
        print("[WARN] write-back-labels is disabled in master-CSV/parquet mode.")

    try:
        render_pdf(paths.pdf_path, payload)
        print(f"[OK] wrote PDF: {paths.pdf_path}")
    except Exception as e:
        print(f"[WARN] PDF not written: {e}")


if __name__ == "__main__":
    main()
