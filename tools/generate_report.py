from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
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
    season: int,
    week: int,
    edition: str,
    reports_dir_arg: str | None = None,
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
        exports_dir=exports,
        reports_dir=reports,
        historical_odds_path=historical_odds / "nfl_historical_odds_2020_2025_master.parquet",
        merged_odds_path=historical_odds / "nfl_odds_full_merged.parquet",
        open_mid_close_path=historical_odds / "nfl_open_mid_close_odds.parquet",
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


def load_canonical_csv(paths: ReportPaths, name: str, season: int, week: int) -> pd.DataFrame:
    root = _find_root()
    extra_root = os.environ.get("CR_CANONICAL_ROOT", "").strip()

    candidates = [
        root / "tools" / "exports" / "canonical" / f"{name}_{season}_w{week}.csv",
        paths.exports_dir / "canonical" / f"{name}_{season}_w{week}.csv",
        Path("/workspaces/calculated-risk-beta-edition/tools/exports/canonical") / f"{name}_{season}_w{week}.csv",
        Path("/workspaces/calculated-risk-beta-edition/exports/canonical") / f"{name}_{season}_w{week}.csv",
    ]

    if extra_root:
        er = Path(extra_root)
        candidates.extend(
            [
                er / f"{name}_{season}_w{week}.csv",
                er / "canonical" / f"{name}_{season}_w{week}.csv",
            ]
        )

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for p in candidates:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            unique_candidates.append(p)

    for path in unique_candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False)
                if not df.empty:
                    print(f"[INFO] loaded canonical {name} from: {path}")
                    return df
            except pd.errors.EmptyDataError:
                pass
            except Exception as e:
                print(f"[WARN] failed reading canonical {name} from {path}: {e}")

    print(f"[WARN] no canonical file found for {name}_{season}_w{week}.csv")
    return pd.DataFrame()


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


def clean_team_name(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""

    aliases = {
        "football team": "Commanders",
        "washington football team": "Commanders",
        "washington": "Commanders",
        "was": "Commanders",
        "wsh": "Commanders",
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
        "buf": "Bills",
    }
    return aliases.get(s.lower(), s)


def canonical_team_name(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""

    s = str(x).strip().lower()
    if not s:
        return ""

    full_map = {
        "arizona cardinals": "Cardinals",
        "atlanta falcons": "Falcons",
        "baltimore ravens": "Ravens",
        "buffalo bills": "Bills",
        "carolina panthers": "Panthers",
        "chicago bears": "Bears",
        "cincinnati bengals": "Bengals",
        "cleveland browns": "Browns",
        "dallas cowboys": "Cowboys",
        "denver broncos": "Broncos",
        "detroit lions": "Lions",
        "green bay packers": "Packers",
        "houston texans": "Texans",
        "indianapolis colts": "Colts",
        "jacksonville jaguars": "Jaguars",
        "kansas city chiefs": "Chiefs",
        "las vegas raiders": "Raiders",
        "los angeles chargers": "Chargers",
        "los angeles rams": "Rams",
        "miami dolphins": "Dolphins",
        "minnesota vikings": "Vikings",
        "new england patriots": "Patriots",
        "new orleans saints": "Saints",
        "new york giants": "Giants",
        "new york jets": "Jets",
        "philadelphia eagles": "Eagles",
        "pittsburgh steelers": "Steelers",
        "san francisco 49ers": "49ers",
        "seattle seahawks": "Seahawks",
        "tampa bay buccaneers": "Buccaneers",
        "tennessee titans": "Titans",
        "washington commanders": "Commanders",
        "washington football team": "Commanders",
    }

    if s in full_map:
        return full_map[s]

    cleaned = clean_team_name(s)
    if cleaned:
        return str(cleaned).strip()

    return str(x).strip().title()


def matchup_key(team_a: Any, team_b: Any) -> str:
    a = canonical_team_name(team_a).strip().lower()
    b = canonical_team_name(team_b).strip().lower()
    if not a and not b:
        return ""
    return " | ".join(sorted([a, b]))


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


def fmt_ml(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        v = int(round(float(x)))
    except Exception:
        return str(x)
    return f"{v:+d}"


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


def _fmt_edge_points(x: Any) -> str:
    val = _safe_float(x)
    if pd.isna(val):
        return ""
    return f"{val * 100:+.1f}"


def load_game_projection_master(paths: ReportPaths, season: int, week: int) -> pd.DataFrame:
    candidates = [
        paths.exports_dir / "canonical" / f"game_projection_master_{season}_w{week}.parquet",
        paths.exports_dir / "canonical" / f"game_projection_master_{season}_w{week}.csv",
        _find_root() / "tools" / "exports" / "canonical" / f"game_projection_master_{season}_w{week}.parquet",
        _find_root() / "tools" / "exports" / "canonical" / f"game_projection_master_{season}_w{week}.csv",
        Path("/workspaces/calculated-risk-beta-edition/tools/exports/canonical") / f"game_projection_master_{season}_w{week}.parquet",
        Path("/workspaces/calculated-risk-beta-edition/tools/exports/canonical") / f"game_projection_master_{season}_w{week}.csv",
    ]

    for path in candidates:
        if path.exists():
            df = load_table(path)
            if not df.empty:
                print(f"[INFO] loaded projection master from: {path}")
                return df

    return pd.DataFrame()


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
        "money line": "moneyline",
        "moneyline": "moneyline",
        "spreads": "spread",
        "spread": "spread",
        "ats": "spread",
        "totals": "total",
        "total": "total",
        "game total": "total",
        "team totals": "team_total",
    }
    return mapping.get(s, s)


def resolve_picked_side(row: pd.Series) -> str:
    candidates = [
        "side_team",
        "team",
        "selection_team",
        "pick_team",
        "selection",
        "side",
        "bet_side",
        "outcome_name",
        "runner_name",
        "label",
    ]

    for col in candidates:
        val = row.get(col, "")
        cleaned = clean_team_name(val)
        if cleaned:
            return cleaned

    text = " ".join(str(row.get(c, "")) for c in candidates).lower()

    home = clean_team_name(row.get("home_team", ""))
    away = clean_team_name(row.get("away_team", ""))

    if home and home.lower() in text:
        return home
    if away and away.lower() in text:
        return away

    return ""


def format_play_label(row: pd.Series) -> str:
    market = str(row.get("market_norm", row.get("market", ""))).strip().lower()
    side = clean_team_name(row.get("picked_side", row.get("side_team", "")))
    line = row.get("line")
    total = row.get("total_line", row.get("line"))
    direction = str(row.get("direction", row.get("bet_direction", ""))).strip().title()

    raw_label = str(row.get("label", "")).strip()
    if raw_label and raw_label.lower() not in {"unknown play", "moneyline"}:
        return raw_label

    if market in {"moneyline", "ml", "h2h"}:
        return f"{side} ML" if side else "Moneyline"

    if market in {"spread", "spreads", "ats"}:
        return f"{side} {fmt_line(line)}".strip() if side else f"Spread {fmt_line(line)}".strip()

    if market in {"total", "totals", "game_total"}:
        return f"{direction or 'Over'} {fmt_total(total)}".strip()

    return side or raw_label or "Play"


def enrich_edges(edges: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()

    out = edges.copy()
    out["game_id"] = first_existing(out, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
    out["market_norm"] = first_existing(out, ["market_norm", "market_type", "market"]).map(normalize_market_name)
    out["line"] = to_num(first_existing(out, ["line", "spread_line", "points", "handicap"]))
    out["total_line"] = to_num(first_existing(out, ["total_line", "points_line", "line"]))
    out["side_team"] = first_existing(
        out,
        ["team", "side_team", "selection_team", "pick_team", "side", "selection"],
    ).map(clean_team_name)

    out["confidence"] = to_num(first_existing(out, ["confidence", "confidence_score", "conf", "grade_confidence"]))
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
    out["p_win"] = to_num(first_existing(out, ["p_win", "win_prob", "prob", "model_prob"]))
    out["market_odds"] = to_num(
        first_existing(
            out,
            ["odds", "price", "american_odds", "moneyline", "market_odds", "line_price", "selection_price", "bet_price", "price_american"],
        )
    )
    out["implied_prob"] = out["market_odds"].map(implied_prob_from_american)
    out["edge_prob"] = out["p_win"] - out["implied_prob"]

    lookup = build_game_lookup(games)
    if not lookup.empty:
        out = out.merge(lookup, on="game_id", how="left", suffixes=("", "_lk"))

    if "game_label" not in out.columns:
        out["game_label"] = first_existing(out, ["matchup", "event_name"], default="").astype(str)
    if "home_team" not in out.columns:
        out["home_team"] = first_existing(out, ["home_team", "home"], default="").map(clean_team_name)
    if "away_team" not in out.columns:
        out["away_team"] = first_existing(out, ["away_team", "away"], default="").map(clean_team_name)

    out["picked_side"] = out.apply(resolve_picked_side, axis=1)
    out["play_label"] = out.apply(format_play_label, axis=1)
    out["board_label"] = np.where(
        out["game_label"].astype(str).str.len() > 0,
        out["game_label"].astype(str) + " | " + out["play_label"].astype(str),
        out["play_label"].astype(str),
    )

    return out.sort_values(["score", "confidence"], ascending=False, na_position="last").reset_index(drop=True)


def build_market_snapshot_from_open_mid_close(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "home_team_odds",
                "away_team_odds",
                "matchup_key",
                "open_home_moneyline",
                "mid_home_moneyline",
                "close_home_moneyline",
                "open_away_moneyline",
                "mid_away_moneyline",
                "close_away_moneyline",
                "open_spread_home",
                "mid_spread_home",
                "close_spread_home",
                "open_total",
                "mid_total",
                "close_total",
            ]
        )

    d = df.copy()
    d["event_id"] = first_existing(d, ["event_id", "game_id", "id"], default="").astype(str)
    d["market_key"] = first_existing(d, ["market_key", "market", "market_type"], default="").astype(str).str.lower().str.strip()
    d["outcome_name"] = first_existing(d, ["outcome_name", "selection", "team"], default="").astype(str).str.strip()
    d["home_team"] = first_existing(d, ["home_team", "home", "HomeTeam"], default="").astype(str).str.strip()
    d["away_team"] = first_existing(d, ["away_team", "away", "AwayTeam"], default="").astype(str).str.strip()

    def _first_valid(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return s.iloc[0] if not s.empty else np.nan

    base = (
        d[["event_id", "home_team", "away_team"]]
        .drop_duplicates(subset=["event_id"])
        .rename(columns={"home_team": "home_team_odds", "away_team": "away_team_odds"})
    )

    base["home_team_norm"] = base["home_team_odds"].map(canonical_team_name).astype(str).str.strip().str.lower()
    base["away_team_norm"] = base["away_team_odds"].map(canonical_team_name).astype(str).str.strip().str.lower()
    base["matchup_key"] = base.apply(lambda r: matchup_key(r["home_team_odds"], r["away_team_odds"]), axis=1)

    h2h = d[d["market_key"].isin(["h2h", "moneyline", "ml"])].copy()
    home_h2h = h2h[h2h["outcome_name"] == h2h["home_team"]].copy()
    away_h2h = h2h[h2h["outcome_name"] == h2h["away_team"]].copy()

    home_ml = (
        home_h2h.groupby("event_id", as_index=False)
        .agg(
            open_home_moneyline=("open_price", _first_valid),
            mid_home_moneyline=("mid_price", _first_valid),
            close_home_moneyline=("close_price", _first_valid),
        )
    )

    away_ml = (
        away_h2h.groupby("event_id", as_index=False)
        .agg(
            open_away_moneyline=("open_price", _first_valid),
            mid_away_moneyline=("mid_price", _first_valid),
            close_away_moneyline=("close_price", _first_valid),
        )
    )

    spreads = d[d["market_key"] == "spreads"].copy()
    home_spreads = spreads[spreads["outcome_name"] == spreads["home_team"]].copy()

    spread_df = (
        home_spreads.groupby("event_id", as_index=False)
        .agg(
            open_spread_home=("open_point", _first_valid),
            mid_spread_home=("mid_point", _first_valid),
            close_spread_home=("close_point", _first_valid),
        )
    )

    totals = d[d["market_key"] == "totals"].copy()
    total_df = (
        totals.groupby("event_id", as_index=False)
        .agg(
            open_total=("open_point", _first_valid),
            mid_total=("mid_point", _first_valid),
            close_total=("close_point", _first_valid),
        )
    )

    out = base.merge(home_ml, on="event_id", how="left")
    out = out.merge(away_ml, on="event_id", how="left")
    out = out.merge(spread_df, on="event_id", how="left")
    out = out.merge(total_df, on="event_id", how="left")
    return out


def merge_open_mid_close_into_games(games: pd.DataFrame, open_mid_close: pd.DataFrame) -> pd.DataFrame:
    if games.empty or open_mid_close.empty:
        return games

    odds_snap = build_market_snapshot_from_open_mid_close(open_mid_close)
    if odds_snap.empty:
        return games

    g = games.copy()
    o = odds_snap.copy()

    g["game_id"] = first_existing(g, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
    o["event_id"] = o["event_id"].astype(str)

    g["home_team_norm"] = g["home_team"].map(canonical_team_name).astype(str).str.strip().str.lower()
    g["away_team_norm"] = g["away_team"].map(canonical_team_name).astype(str).str.strip().str.lower()
    g["matchup_key"] = g.apply(lambda r: matchup_key(r["home_team"], r["away_team"]), axis=1)

    o["home_team_norm"] = o["home_team_odds"].map(canonical_team_name).astype(str).str.strip().str.lower()
    o["away_team_norm"] = o["away_team_odds"].map(canonical_team_name).astype(str).str.strip().str.lower()
    o["matchup_key"] = o.apply(lambda r: matchup_key(r["home_team_odds"], r["away_team_odds"]), axis=1)

    merged = g.merge(o, left_on="game_id", right_on="event_id", how="left", suffixes=("", "_omc"))

    need_fallback = (
        merged["close_total"].isna()
        & merged["close_spread_home"].isna()
        & merged["close_home_moneyline"].isna()
    )

    if need_fallback.any():
        odds_cols = [
            "event_id",
            "home_team_odds",
            "away_team_odds",
            "home_team_norm",
            "away_team_norm",
            "matchup_key",
            "open_home_moneyline",
            "mid_home_moneyline",
            "close_home_moneyline",
            "open_away_moneyline",
            "mid_away_moneyline",
            "close_away_moneyline",
            "open_spread_home",
            "mid_spread_home",
            "close_spread_home",
            "open_total",
            "mid_total",
            "close_total",
        ]

        o_dedup = o[odds_cols].drop_duplicates(subset=["matchup_key"], keep="first").copy()
        fallback_left = merged.loc[need_fallback].copy()

        fallback = fallback_left.drop(
            columns=[c for c in odds_cols if c in fallback_left.columns and c != "matchup_key"],
            errors="ignore",
        ).merge(
            o_dedup,
            on="matchup_key",
            how="left",
            suffixes=("", "_fb"),
        )
        fallback.index = fallback_left.index

        for col in [
            "event_id",
            "home_team_odds",
            "away_team_odds",
            "open_home_moneyline",
            "mid_home_moneyline",
            "close_home_moneyline",
            "open_away_moneyline",
            "mid_away_moneyline",
            "close_away_moneyline",
            "open_spread_home",
            "mid_spread_home",
            "close_spread_home",
            "open_total",
            "mid_total",
            "close_total",
        ]:
            if col in fallback.columns:
                merged.loc[need_fallback, col] = fallback[col].values

    for col in ["market_spread_raw", "market_total", "home_moneyline", "away_moneyline"]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["market_spread_raw"] = merged["market_spread_raw"].fillna(merged.get("close_spread_home"))
    merged["market_spread_raw"] = merged["market_spread_raw"].fillna(merged.get("mid_spread_home"))
    merged["market_spread_raw"] = merged["market_spread_raw"].fillna(merged.get("open_spread_home"))

    merged["market_total"] = merged["market_total"].fillna(merged.get("close_total"))
    merged["market_total"] = merged["market_total"].fillna(merged.get("mid_total"))
    merged["market_total"] = merged["market_total"].fillna(merged.get("open_total"))

    merged["home_moneyline"] = merged["home_moneyline"].fillna(merged.get("close_home_moneyline"))
    merged["home_moneyline"] = merged["home_moneyline"].fillna(merged.get("mid_home_moneyline"))
    merged["home_moneyline"] = merged["home_moneyline"].fillna(merged.get("open_home_moneyline"))

    merged["away_moneyline"] = merged["away_moneyline"].fillna(merged.get("close_away_moneyline"))
    merged["away_moneyline"] = merged["away_moneyline"].fillna(merged.get("mid_away_moneyline"))
    merged["away_moneyline"] = merged["away_moneyline"].fillna(merged.get("open_away_moneyline"))

    return merged


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

    g["market_spread_raw"] = np.nan
    g["market_total"] = np.nan
    g["home_moneyline"] = np.nan
    g["away_moneyline"] = np.nan
    g["home_score_actual"] = np.nan
    g["away_score_actual"] = np.nan
    g["home_score_mean"] = np.nan
    g["away_score_mean"] = np.nan

    if not markets.empty:
        m = markets.copy()
        m["game_id"] = first_existing(m, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)

        spread_df = pd.DataFrame(
            {
                "game_id": m["game_id"],
                "market_spread_raw": to_num(
                    first_existing(
                        m,
                        ["market_spread", "spread", "spread_home", "closing_spread_home", "home_spread"],
                    )
                ),
            }
        ).dropna(subset=["market_spread_raw"]).drop_duplicates("game_id")

        total_df = pd.DataFrame(
            {
                "game_id": m["game_id"],
                "market_total": to_num(
                    first_existing(
                        m,
                        ["market_total", "total", "total_line", "closing_total", "game_total"],
                    )
                ),
            }
        ).dropna(subset=["market_total"]).drop_duplicates("game_id")

        home_ml_df = pd.DataFrame(
            {
                "game_id": m["game_id"],
                "home_moneyline": to_num(
                    first_existing(
                        m,
                        ["home_moneyline", "moneyline_home", "ml_home", "home_ml", "price_home", "home_price"],
                    )
                ),
            }
        ).dropna(subset=["home_moneyline"]).drop_duplicates("game_id")

        away_ml_df = pd.DataFrame(
            {
                "game_id": m["game_id"],
                "away_moneyline": to_num(
                    first_existing(
                        m,
                        ["away_moneyline", "moneyline_away", "ml_away", "away_ml", "price_away", "away_price"],
                    )
                ),
            }
        ).dropna(subset=["away_moneyline"]).drop_duplicates("game_id")

        for df_piece, colname in [
            (spread_df, "market_spread_raw"),
            (total_df, "market_total"),
            (home_ml_df, "home_moneyline"),
            (away_ml_df, "away_moneyline"),
        ]:
            if not df_piece.empty:
                g = g.merge(df_piece, on="game_id", how="left", suffixes=("", "_m"))
                alt = f"{colname}_m"
                if alt in g.columns:
                    g[colname] = g[colname].fillna(g[alt])
                    g = g.drop(columns=[alt])

    if not scores.empty:
        s = scores.copy()
        s["game_id"] = first_existing(s, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)

        mean_df = pd.DataFrame(
            {
                "game_id": s["game_id"],
                "home_score_mean": to_num(
                    first_existing(s, ["home_score_mean", "home_points_mean", "home_proj_score", "proj_home_score"])
                ),
                "away_score_mean": to_num(
                    first_existing(s, ["away_score_mean", "away_points_mean", "away_proj_score", "proj_away_score"])
                ),
                "home_score_actual": to_num(
                    first_existing(s, ["home_score_actual", "home_score", "home_points", "HomeScore", "score_home"])
                ),
                "away_score_actual": to_num(
                    first_existing(s, ["away_score_actual", "away_score", "away_points", "AwayScore", "score_away"])
                ),
            }
        ).drop_duplicates("game_id")

        g = g.merge(mean_df, on="game_id", how="left", suffixes=("", "_s"))
        for col in ["home_score_mean", "away_score_mean", "home_score_actual", "away_score_actual"]:
            alt = f"{col}_s"
            if alt in g.columns:
                g[col] = g[col].fillna(g[alt])
                g = g.drop(columns=[alt])

    power_diff = _coalesce_num(g, ["power_diff", "team_power_diff", "elo_diff", "rating_diff", "home_power_minus_away_power"])
    recent_form_diff = _coalesce_num(g, ["recent_form_diff", "form_diff", "last5_diff", "rolling_form_diff", "home_form_minus_away_form"])
    qb_diff = _coalesce_num(g, ["qb_diff", "qb_edge", "home_qb_minus_away_qb"])
    pace_diff = _coalesce_num(g, ["pace_diff", "plays_diff", "tempo_diff"])
    home_field_pts = _coalesce_num(g, ["home_field_pts", "hfa_pts", "home_field_advantage"], default=1.75)
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

    derived_total = market_total.fillna(41.5) + 0.60 * pace_diff + 0.12 * recent_form_diff.abs()

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
            raw_total,
        ),
        index=g.index,
    ).clip(30.0, 62.0)

    g["proj_home_points"] = ((g["shrunk_proj_total"] + g["shrunk_proj_margin"]) / 2.0).clip(lower=10.0)
    g["proj_away_points"] = ((g["shrunk_proj_total"] - g["shrunk_proj_margin"]) / 2.0).clip(lower=10.0)

    g["proj_home_points"] = g["proj_home_points"] + 0.20 * home_field_pts + 0.10 * qb_diff.clip(lower=0)
    g["proj_away_points"] = g["proj_away_points"] - 0.05 * home_field_pts - 0.10 * qb_diff.clip(upper=0).abs()

    g["proj_home_points"] = g["proj_home_points"].round(1)
    g["proj_away_points"] = g["proj_away_points"].round(1)
    g["median_home_points"] = g["proj_home_points"].round().astype(int)
    g["median_away_points"] = g["proj_away_points"].round().astype(int)

    g["actual_margin_home"] = g["home_score_actual"] - g["away_score_actual"]
    g["actual_total"] = g["home_score_actual"] + g["away_score_actual"]
    g["completed_flag"] = (g["home_score_actual"].notna() & g["away_score_actual"].notna()).astype(int)

    return g


def recompute_game_projections(g: pd.DataFrame, proj_master: pd.DataFrame | None = None) -> pd.DataFrame:
    if g.empty:
        return g.copy()

    out = g.copy()

    if proj_master is not None and not proj_master.empty:
        p = proj_master.copy()
        p["game_id"] = first_existing(p, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
        p["home_team"] = first_existing(p, ["home_team", "home", "HomeTeam", "team_home"]).map(clean_team_name)
        p["away_team"] = first_existing(p, ["away_team", "away", "AwayTeam", "team_away"]).map(clean_team_name)
        p["matchup_key"] = p.apply(lambda r: matchup_key(r["home_team"], r["away_team"]), axis=1)

        p_small = pd.DataFrame(
            {
                "game_id": p["game_id"],
                "matchup_key": p["matchup_key"],
                "proj_home_points_pm": to_num(
                    first_existing(
                        p,
                        ["proj_home_points", "home_proj_score", "proj_home_score", "home_score_mean", "home_points_mean"],
                    )
                ),
                "proj_away_points_pm": to_num(
                    first_existing(
                        p,
                        ["proj_away_points", "away_proj_score", "proj_away_score", "away_score_mean", "away_points_mean"],
                    )
                ),
                "proj_margin_pm": to_num(
                    first_existing(p, ["proj_margin", "projected_margin", "shrunk_proj_margin", "home_margin"])
                ),
                "proj_total_pm": to_num(
                    first_existing(p, ["proj_total", "projected_total", "shrunk_proj_total", "total_proj"])
                ),
            }
        ).drop_duplicates(subset=["game_id", "matchup_key"])

        out["matchup_key"] = out.apply(lambda r: matchup_key(r["home_team"], r["away_team"]), axis=1)
        out = out.merge(p_small.drop(columns=["matchup_key"]), on="game_id", how="left")

        miss_pm = out["proj_home_points_pm"].isna() & out["proj_away_points_pm"].isna()
        if miss_pm.any():
            p_by_matchup = p_small.drop(columns=["game_id"]).drop_duplicates(subset=["matchup_key"]).copy()
            tmp = out.loc[miss_pm, ["matchup_key"]].merge(p_by_matchup, on="matchup_key", how="left")
            tmp.index = out.loc[miss_pm].index
            for col in ["proj_home_points_pm", "proj_away_points_pm", "proj_margin_pm", "proj_total_pm"]:
                if col in tmp.columns:
                    out.loc[miss_pm, col] = tmp[col]

    if "proj_home_points_pm" in out.columns:
        out["proj_home_points"] = out["proj_home_points"].fillna(out["proj_home_points_pm"])
    if "proj_away_points_pm" in out.columns:
        out["proj_away_points"] = out["proj_away_points"].fillna(out["proj_away_points_pm"])

    both_pts = out["proj_home_points"].notna() & out["proj_away_points"].notna()
    out.loc[both_pts, "shrunk_proj_margin"] = out.loc[both_pts, "proj_home_points"] - out.loc[both_pts, "proj_away_points"]
    out.loc[both_pts, "shrunk_proj_total"] = out.loc[both_pts, "proj_home_points"] + out.loc[both_pts, "proj_away_points"]

    if "proj_margin_pm" in out.columns:
        out["shrunk_proj_margin"] = out["shrunk_proj_margin"].fillna(out["proj_margin_pm"])
    if "proj_total_pm" in out.columns:
        out["shrunk_proj_total"] = out["shrunk_proj_total"].fillna(out["proj_total_pm"])

    need_points = out["proj_home_points"].isna() | out["proj_away_points"].isna()
    if need_points.any():
        out.loc[need_points, "proj_home_points"] = (
            (out.loc[need_points, "shrunk_proj_total"] + out.loc[need_points, "shrunk_proj_margin"]) / 2.0
        ).clip(lower=10.0).round(1)
        out.loc[need_points, "proj_away_points"] = (
            (out.loc[need_points, "shrunk_proj_total"] - out.loc[need_points, "shrunk_proj_margin"]) / 2.0
        ).clip(lower=10.0).round(1)

    return out


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

    leg_game_cols = [c for c in p.columns if c.lower() in {"game_id_1", "game_id_2", "game_id_3", "leg1_game_id", "leg2_game_id", "leg3_game_id"}]
    leg_label_cols = [c for c in p.columns if c.lower() in {"leg_1_label", "leg_2_label", "leg_3_label", "leg1_label", "leg2_label", "leg3_label"}]

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
                    label = str(best.get("play_label", "")).strip() or str(best.get("board_label", "")).strip()
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

        if not labels and generic_label_col is not None:
            raw = str(row.get(generic_label_col, "")).strip()
            if raw and raw.lower() != "nan":
                labels = [raw]

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

        confidences.append(float(max(25.0, min(92.0, conf))))

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

    rows: list[dict[str, Any]] = []
    for idx, (_, r) in enumerate(tmp.iterrows(), start=1):
        rows.append(
            {
                "rank": idx,
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


def section_scripts(
    games: pd.DataFrame,
    edges: pd.DataFrame,
    proj_master: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if games.empty:
        return []

    rows: list[dict[str, Any]] = []
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
        src = proj_map.get(gid, g)

        rows.append(
            {
                "game_label": str(
                    src.get("matchup", g.get("game_label", f"{g.get('away_team', '')} @ {g.get('home_team', '')}"))
                ).strip(),
                "script": build_game_script(src, e),
            }
        )

    return rows


def build_parlay_summary(legs: list[str], conf: float, pwin: Any, score: Any, corr: str) -> str:
    score_txt = f"Composite score {float(score):.2f}" if pd.notna(score) else "Composite score unavailable"
    pwin_txt = f"joint win estimate {float(pwin) * 100:.1f}%" if pd.notna(pwin) else "joint win estimate unavailable"
    legs_txt = "; ".join(legs[:3]) if legs else "legs unavailable"
    return f"{len(legs)}-leg suggested mix. {score_txt}; {pwin_txt}; {corr}. Legs: {legs_txt}."


def section_parlays(parlays: pd.DataFrame, limit: int = 3, edges: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if edges is not None and not edges.empty:
        e = edges.copy()
        sort_cols = [c for c in ["score", "confidence"] if c in e.columns]
        e = e.sort_values(sort_cols, ascending=False) if sort_cols else e

        if "game_id" in e.columns:
            e = e.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)

        top = e.head(9).copy()
        picks = []
        for _, r in top.iterrows():
            label = str(r.get("play_label", "")).strip() or str(r.get("board_label", "")).strip()
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


def pick_featured_game(games: pd.DataFrame, edges: pd.DataFrame, edition: str) -> Optional[pd.Series]:
    if games.empty:
        return None

    g = games.copy()

    text_cols = []
    for c in ["slot", "window", "kickoff_window", "game_window", "broadcast_window", "day_slot", "game_label"]:
        if c in g.columns:
            text_cols.append(c)

    if text_cols:
        combined = pd.Series([""] * len(g), index=g.index, dtype="object")
        for c in text_cols:
            combined = combined + " " + g[c].astype(str)
        g["_edition_text"] = combined.str.lower()
    else:
        g["_edition_text"] = g["game_label"].astype(str).str.lower()

    keywords = {
        "tnf": ["thursday", "tnf"],
        "snf": ["snf", "sunday night"],
        "monday": ["monday", "mnf"],
        "tuesday": ["monday", "mnf"],
    }

    if edition in keywords:
        pats = keywords[edition]
        subset = g[g["_edition_text"].apply(lambda x: any(p in x for p in pats))]
        if not subset.empty:
            if not edges.empty and "game_id" in edges.columns:
                edge_scores = (
                    edges.groupby("game_id")["score"]
                    .max()
                    .reset_index()
                    .rename(columns={"score": "_edge_score"})
                )
                subset = subset.merge(edge_scores, on="game_id", how="left")
                subset = subset.sort_values(["_edge_score", "game_label"], ascending=[False, True], na_position="last")
            return subset.iloc[0]

    if not edges.empty and "game_id" in edges.columns:
        edge_scores = (
            edges.groupby("game_id")["score"]
            .max()
            .reset_index()
            .rename(columns={"score": "_edge_score"})
        )
        g = g.merge(edge_scores, on="game_id", how="left")
        g = g.sort_values(["_edge_score", "game_label"], ascending=[False, True], na_position="last")
        return g.iloc[0]

    return g.sort_values("game_label").iloc[0]


def section_market_snapshot(featured_game: Optional[pd.Series], edges: pd.DataFrame) -> list[dict[str, Any]]:
    if featured_game is None:
        return []

    gid = str(featured_game.get("game_id", ""))
    home = str(featured_game.get("home_team", "Home"))
    away = str(featured_game.get("away_team", "Away"))

    game_edges = edges[edges["game_id"].astype(str) == gid].copy() if (not edges.empty and "game_id" in edges.columns) else pd.DataFrame()

    rows = [
        {"label": "Matchup", "value": f"{away} @ {home}"},
        {"label": "Spread (open / mid / close)", "value": f"{fmt_line(featured_game.get('open_spread_home', np.nan))} / {fmt_line(featured_game.get('mid_spread_home', np.nan))} / {fmt_line(featured_game.get('close_spread_home', np.nan))}"},
        {"label": "Total (open / mid / close)", "value": f"{fmt_total(featured_game.get('open_total', np.nan))} / {fmt_total(featured_game.get('mid_total', np.nan))} / {fmt_total(featured_game.get('close_total', np.nan))}"},
        {"label": f"{home} ML (open / mid / close)", "value": f"{fmt_ml(featured_game.get('open_home_moneyline', np.nan))} / {fmt_ml(featured_game.get('mid_home_moneyline', np.nan))} / {fmt_ml(featured_game.get('close_home_moneyline', np.nan))}"},
        {"label": f"{away} ML (open / mid / close)", "value": f"{fmt_ml(featured_game.get('open_away_moneyline', np.nan))} / {fmt_ml(featured_game.get('mid_away_moneyline', np.nan))} / {fmt_ml(featured_game.get('close_away_moneyline', np.nan))}"},
        {"label": "Projected spread / margin", "value": fmt_line(featured_game.get("shrunk_proj_margin", np.nan))},
        {"label": "Projected total", "value": fmt_total(featured_game.get("shrunk_proj_total", np.nan))},
        {"label": "Projected score", "value": f"{home} {fmt_total(featured_game.get('proj_home_points', np.nan))} / {away} {fmt_total(featured_game.get('proj_away_points', np.nan))}"},
    ]

    if not game_edges.empty:
        best = game_edges.sort_values(["score", "confidence"], ascending=False).iloc[0]
        rows.append({"label": "Top angle", "value": str(best.get("play_label", ""))})
        rows.append({"label": "Top angle confidence", "value": f"{_safe_float(best.get('confidence', np.nan)):.1f}"})
        rows.append({"label": "Top angle score", "value": f"{_safe_float(best.get('score', np.nan)):.2f}"})

    return rows


def section_featured_game(featured_game: Optional[pd.Series], edges: pd.DataFrame, edition: str) -> list[dict[str, Any]]:
    if featured_game is None:
        return []

    gid = str(featured_game.get("game_id", ""))
    game_edges = edges[edges["game_id"].astype(str) == gid].copy() if (not edges.empty and "game_id" in edges.columns) else pd.DataFrame()

    home = str(featured_game.get("home_team", "Home"))
    away = str(featured_game.get("away_team", "Away"))
    proj_margin = _safe_float(featured_game.get("shrunk_proj_margin", np.nan))
    proj_total = _safe_float(featured_game.get("shrunk_proj_total", np.nan))
    market_spread = _safe_float(featured_game.get("market_spread_raw", np.nan))
    market_total = _safe_float(featured_game.get("market_total", np.nan))

    notes = []
    lead = f"{away} @ {home} is the featured matchup for this edition."
    notes.append({"title": "Overview", "text": lead})

    if pd.notna(proj_margin) and pd.notna(market_spread):
        diff = proj_margin - market_spread
        lean = home if proj_margin > 0 else away
        notes.append(
            {
                "title": "Spread view",
                "text": f"Model margin is {proj_margin:+.1f} versus market spread {market_spread:+.1f}. That implies relative support toward {lean}, with model-vs-market gap {diff:+.1f}.",
            }
        )

    if pd.notna(proj_total) and pd.notna(market_total):
        diff_total = proj_total - market_total
        side = "Over" if diff_total > 0 else "Under"
        notes.append(
            {
                "title": "Total view",
                "text": f"Model total is {proj_total:.1f} versus market total {market_total:.1f}. Directional lean: {side}, with model-vs-market gap {diff_total:+.1f}.",
            }
        )

    if not game_edges.empty:
        best = game_edges.sort_values(["score", "confidence"], ascending=False).iloc[0]
        notes.append(
            {
                "title": "Best playable angle",
                "text": build_edge_summary(best),
            }
        )

    if edition in {"tnf", "snf", "monday", "tuesday"}:
        notes.append(
            {
                "title": "Scenario note",
                "text": "Primetime editions emphasize a tighter matchup-specific read: margin, total, moneyline context, and the highest-confidence available angle for this board.",
            }
        )

    return notes


def grade_game_row(game_row: pd.Series) -> dict[str, Any]:
    if int(game_row.get("completed_flag", 0)) != 1:
        return {}

    home = str(game_row.get("home_team", "Home"))
    away = str(game_row.get("away_team", "Away"))
    hs = _safe_float(game_row.get("home_score_actual", np.nan))
    aws = _safe_float(game_row.get("away_score_actual", np.nan))
    spread = _safe_float(game_row.get("market_spread_raw", np.nan))
    total = _safe_float(game_row.get("market_total", np.nan))
    margin = hs - aws
    total_actual = hs + aws

    winner = home if margin > 0 else away if margin < 0 else "Push"
    if pd.notna(spread):
        if margin + spread > 0:
            ats = home
        elif margin + spread < 0:
            ats = away
        else:
            ats = "Push"
    else:
        ats = ""

    if pd.notna(total):
        if total_actual > total:
            total_side = "Over"
        elif total_actual < total:
            total_side = "Under"
        else:
            total_side = "Push"
    else:
        total_side = ""

    return {
        "game_label": str(game_row.get("game_label", f"{away} @ {home}")),
        "final_score": f"{away} {int(aws) if pd.notna(aws) else '-'} @ {home} {int(hs) if pd.notna(hs) else '-'}",
        "winner": winner,
        "ats_winner": ats,
        "total_result": total_side,
        "market_spread": "" if pd.isna(spread) else fmt_line(spread),
        "market_total": "" if pd.isna(total) else fmt_total(total),
        "actual_margin_home": None if pd.isna(margin) else round(margin, 1),
        "actual_total": None if pd.isna(total_actual) else round(total_actual, 1),
    }


def section_recent_results(games: pd.DataFrame, edition: str, limit: int = 6) -> list[dict[str, Any]]:
    if games.empty or "completed_flag" not in games.columns:
        return []

    completed = games[games["completed_flag"] == 1].copy()
    if completed.empty:
        return []

    subset = completed.tail(min(limit, len(completed)))
    rows = [grade_game_row(r) for _, r in subset.iterrows()]
    return [r for r in rows if r]


def section_weekly_report_card(games: pd.DataFrame, edges: pd.DataFrame) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []

    completed_games = games[games["completed_flag"] == 1].copy() if ("completed_flag" in games.columns) else pd.DataFrame()
    cards.append({"label": "Completed games in current week file", "value": int(len(completed_games))})
    cards.append({"label": "Ranked edge rows in current edition", "value": int(len(edges))})

    if not edges.empty:
        if "confidence" in edges.columns:
            conf = pd.to_numeric(edges["confidence"], errors="coerce").dropna()
            if not conf.empty:
                cards.append({"label": "Avg confidence of ranked edges", "value": round(float(conf.mean()), 2)})

        if "score" in edges.columns:
            sc = pd.to_numeric(edges["score"], errors="coerce").dropna()
            if not sc.empty:
                cards.append({"label": "Avg Fort Knox score of ranked edges", "value": round(float(sc.mean()), 2)})

        if "market_norm" in edges.columns:
            market_counts = edges["market_norm"].astype(str).str.lower().value_counts().to_dict()
            for market_name in ["moneyline", "spread", "total"]:
                if market_name in market_counts:
                    cards.append({"label": f"{market_name.title()} edge rows", "value": int(market_counts[market_name])})

    if not completed_games.empty:
        avg_total = pd.to_numeric(completed_games["actual_total"], errors="coerce").dropna()
        if not avg_total.empty:
            cards.append({"label": "Avg completed-game total", "value": round(float(avg_total.mean()), 2)})

    return cards


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
    featured_game = pick_featured_game(games, edges, edition)
    top_edges = section_top_edges(edges)
    heatmap = section_heatmap(edges)
    game_scripts = section_scripts(games, edges, proj_master=proj_master)
    market_snapshot = section_market_snapshot(featured_game, edges)
    featured_game_notes = section_featured_game(featured_game, edges, edition)
    recent_results = section_recent_results(games, edition)
    weekly_report_card = section_weekly_report_card(games, edges) if edition == "tuesday" else []

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
            "recent_results": recent_results,
            "market_snapshot": market_snapshot,
            "featured_game": featured_game_notes,
            "top_edges": top_edges,
            "heatmap": heatmap,
            "game_scripts": game_scripts,
            "parlays": section_parlays(parlays, edges=edges),
            "weekly_report_card": weekly_report_card,
        },
        "appendix": {
            "edge_count": int(len(edges)),
            "game_count": int(len(games)),
            "parlay_count": int(len(parlays)),
            "notes": [
                "This report is for educational and entertainment purposes only.",
                "Use confidence bands and ranking as context tools, not guarantees.",
                "Canonical week files are treated as the source of truth for this report build.",
            ],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def wrap_text(text: str, width: int = 100) -> list[str]:
    if not text:
        return [""]
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def render_pdf(path: Path, payload: dict[str, Any]) -> None:
    if not HAVE_REPORTLAB:
        raise RuntimeError("reportlab is not installed")

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    page_width, page_height = letter
    left = 42
    right = page_width - 42
    top = page_height - 42
    y = top

    def new_page() -> None:
        nonlocal y
        c.showPage()
        y = top

    def ensure_space(min_y: int = 70) -> None:
        nonlocal y
        if y < min_y:
            new_page()

    def draw_line(text: str, size: int = 10, bold: bool = False, gap: int = 13) -> None:
        nonlocal y
        ensure_space()
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(left, y, text[:180])
        y -= gap

    def draw_wrapped(text: str, size: int = 10, bold: bool = False, width: int = 100, gap: int = 12) -> None:
        for line in wrap_text(text, width=width):
            draw_line(line, size=size, bold=bold, gap=gap)

    def section_header(title: str) -> None:
        nonlocal y
        y -= 4
        ensure_space(90)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, title)
        y -= 8
        c.line(left, y, right, y)
        y -= 14

    for idx, line in enumerate(payload["title"].split("\n")):
        draw_line(line, size=14 if idx == 1 else 13, bold=True, gap=16)

    meta = payload.get("meta", {})
    draw_line(f"Prepared: {meta.get('prepared_by', '')}", size=9, gap=11)
    draw_line(f"As of: {meta.get('as_of', '')}", size=9, gap=11)
    draw_line(f"Simulation basis: {meta.get('simulation_basis', '')}", size=9, gap=14)

    section_header("CONFIDENCE BANDS")
    for row in meta.get("confidence_bands", []):
        draw_wrapped(f"- {row}", size=9, width=105, gap=11)

    recent_results = payload.get("sections", {}).get("recent_results", [])
    if recent_results:
        section_header("RECAP / RECENT RESULTS")
        for row in recent_results:
            draw_wrapped(
                f"{row['game_label']} | Final: {row['final_score']} | Winner: {row['winner']} | ATS: {row['ats_winner']} | Total: {row['total_result']}",
                size=9,
                width=106,
                gap=11,
            )

    market_snapshot = payload.get("sections", {}).get("market_snapshot", [])
    if market_snapshot:
        section_header("FEATURED GAME — MARKET SNAPSHOT")
        for row in market_snapshot:
            draw_wrapped(f"{row['label']}: {row['value']}", size=9, width=106, gap=11)

    featured_game = payload.get("sections", {}).get("featured_game", [])
    if featured_game:
        section_header("FEATURED GAME — DEEP DIVE")
        for row in featured_game:
            draw_wrapped(str(row.get("title", "")), size=10, bold=True, width=104, gap=12)
            draw_wrapped(str(row.get("text", "")), size=9, width=105, gap=11)

    top_edges = payload.get("sections", {}).get("top_edges", [])
    section_header("SECTION 1 — TOP EDGES")
    if not top_edges:
        draw_line("No qualified edge rows for this build.", size=10, gap=12)
    else:
        for row in top_edges:
            draw_wrapped(
                f"{row['rank']}) {row['board_label']} | Confidence {row['confidence']} | Band {row['band']} | Score {row['score']}",
                size=10,
                bold=True,
                width=100,
                gap=12,
            )
            draw_wrapped(row["summary"], size=9, width=102, gap=11)
            y -= 2

    heatmap = payload.get("sections", {}).get("heatmap", [])
    section_header("SECTION 2 — HEATMAP / BOARD SUMMARY")
    if not heatmap:
        draw_line("No board summary rows available.", size=10, gap=12)
    else:
        for row in heatmap:
            draw_wrapped(
                f"- {row['board_label']} | {row['band']} | Score {row['score']}",
                size=9,
                width=108,
                gap=11,
            )

    game_scripts = payload.get("sections", {}).get("game_scripts", [])
    section_header("SECTION 3 — GAME CARD NOTES / SIMULATION VALUE BOARD")
    if not game_scripts:
        draw_line("No game card notes available.", size=10, gap=12)
    else:
        for row in game_scripts:
            draw_wrapped(str(row.get("game_label", "Matchup")), size=10, bold=True, width=104, gap=12)
            draw_wrapped(str(row.get("script", "")), size=9, width=105, gap=11)
            y -= 2

    parlays = payload.get("sections", {}).get("parlays", [])
    section_header("SECTION 4 — SUGGESTED PARLAYS")
    if not parlays:
        draw_line("No parlay rows available.", size=10, gap=12)
    else:
        for row in parlays:
            draw_wrapped(f"Parlay #{row['rank']} | Confidence {row['confidence']} | {row['band']}", size=10, bold=True, width=104, gap=12)
            for leg in row.get("legs", []):
                draw_wrapped(f"- {leg}", size=9, width=105, gap=11)
            draw_wrapped(str(row.get("summary", "")), size=9, width=105, gap=11)
            y -= 2

    weekly_report_card = payload.get("sections", {}).get("weekly_report_card", [])
    if weekly_report_card:
        section_header("WEEKLY REPORT CARD")
        for row in weekly_report_card:
            draw_wrapped(f"{row['label']}: {row['value']}", size=9, width=106, gap=11)

    appendix = payload.get("appendix", {})
    section_header("APPENDIX")
    draw_line(f"Edge count: {appendix.get('edge_count', 0)}", size=9, gap=11)
    draw_line(f"Game count: {appendix.get('game_count', 0)}", size=9, gap=11)
    draw_line(f"Parlay count: {appendix.get('parlay_count', 0)}", size=9, gap=11)
    for note in appendix.get("notes", []):
        draw_wrapped(f"- {note}", size=9, width=105, gap=11)

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
    parser = argparse.ArgumentParser(
        description="Full Fort Knox 3v1 report generator using canonical week files and richer report sections."
    )
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--edition", type=str, default="monday")
    parser.add_argument("--reports-dir", type=str, default=None)
    parser.add_argument("--asof", type=str, default=None)
    args = parser.parse_args()

    print("RUNNING FULL CANONICAL RICH VERSION")

    paths = resolve_paths(args.season, args.week, args.edition, args.reports_dir)
    proj_master = load_game_projection_master(paths, args.season, args.week)

    historical = load_table(paths.historical_odds_path)
    merged = load_table(paths.merged_odds_path)
    open_mid_close = load_table(paths.open_mid_close_path)

    games = load_canonical_csv(paths, "cr_games", args.season, args.week)
    edges = load_canonical_csv(paths, "cr_edges", args.season, args.week)
    markets = load_canonical_csv(paths, "cr_markets", args.season, args.week)
    scores = load_canonical_csv(paths, "cr_scores", args.season, args.week)
    parlays = load_canonical_csv(paths, "cr_parlay_scores", args.season, args.week)

    print(f"[INFO] root={_find_root()}")
    print(f"[INFO] merged source: {paths.merged_odds_path}")
    print(f"[INFO] historical source: {paths.historical_odds_path}")
    print(f"[INFO] open_mid_close source: {paths.open_mid_close_path}")
    print(f"[INFO] season={args.season} week={args.week}")
    print(f"[INFO] merged rows: {len(merged):,}")
    print(f"[INFO] historical rows: {len(historical):,}")
    print(f"[INFO] open_mid_close rows: {len(open_mid_close):,}")
    print(f"[INFO] games rows after canonical load: {len(games):,}")
    print(f"[INFO] edges rows after canonical load: {len(edges):,}")
    print(f"[INFO] markets rows after canonical load: {len(markets):,}")
    print(f"[INFO] scores rows after canonical load: {len(scores):,}")
    print(f"[INFO] parlays rows after canonical load: {len(parlays):,}")

    if games.empty:
        raise SystemExit(
            "No canonical game rows found for "
            f"season={args.season}, week={args.week}. "
            f"Checked: {_find_root() / 'tools' / 'exports' / 'canonical'} "
            f"and {paths.exports_dir / 'canonical'}"
    )

    edges2 = enrich_edges(edges, games)
    games2 = enrich_games(games, markets, scores)
    games2 = merge_open_mid_close_into_games(games2, open_mid_close)
    games2 = recompute_game_projections(games2, proj_master=proj_master)
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

    try:
        render_pdf(paths.pdf_path, payload)
        print(f"[OK] wrote PDF: {paths.pdf_path}")
    except Exception as e:
        print(f"[WARN] PDF not written: {e}")


if __name__ == "__main__":
    main()