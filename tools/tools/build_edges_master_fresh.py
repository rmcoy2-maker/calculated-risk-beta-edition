from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------

def to_num(s, default=np.nan):
    out = pd.to_numeric(s, errors="coerce")
    if pd.isna(default):
        return out
    return out.fillna(default)


def maybe_col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


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
        return pd.Series(default, index=df.index)
    return df[col]


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, low_memory=False)


def find_root(start: Path) -> Path:
    start = start.resolve()
    for up in [start.parent] + list(start.parents):
        if (up / "exports").exists():
            return up
    return start.parent


def american_to_implied_prob(odds: pd.Series) -> pd.Series:
    x = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    pos = x > 0
    neg = x < 0
    out.loc[pos] = 100.0 / (x.loc[pos] + 100.0)
    out.loc[neg] = (-x.loc[neg]) / ((-x.loc[neg]) + 100.0)
    return out


def american_to_decimal(odds: pd.Series) -> pd.Series:
    x = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    pos = x > 0
    neg = x < 0
    out.loc[pos] = 1.0 + (x.loc[pos] / 100.0)
    out.loc[neg] = 1.0 + (100.0 / (-x.loc[neg]))
    return out


def american_to_payout_per_1(odds: pd.Series) -> pd.Series:
    dec = american_to_decimal(odds)
    return dec - 1.0


def expected_value_per_1(p_win: pd.Series, odds: pd.Series) -> pd.Series:
    payout = american_to_payout_per_1(odds)
    p = pd.to_numeric(p_win, errors="coerce")
    return p * payout - (1.0 - p)


# ------------------------------------------------------------
# Team normalization
# ------------------------------------------------------------

TEAM_ALIASES = {
    "football team": "Commanders",
    "washington football team": "Commanders",
    "washington": "Commanders",
    "wsn": "Commanders",
    "was": "Commanders",
    "tb": "Buccaneers",
    "tampa bay": "Buccaneers",
    "sf": "49ers",
    "san francisco": "49ers",
    "ne": "Patriots",
    "new england": "Patriots",
    "kc": "Chiefs",
    "kansas city": "Chiefs",
    "gb": "Packers",
    "green bay": "Packers",
    "det": "Lions",
    "detroit": "Lions",
    "ind": "Colts",
    "indianapolis": "Colts",
    "hou": "Texans",
    "houston": "Texans",
    "mia": "Dolphins",
    "miami": "Dolphins",
    "lar": "Rams",
    "la rams": "Rams",
    "lv": "Raiders",
    "las vegas": "Raiders",
    "oakland": "Raiders",
    "no": "Saints",
    "new orleans": "Saints",
    "nyj": "Jets",
    "nyg": "Giants",
    "phi": "Eagles",
    "philadelphia": "Eagles",
    "pit": "Steelers",
    "pittsburgh": "Steelers",
    "bal": "Ravens",
    "baltimore": "Ravens",
    "cin": "Bengals",
    "cincinnati": "Bengals",
    "dal": "Cowboys",
    "dallas": "Cowboys",
    "ari": "Cardinals",
    "arizona": "Cardinals",
    "atl": "Falcons",
    "atlanta": "Falcons",
    "car": "Panthers",
    "carolina": "Panthers",
    "chi": "Bears",
    "chicago": "Bears",
    "cle": "Browns",
    "cleveland": "Browns",
    "den": "Broncos",
    "denver": "Broncos",
    "jax": "Jaguars",
    "jacksonville": "Jaguars",
    "ten": "Titans",
    "tennessee": "Titans",
    "sea": "Seahawks",
    "seattle": "Seahawks",
    "min": "Vikings",
    "minnesota": "Vikings",
    "lac": "Chargers",
    "la chargers": "Chargers",
    "buf": "Bills",
    "buffalo": "Bills",
}


def clean_team_name(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    return TEAM_ALIASES.get(s.lower(), s)


def canonical_team_key(x: Any) -> str:
    s = str(x or "").strip().lower()
    if not s:
        return ""
    mapping = {
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
        "over": "Over",
        "under": "Under",
    }
    return mapping.get(s, clean_team_name(x))


# ------------------------------------------------------------
# Input resolution
# ------------------------------------------------------------

def resolve_projection_input(root: Path, season: int, week: int, input_path: Optional[str]) -> Path:
    if input_path:
        p = Path(input_path)
        if not p.is_absolute():
            p = root / p
        return p
    return root / "exports" / "canonical" / f"game_projection_master_{season}_w{week}.csv"


def resolve_odds_input(root: Path, odds_path: Optional[str]) -> Path:
    if odds_path:
        p = Path(odds_path)
        if not p.is_absolute():
            p = root / p
        return p
    return root / "exports" / "historical_odds" / "nfl_historical_odds_2020_2025_master.csv"


# ------------------------------------------------------------
# Odds snapshot extraction
# ------------------------------------------------------------

def normalize_market_key(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip().str.lower()
    mapping = {
        "spreads": "spreads",
        "spread": "spreads",
        "ats": "spreads",
        "h2h": "h2h",
        "moneyline": "h2h",
        "ml": "h2h",
        "totals": "totals",
        "total": "totals",
        "game_total": "totals",
    }
    return raw.map(lambda x: mapping.get(x, x))


def snapshot_col(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        first_existing(
            df,
            [
                "snapshot_timestamp",
                "requested_snapshot",
                "close_snapshot_timestamp",
                "market_last_update",
                "last_update",
                "pulled_at",
                "commence_time",
            ],
            default="",
        ),
        errors="coerce",
        utc=True,
    )


def build_event_odds_snapshot_table(proj: pd.DataFrame, raw_odds: pd.DataFrame) -> pd.DataFrame:
    if raw_odds.empty or proj.empty:
        return pd.DataFrame()

    p = proj.copy()
    p["event_id"] = maybe_col(p, "event_id", "").astype(str).str.strip()
    p["home_team"] = maybe_col(p, "home_team", "").map(clean_team_name)
    p["away_team"] = maybe_col(p, "away_team", "").map(clean_team_name)
    p = p[p["event_id"].str.len() > 0].drop_duplicates("event_id")
    if p.empty:
        return pd.DataFrame()

    o = raw_odds.copy()
    o["event_id"] = first_existing(o, ["event_id", "EventID", "eventid"], default="").astype(str).str.strip()
    o = o[o["event_id"].isin(set(p["event_id"]))].copy()
    if o.empty:
        return pd.DataFrame()

    o["market_key"] = normalize_market_key(first_existing(o, ["market_key", "market", "market_type"], default=""))
    o = o[o["market_key"].isin(["h2h", "spreads", "totals"])].copy()

    o["outcome_name"] = first_existing(o, ["outcome_name", "selection", "side", "team"], default="").map(canonical_team_key)
    o["point"] = to_num(first_existing(o, ["outcome_point", "close_point", "point", "line"]))
    o["price"] = to_num(first_existing(o, ["outcome_price", "close_price", "price", "odds"]))
    o["snapshot_timestamp"] = snapshot_col(o)
    o["book_key"] = first_existing(o, ["book_key", "book", "sportsbook"], default="").astype(str).str.strip()
    o = o[~(o["point"].isna() & o["price"].isna())].copy()

    # Current/market snapshot: latest row per event-book-market-outcome.
    latest_by_book = (
        o.sort_values(["event_id", "book_key", "market_key", "outcome_name", "snapshot_timestamp"])
         .groupby(["event_id", "book_key", "market_key", "outcome_name"], as_index=False)
         .tail(1)
         .copy()
    )

    # Close snapshot: currently same definition, since master snapshots are historical and we want final book state.
    close_by_book = latest_by_book.copy()

    rows: list[dict[str, Any]] = []
    for _, game in p.iterrows():
        eid = str(game["event_id"])
        home = str(game["home_team"])
        away = str(game["away_team"])
        cur = latest_by_book[latest_by_book["event_id"] == eid].copy()
        cls = close_by_book[close_by_book["event_id"] == eid].copy()
        rows.append(_aggregate_event_side_prices(eid, home, away, cur, cls))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out


def _median_or_nan(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if not s.empty else np.nan


def _aggregate_event_side_prices(event_id: str, home_team: str, away_team: str, cur: pd.DataFrame, cls: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_id": event_id,
        "market_snapshot_timestamp": cur["snapshot_timestamp"].max() if not cur.empty else pd.NaT,
        "close_snapshot_timestamp": cls["snapshot_timestamp"].max() if not cls.empty else pd.NaT,
        "market_books": int(cur["book_key"].replace("", np.nan).dropna().nunique()) if not cur.empty else 0,
        "close_books": int(cls["book_key"].replace("", np.nan).dropna().nunique()) if not cls.empty else 0,
    }

    def fills(prefix: str, sub: pd.DataFrame):
        h2h = sub[sub["market_key"] == "h2h"]
        row[f"{prefix}_home_ml"] = _median_or_nan(h2h.loc[h2h["outcome_name"] == home_team, "price"])
        row[f"{prefix}_away_ml"] = _median_or_nan(h2h.loc[h2h["outcome_name"] == away_team, "price"])

        spreads = sub[sub["market_key"] == "spreads"]
        home_spread_points = pd.to_numeric(spreads.loc[spreads["outcome_name"] == home_team, "point"], errors="coerce").dropna()
        away_spread_points = pd.to_numeric(spreads.loc[spreads["outcome_name"] == away_team, "point"], errors="coerce").dropna()
        row[f"{prefix}_spread_home"] = float(home_spread_points.median()) if not home_spread_points.empty else (
            -float(away_spread_points.median()) if not away_spread_points.empty else np.nan
        )
        row[f"{prefix}_spread_away"] = -row[f"{prefix}_spread_home"] if pd.notna(row[f"{prefix}_spread_home"]) else np.nan
        row[f"{prefix}_spread_home_odds"] = _median_or_nan(spreads.loc[spreads["outcome_name"] == home_team, "price"])
        row[f"{prefix}_spread_away_odds"] = _median_or_nan(spreads.loc[spreads["outcome_name"] == away_team, "price"])

        totals = sub[sub["market_key"] == "totals"]
        over_points = pd.to_numeric(totals.loc[totals["outcome_name"] == "Over", "point"], errors="coerce").dropna()
        under_points = pd.to_numeric(totals.loc[totals["outcome_name"] == "Under", "point"], errors="coerce").dropna()
        pool = pd.concat([over_points, under_points], ignore_index=True).dropna()
        row[f"{prefix}_total"] = float(pool.median()) if not pool.empty else np.nan
        row[f"{prefix}_over_odds"] = _median_or_nan(totals.loc[totals["outcome_name"] == "Over", "price"])
        row[f"{prefix}_under_odds"] = _median_or_nan(totals.loc[totals["outcome_name"] == "Under", "price"])

    fills("market", cur)
    fills("close", cls)
    return row


# ------------------------------------------------------------
# Side-specific CLV
# ------------------------------------------------------------

def safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = numer / denom
    out[(denom == 0) | denom.isna()] = np.nan
    return out


def calc_clv_points(side_market: str, current_line: pd.Series, close_line: pd.Series) -> pd.Series:
    current_line = pd.to_numeric(current_line, errors="coerce")
    close_line = pd.to_numeric(close_line, errors="coerce")
    if side_market == "SPREADS":
        return current_line - close_line
    if side_market == "TOTALS_OVER":
        return close_line - current_line
    if side_market == "TOTALS_UNDER":
        return current_line - close_line
    return pd.Series(np.nan, index=current_line.index)


def calc_clv_price(current_odds: pd.Series, close_odds: pd.Series) -> pd.Series:
    cur_dec = american_to_decimal(current_odds)
    cls_dec = american_to_decimal(close_odds)
    return safe_ratio(cur_dec, cls_dec) - 1.0


def combine_clv(points: pd.Series, price: pd.Series) -> pd.Series:
    out = price.copy()
    use_points = points.notna() & points.ne(0)
    out.loc[use_points] = points.loc[use_points]
    return out


# ------------------------------------------------------------
# Projection merge helpers
# ------------------------------------------------------------

def coalesce(df: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    a = maybe_col(df, preferred)
    b = maybe_col(df, fallback)
    return a.where(a.notna(), b)


def merge_projection_with_side_odds(proj: pd.DataFrame, odds_side: pd.DataFrame) -> pd.DataFrame:
    out = proj.copy()
    if odds_side.empty:
        return out

    out = out.merge(odds_side, on="event_id", how="left", suffixes=("", "_snap"))

    fill_pairs = [
        ("market_home_ml", "market_home_ml_snap"),
        ("market_away_ml", "market_away_ml_snap"),
        ("market_spread_home", "market_spread_home_snap"),
        ("market_spread_away", "market_spread_away_snap"),
        ("market_total", "market_total_snap"),
        ("close_home_ml", "close_home_ml_snap"),
        ("close_away_ml", "close_away_ml_snap"),
        ("close_spread_home", "close_spread_home_snap"),
        ("close_spread_away", "close_spread_away_snap"),
        ("close_total", "close_total_snap"),
        ("market_snapshot_timestamp", "market_snapshot_timestamp_snap"),
        ("close_snapshot_timestamp", "close_snapshot_timestamp_snap"),
        ("market_books", "market_books_snap"),
        ("close_books", "close_books_snap"),
    ]

    for base, snap in fill_pairs:
        if snap in out.columns:
            if base not in out.columns:
                out[base] = out[snap]
            else:
                out[base] = out[base].where(out[base].notna(), out[snap])
            out = out.drop(columns=[snap])

    return out


# ------------------------------------------------------------
# Edge row builders with real prices
# ------------------------------------------------------------

def base_fields(df: pd.DataFrame, ts: str) -> dict[str, pd.Series | str]:
    return {
        "ts": ts,
        "season": maybe_col(df, "season"),
        "week": maybe_col(df, "week"),
        "sport": "NFL",
        "league": "NFL",
        "home": maybe_col(df, "home_team"),
        "away": maybe_col(df, "away_team"),
        "game_id": maybe_col(df, "game_id"),
        "event_id": maybe_col(df, "event_id"),
        "game_date": maybe_col(df, "game_date"),
        "commence_time": maybe_col(df, "commence_time"),
        "best_overall_market": maybe_col(df, "best_overall_market"),
        "best_overall_label": maybe_col(df, "best_overall_label"),
        "best_overall_edge": maybe_col(df, "best_overall_edge"),
        "best_overall_confidence": maybe_col(df, "best_overall_confidence"),
        "fort_knox_score": maybe_col(df, "fort_knox_score"),
        "market_snapshot_timestamp": maybe_col(df, "market_snapshot_timestamp"),
        "close_snapshot_timestamp": maybe_col(df, "close_snapshot_timestamp"),
        "market_books": maybe_col(df, "market_books"),
        "close_books": maybe_col(df, "close_books"),
        "source_projection_file": "game_projection_master",
    }


def build_moneyline_rows(df: pd.DataFrame, ts: str) -> pd.DataFrame:
    base = base_fields(df, ts)

    home_odds = coalesce(df, "market_home_ml", "market_home_ml")
    away_odds = coalesce(df, "market_away_ml", "market_away_ml")
    close_home = maybe_col(df, "close_home_ml")
    close_away = maybe_col(df, "close_away_ml")

    home = pd.DataFrame({
        **base,
        "market": "H2H",
        "side": maybe_col(df, "home_team") + " ML",
        "team_name": maybe_col(df, "home_team"),
        "line": np.nan,
        "odds": home_odds,
        "close_line": close_home,
        "close_odds": close_home,
        "p_win": maybe_col(df, "sim_win_prob_home"),
        "edge": maybe_col(df, "home_ml_edge"),
        "clv_points": np.nan,
        "clv_price": calc_clv_price(home_odds, close_home),
    })
    away = pd.DataFrame({
        **base,
        "market": "H2H",
        "side": maybe_col(df, "away_team") + " ML",
        "team_name": maybe_col(df, "away_team"),
        "line": np.nan,
        "odds": away_odds,
        "close_line": close_away,
        "close_odds": close_away,
        "p_win": maybe_col(df, "sim_win_prob_away"),
        "edge": maybe_col(df, "away_ml_edge"),
        "clv_points": np.nan,
        "clv_price": calc_clv_price(away_odds, close_away),
    })
    return pd.concat([home, away], ignore_index=True)


def build_spread_rows(df: pd.DataFrame, ts: str) -> pd.DataFrame:
    base = base_fields(df, ts)
    home_line = maybe_col(df, "market_spread_home")
    away_line = maybe_col(df, "market_spread_away")
    close_home_line = maybe_col(df, "close_spread_home")
    close_away_line = maybe_col(df, "close_spread_away")
    home_odds = coalesce(df, "market_spread_home_odds", "market_home_spread_odds")
    away_odds = coalesce(df, "market_spread_away_odds", "market_away_spread_odds")
    close_home_odds = coalesce(df, "close_spread_home_odds", "close_home_spread_odds")
    close_away_odds = coalesce(df, "close_spread_away_odds", "close_away_spread_odds")

    home_points = calc_clv_points("SPREADS", home_line, close_home_line)
    away_points = calc_clv_points("SPREADS", away_line, close_away_line)
    home_price = calc_clv_price(home_odds, close_home_odds)
    away_price = calc_clv_price(away_odds, close_away_odds)

    home = pd.DataFrame({
        **base,
        "market": "SPREADS",
        "side": maybe_col(df, "home_team"),
        "team_name": maybe_col(df, "home_team"),
        "line": home_line,
        "odds": home_odds,
        "close_line": close_home_line,
        "close_odds": close_home_odds,
        "p_win": maybe_col(df, "sim_cover_prob_home"),
        "edge": maybe_col(df, "home_spread_edge"),
        "clv_points": home_points,
        "clv_price": home_price,
    })
    away = pd.DataFrame({
        **base,
        "market": "SPREADS",
        "side": maybe_col(df, "away_team"),
        "team_name": maybe_col(df, "away_team"),
        "line": away_line,
        "odds": away_odds,
        "close_line": close_away_line,
        "close_odds": close_away_odds,
        "p_win": maybe_col(df, "sim_cover_prob_away"),
        "edge": maybe_col(df, "away_spread_edge"),
        "clv_points": away_points,
        "clv_price": away_price,
    })
    return pd.concat([home, away], ignore_index=True)


def build_total_rows(df: pd.DataFrame, ts: str) -> pd.DataFrame:
    base = base_fields(df, ts)
    total_line = maybe_col(df, "market_total")
    close_total = maybe_col(df, "close_total")
    over_odds = coalesce(df, "market_over_odds", "market_total_over_odds")
    under_odds = coalesce(df, "market_under_odds", "market_total_under_odds")
    close_over_odds = coalesce(df, "close_over_odds", "close_total_over_odds")
    close_under_odds = coalesce(df, "close_under_odds", "close_total_under_odds")

    over_points = calc_clv_points("TOTALS_OVER", total_line, close_total)
    under_points = calc_clv_points("TOTALS_UNDER", total_line, close_total)
    over_price = calc_clv_price(over_odds, close_over_odds)
    under_price = calc_clv_price(under_odds, close_under_odds)

    over = pd.DataFrame({
        **base,
        "market": "TOTALS",
        "side": "Over",
        "team_name": "",
        "line": total_line,
        "odds": over_odds,
        "close_line": close_total,
        "close_odds": close_over_odds,
        "p_win": maybe_col(df, "sim_over_prob"),
        "edge": maybe_col(df, "over_edge"),
        "clv_points": over_points,
        "clv_price": over_price,
    })
    under = pd.DataFrame({
        **base,
        "market": "TOTALS",
        "side": "Under",
        "team_name": "",
        "line": total_line,
        "odds": under_odds,
        "close_line": close_total,
        "close_odds": close_under_odds,
        "p_win": maybe_col(df, "sim_under_prob"),
        "edge": maybe_col(df, "under_edge"),
        "clv_points": under_points,
        "clv_price": under_price,
    })
    return pd.concat([over, under], ignore_index=True)


# ------------------------------------------------------------
# Standardize / save
# ------------------------------------------------------------

def build_standardized_edges(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.Timestamp.utcnow().isoformat()
    edges = pd.concat([
        build_moneyline_rows(df, ts),
        build_spread_rows(df, ts),
        build_total_rows(df, ts),
    ], ignore_index=True)

    for c in [
        "odds", "close_odds", "line", "close_line", "p_win", "edge",
        "clv_points", "clv_price", "fort_knox_score", "best_overall_confidence",
    ]:
        edges[c] = pd.to_numeric(edges[c], errors="coerce")

    edges["clv"] = combine_clv(edges["clv_points"], edges["clv_price"])
    edges["implied_prob"] = american_to_implied_prob(edges["odds"])
    edges["close_implied_prob"] = american_to_implied_prob(edges["close_odds"])
    edges["ev"] = expected_value_per_1(edges["p_win"], edges["odds"])
    edges["ev_pct"] = edges["ev"]

    market_rank = {"H2H": 1, "SPREADS": 2, "TOTALS": 3}
    edges["market_rank"] = edges["market"].map(market_rank)

    desired = [
        "ts", "season", "week", "sport", "league", "game_date", "commence_time",
        "game_id", "event_id", "home", "away", "market", "side", "team_name",
        "line", "odds", "implied_prob", "close_line", "close_odds", "close_implied_prob",
        "p_win", "edge", "ev", "ev_pct", "clv", "clv_points", "clv_price",
        "fort_knox_score", "best_overall_confidence", "best_overall_market", "best_overall_label",
        "best_overall_edge", "market_snapshot_timestamp", "close_snapshot_timestamp",
        "market_books", "close_books", "source_projection_file", "market_rank",
    ]
    for c in desired:
        if c not in edges.columns:
            edges[c] = np.nan

    edges = edges[desired].sort_values(
        ["season", "week", "game_date", "home", "away", "market_rank", "edge"],
        ascending=[True, True, True, True, True, True, False],
    ).reset_index(drop=True)
    return edges


def save_outputs(edges: pd.DataFrame, root: Path, season: int, week: int) -> list[Path]:
    exports_dir = root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = exports_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []

    full_week = exports_dir / f"edges_standardized_{season}_w{week}.csv"
    edges.to_csv(full_week, index=False)
    out_paths.append(full_week)

    standard = exports_dir / "edges_standardized.csv"
    edges.to_csv(standard, index=False)
    out_paths.append(standard)

    master = exports_dir / "edges_master.csv"
    edges.to_csv(master, index=False)
    out_paths.append(master)

    generic = exports_dir / "edges.csv"
    edges.to_csv(generic, index=False)
    out_paths.append(generic)

    pick_explorer = exports_dir / "pick_explorer_edges.csv"
    edges.sort_values(["fort_knox_score", "edge", "ev"], ascending=False).to_csv(pick_explorer, index=False)
    out_paths.append(pick_explorer)

    parlay_builder = exports_dir / "parlay_builder_edges.csv"
    parlay = edges[(edges["ev"].fillna(-999) > 0) & (edges["p_win"].fillna(0) >= 0.50)].copy()
    parlay = parlay.sort_values(["p_win", "ev", "fort_knox_score"], ascending=False)
    parlay.to_csv(parlay_builder, index=False)
    out_paths.append(parlay_builder)

    edge_finder = exports_dir / "edge_finder_edges.csv"
    edge_scan = edges[(edges["edge"].fillna(-999) > 0) | (edges["ev"].fillna(-999) > 0)].copy()
    edge_scan = edge_scan.sort_values(["edge", "ev", "fort_knox_score"], ascending=False)
    edge_scan.to_csv(edge_finder, index=False)
    out_paths.append(edge_finder)

    canonical_week = canonical_dir / f"edges_standardized_{season}_w{week}.csv"
    edges.to_csv(canonical_week, index=False)
    out_paths.append(canonical_week)

    return out_paths


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build downstream edge outputs using side-specific prices from odds snapshots.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="Projection master CSV path.")
    parser.add_argument("--odds", type=str, default=None, help="Historical odds master CSV path.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else find_root(script_path)
    proj_path = resolve_projection_input(root, args.season, args.week, args.input)
    odds_path = resolve_odds_input(root, args.odds)

    if not proj_path.exists():
        raise SystemExit(f"Missing projection input file: {proj_path}")
    if not odds_path.exists():
        raise SystemExit(f"Missing odds file: {odds_path}")

    proj = read_csv_safe(proj_path)
    if proj.empty:
        raise SystemExit(f"Projection input is empty: {proj_path}")

    raw_odds = read_csv_safe(odds_path)
    if raw_odds.empty:
        raise SystemExit(f"Odds input is empty: {odds_path}")

    odds_side = build_event_odds_snapshot_table(proj, raw_odds)
    merged = merge_projection_with_side_odds(proj, odds_side)
    edges = build_standardized_edges(merged)
    out_paths = save_outputs(edges, root, args.season, args.week)

    print(f"[OK] projection source: {proj_path}")
    print(f"[OK] odds source: {odds_path}")
    print(f"[OK] games: {len(proj):,}")
    print(f"[OK] edge rows: {len(edges):,}")
    preview_cols = [
        c for c in [
            "week", "away", "home", "market", "side", "line", "odds",
            "close_line", "close_odds", "p_win", "edge", "ev", "clv", "clv_points", "clv_price"
        ] if c in edges.columns
    ]
    print(edges[preview_cols].head(12).to_string(index=False))
    for p in out_paths:
        print(f"[OK] wrote {p}")


if __name__ == "__main__":
    main()
