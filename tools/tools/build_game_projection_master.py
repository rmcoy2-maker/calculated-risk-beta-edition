from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------

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


def to_num(s: pd.Series | Any, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if np.isnan(default):
        return out
    return out.fillna(default)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, low_memory=False)


def implied_prob_from_american_series(odds: pd.Series) -> pd.Series:
    x = to_num(odds)
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    pos = x > 0
    neg = x < 0
    out.loc[pos] = 100.0 / (x.loc[pos] + 100.0)
    out.loc[neg] = (-x.loc[neg]) / ((-x.loc[neg]) + 100.0)
    return out


def clip(series: pd.Series | float, lo: float, hi: float):
    if isinstance(series, pd.Series):
        return series.clip(lo, hi)
    return max(lo, min(hi, series))


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
    "rams": "Rams",
    "lv": "Raiders",
    "las vegas": "Raiders",
    "oakland": "Raiders",
    "no": "Saints",
    "new orleans": "Saints",
    "nyj": "Jets",
    "jets": "Jets",
    "nyg": "Giants",
    "giants": "Giants",
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
    "chargers": "Chargers",
    "buf": "Bills",
    "buffalo": "Bills",
    "bills": "Bills",
    "browns": "Browns",
    "49ers": "49ers",
    "eagles": "Eagles",
    "steelers": "Steelers",
    "ravens": "Ravens",
    "chiefs": "Chiefs",
    "cowboys": "Cowboys",
    "packers": "Packers",
    "lions": "Lions",
    "bears": "Bears",
    "texans": "Texans",
    "colts": "Colts",
    "patriots": "Patriots",
    "falcons": "Falcons",
    "jets": "Jets",
    "jaguars": "Jaguars",
    "titans": "Titans",
    "raiders": "Raiders",
    "saints": "Saints",
    "dolphins": "Dolphins",
    "vikings": "Vikings",
    "seahawks": "Seahawks",
    "panthers": "Panthers",
    "broncos": "Broncos",
    "cardinals": "Cardinals",
    "buccaneers": "Buccaneers",
    "rams": "Rams",
    "giants": "Giants",
    "commanders": "Commanders",
}


def clean_team_name(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    return TEAM_ALIASES.get(s.lower(), s)


# ------------------------------------------------------------
# Root / file loading
# ------------------------------------------------------------

def find_root(start: Path) -> Path:
    start = start.resolve()
    for up in [start.parent] + list(start.parents):
        if (up / "exports").exists():
            return up
    return start.parent

def attach_event_id_from_markets(games: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    g = prep_games(games).copy()

    # preserve existing game_id
    g["game_id"] = first_existing(g, ["game_id", "GameID", "gid"], default="").astype(str).str.strip()
    g["event_id"] = ""

    if markets.empty:
        return g

    m = markets.copy()
    m["game_id"] = first_existing(m, ["game_id", "GameID", "gid"], default="").astype(str).str.strip()
    m["event_id"] = first_existing(m, ["event_id", "EventID", "eventid"], default="").astype(str).str.strip()
    m["away_team"] = first_existing(m, ["away_team", "away", "AwayTeam"]).map(clean_team_name)
    m["home_team"] = first_existing(m, ["home_team", "home", "HomeTeam"]).map(clean_team_name)
    m["matchup"] = np.where(
        m["away_team"].astype(str).str.len() > 0,
        m["away_team"] + " @ " + m["home_team"],
        first_existing(m, ["matchup"], default="").astype(str),
    )

    keep = m[["game_id", "event_id", "matchup", "away_team", "home_team"]].drop_duplicates()

    # First try game_id
    g = g.merge(
        keep[["game_id", "event_id"]].drop_duplicates(),
        on="game_id",
        how="left",
        suffixes=("", "_m"),
    )
    if "event_id_m" in g.columns:
        g["event_id"] = g["event_id"].replace("", np.nan).fillna(g["event_id_m"]).fillna("").astype(str)
        g = g.drop(columns=["event_id_m"])

    # Fallback on matchup if still missing
    missing = g["event_id"].astype(str).str.len().eq(0)
    if missing.any():
        g_missing = g.loc[missing].drop(columns=["event_id"]).copy()
        g_missing = g_missing.merge(
            keep[["matchup", "event_id"]].drop_duplicates(),
            on="matchup",
            how="left",
        )
        g.loc[missing, "event_id"] = g_missing["event_id"].fillna("").astype(str).values

    return g

def load_week_games(root: Path, season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_candidates = [
        root / "exports" / "canonical" / f"cr_games_{season}_w{week}.csv",
        root / "exports" / f"cr_games_{season}_w{week}.csv",
    ]
    market_candidates = [
        root / "exports" / "canonical" / f"cr_markets_{season}_w{week}.csv",
        root / "exports" / f"cr_markets_{season}_w{week}.csv",
    ]

    games = pd.DataFrame()
    markets = pd.DataFrame()

    for path in game_candidates:
        games = read_csv_safe(path)
        if not games.empty:
            break

    for path in market_candidates:
        markets = read_csv_safe(path)
        if not markets.empty:
            break

    return games, markets


def prep_games(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["game_id"] = first_existing(g, ["game_id", "event_id", "id", "gid", "GameID"], default="").astype(str)
    g["season"] = to_num(first_existing(g, ["season", "Season", "year"])).astype("Int64")
    g["week"] = to_num(first_existing(g, ["week", "Week", "week_num", "week_number"])).astype("Int64")
    g["game_date"] = first_existing(g, ["game_date", "Date", "date"])
    g["commence_time"] = first_existing(g, ["commence_time", "game_datetime_utc", "start_time", "kickoff_time"])
    g["away_team"] = first_existing(g, ["away_team", "away", "AwayTeam", "team_away"]).map(clean_team_name)
    g["home_team"] = first_existing(g, ["home_team", "home", "HomeTeam", "team_home"]).map(clean_team_name)
    g["matchup"] = np.where(
        g["away_team"].astype(str).str.len() > 0,
        g["away_team"] + " @ " + g["home_team"],
        first_existing(g, ["matchup", "game_label"], default="").astype(str),
    )
    return g

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
    }

    if s in mapping:
        return mapping[s]

    return clean_team_name(x)
# ------------------------------------------------------------
# Historical odds loading
# ------------------------------------------------------------

def load_primary_odds(root: Path) -> pd.DataFrame:
    candidates = [
        root / "exports" / "historical_odds" / "nfl_historical_odds_2020_2025_master.csv",
        root / "exports" / "historical_odds" / "nfl_open_mid_close_odds.csv",
        root / "exports" / "historical_odds" / "nfl_odds_full_merged.csv",
    ]
    for path in candidates:
        df = read_csv_safe(path)
        if not df.empty:
            print(f"[INFO] Using odds source: {path}")
            return df
    return pd.DataFrame()


def american_to_decimal_series(odds: pd.Series) -> pd.Series:
    x = to_num(odds)
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    pos = x > 0
    neg = x < 0
    out.loc[pos] = 1.0 + (x.loc[pos] / 100.0)
    out.loc[neg] = 1.0 + (100.0 / (-x.loc[neg]))
    return out


def snapshot_col(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(
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
    return ts


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


def normalize_outcome_name(s: pd.Series) -> pd.Series:
    out = s.map(canonical_team_key)
    out = out.replace({"over": "Over", "under": "Under"})
    return out


def aggregate_consensus_snapshot(sub: pd.DataFrame, game: pd.Series, label: str) -> dict[str, Any]:
    home_team = str(game["home_team"]).strip()
    away_team = str(game["away_team"]).strip()

    row = {
        "event_id": str(game.get("event_id", "")),
        "game_id": str(game.get("game_id", "")),
        "home_team": home_team,
        "away_team": away_team,
        "matchup": f"{away_team} @ {home_team}",
        f"{label}_snapshot_timestamp": sub["snapshot_timestamp"].max() if not sub.empty else pd.NaT,
    }

    h2h = sub[sub["market_key"] == "h2h"].copy()
    home_prices = h2h.loc[h2h["outcome_name"] == home_team, "price"].dropna()
    away_prices = h2h.loc[h2h["outcome_name"] == away_team, "price"].dropna()
    row[f"{label}_home_ml"] = float(home_prices.median()) if not home_prices.empty else np.nan
    row[f"{label}_away_ml"] = float(away_prices.median()) if not away_prices.empty else np.nan

    spreads = sub[sub["market_key"] == "spreads"].copy()
    home_spreads = spreads.loc[spreads["outcome_name"] == home_team, "point"].dropna()
    away_spreads = spreads.loc[spreads["outcome_name"] == away_team, "point"].dropna()
    if not home_spreads.empty:
        row[f"{label}_spread_home"] = float(home_spreads.median())
    elif not away_spreads.empty:
        row[f"{label}_spread_home"] = -float(away_spreads.median())
    else:
        row[f"{label}_spread_home"] = np.nan

    totals = sub[sub["market_key"] == "totals"].copy()
    over_pts = totals.loc[totals["outcome_name"] == "Over", "point"].dropna()
    under_pts = totals.loc[totals["outcome_name"] == "Under", "point"].dropna()
    total_pool = pd.concat([over_pts, under_pts], ignore_index=True).dropna()
    row[f"{label}_total"] = float(total_pool.median()) if not total_pool.empty else np.nan

    row[f"{label}_books"] = int(sub["book_key"].replace("", np.nan).dropna().nunique()) if "book_key" in sub.columns else 0
    return row


def compute_clv_columns(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()

    out["close_spread_away"] = -1.0 * to_num(out["close_spread_home"])
    out["market_spread_away"] = -1.0 * to_num(out["market_spread_home"])

    out["home_spread_clv"] = to_num(out["market_spread_home"]) - to_num(out["close_spread_home"])
    out["away_spread_clv"] = to_num(out["market_spread_away"]) - to_num(out["close_spread_away"])

    out["over_clv"] = to_num(out["close_total"]) - to_num(out["market_total"])
    out["under_clv"] = to_num(out["market_total"]) - to_num(out["close_total"])

    market_home_dec = american_to_decimal_series(out["market_home_ml"])
    close_home_dec = american_to_decimal_series(out["close_home_ml"])
    market_away_dec = american_to_decimal_series(out["market_away_ml"])
    close_away_dec = american_to_decimal_series(out["close_away_ml"])

    out["home_ml_clv"] = (market_home_dec / close_home_dec) - 1.0
    out["away_ml_clv"] = (market_away_dec / close_away_dec) - 1.0

    out["close_home_implied"] = implied_prob_from_american_series(out["close_home_ml"])
    out["close_away_implied"] = implied_prob_from_american_series(out["close_away_ml"])
    return out


def prep_odds_for_merge(df: pd.DataFrame, season: int, week: int, games: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    g = prep_games(games).copy()
    g["game_id"] = first_existing(g, ["game_id", "GameID", "gid"], default="").astype(str).str.strip()
    g["event_id"] = first_existing(g, ["event_id", "EventID", "eventid"], default="").astype(str).str.strip()
    g["away_team"] = first_existing(g, ["away_team", "away", "AwayTeam"]).map(clean_team_name)
    g["home_team"] = first_existing(g, ["home_team", "home", "HomeTeam"]).map(clean_team_name)
    g["matchup"] = np.where(
        g["away_team"].astype(str).str.len() > 0,
        g["away_team"] + " @ " + g["home_team"],
        first_existing(g, ["matchup"], default="").astype(str),
    )

    g = g[g["event_id"].astype(str).str.len() > 0].copy()
    if g.empty:
        print("[WARN] Weekly games file has no usable event_id values.")
        return pd.DataFrame()

    o = df.copy()
    o["event_id"] = first_existing(o, ["event_id", "EventID", "eventid"], default="").astype(str).str.strip()
    o = o[o["event_id"].isin(set(g["event_id"]))].copy()
    if o.empty:
        print("[WARN] Odds source had no rows matching weekly event_id values.")
        return pd.DataFrame()

    o["market_key"] = normalize_market_key(first_existing(o, ["market_key", "market", "market_type"], default=""))
    o["outcome_name"] = normalize_outcome_name(first_existing(o, ["outcome_name", "selection", "side", "team"], default=""))
    o["point"] = to_num(first_existing(o, ["outcome_point", "close_point", "point", "line"]))
    o["price"] = to_num(first_existing(o, ["outcome_price", "close_price", "price", "odds"]))
    o["snapshot_timestamp"] = snapshot_col(o)
    o["book_key"] = first_existing(o, ["book_key", "book", "sportsbook"], default="").astype(str).str.strip()

    o = o[o["market_key"].isin(["h2h", "spreads", "totals"])].copy()
    o = o[~(o["point"].isna() & o["price"].isna())].copy()
    if o.empty:
        print("[WARN] Matching odds rows were found, but no usable market snapshots remained after normalization.")
        return pd.DataFrame()

    latest_by_book = (
        o.sort_values(["event_id", "book_key", "market_key", "outcome_name", "snapshot_timestamp"])
         .groupby(["event_id", "book_key", "market_key", "outcome_name"], as_index=False)
         .tail(1)
         .copy()
    )

    close_by_book = latest_by_book.copy()

    rows = []
    for _, game in g.drop_duplicates("event_id").iterrows():
        eid = str(game["event_id"])
        cur = latest_by_book[latest_by_book["event_id"] == eid].copy()
        cls = close_by_book[close_by_book["event_id"] == eid].copy()
        if cur.empty and cls.empty:
            continue

        cur_row = aggregate_consensus_snapshot(cur, game, "market")
        cls_row = aggregate_consensus_snapshot(cls, game, "close")
        merged = {**cur_row, **cls_row}
        rows.append(merged)

    out = pd.DataFrame(rows)
    if out.empty:
        print("[WARN] Could not aggregate odds rows into weekly market rows.")
        return out

    numeric_cols = [
        "market_spread_home", "market_total", "market_home_ml", "market_away_ml",
        "close_spread_home", "close_total", "close_home_ml", "close_away_ml",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = to_num(out[c])

    out = compute_clv_columns(out)
    return out


# ------------------------------------------------------------
# Historical score baselines
# ------------------------------------------------------------

def load_scores_history(root: Path) -> pd.DataFrame:
    path = root / "raw" / "2017-2025_scores.csv"
    df = read_csv_safe(path)
    if not df.empty:
        print(f"[INFO] Using score history: {path}")
    return df


def prep_score_baselines(df: pd.DataFrame, season: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    s = df.copy()

    season_col = find_col(s, ["season", "Season", "year"])
    if season_col is not None:
        s = s[to_num(s[season_col]).lt(season) | to_num(s[season_col]).eq(season)]

    s["home_team"] = first_existing(s, ["home_team", "HomeTeam", "home"]).map(clean_team_name)
    s["away_team"] = first_existing(s, ["away_team", "AwayTeam", "away"]).map(clean_team_name)

    s["home_score"] = to_num(first_existing(s, ["home_score", "HomeScore", "score_home"]))
    s["away_score"] = to_num(first_existing(s, ["away_score", "AwayScore", "score_away"]))

    home_pf = s.groupby("home_team")["home_score"].mean().rename("home_team_avg_scored")
    home_pa = s.groupby("home_team")["away_score"].mean().rename("home_team_avg_allowed")
    away_pf = s.groupby("away_team")["away_score"].mean().rename("away_team_avg_scored")
    away_pa = s.groupby("away_team")["home_score"].mean().rename("away_team_avg_allowed")

    team_df = pd.concat([home_pf, home_pa, away_pf, away_pa], axis=1).reset_index().rename(columns={"index": "team"})
    team_df["team"] = first_existing(team_df, ["home_team", "away_team", "team"], default="").astype(str)

    # collapse rows if same team appears in both home/away group frames
    out = team_df.groupby("team", as_index=False).mean(numeric_only=True)
    out["team"] = out["team"].map(clean_team_name)
    return out


# ------------------------------------------------------------
# Projection builder
# ------------------------------------------------------------

def add_game_features_from_games(out: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()

    feature_map = {
        "power_diff_home": ["power_diff_home", "power_diff", "team_power_diff", "elo_diff", "rating_diff", "home_power_minus_away_power"],
        "recent_form_diff_home": ["recent_form_diff_home", "recent_form_diff", "form_diff", "last5_diff", "rolling_form_diff", "home_form_minus_away_form"],
        "qb_diff_home": ["qb_diff_home", "qb_diff", "qb_edge", "home_qb_minus_away_qb"],
        "offense_diff_home": ["offense_diff_home", "offense_diff", "off_diff", "home_offense_minus_away_offense"],
        "defense_diff_home": ["defense_diff_home", "defense_diff", "def_diff", "home_defense_minus_away_defense"],
        "pace_diff_home": ["pace_diff_home", "pace_diff", "plays_diff", "tempo_diff"],
        "home_field_pts": ["home_field_pts", "hfa_pts", "home_field_advantage"],
        "rest_diff_home": ["rest_diff_home", "rest_diff"],
        "injury_diff_home": ["injury_diff_home", "injury_diff"],
        "weather_penalty_pts": ["weather_penalty_pts", "weather_penalty"],
        "weather_total_penalty": ["weather_total_penalty"],
    }

    for dest, aliases in feature_map.items():
        out[dest] = 0.0

        vals = to_num(first_existing(g, aliases), default=np.nan)
        if vals.notna().any():
            mapper = pd.DataFrame({"game_id": g["game_id"], dest: vals}).drop_duplicates("game_id")
            out = out.merge(mapper, on="game_id", how="left", suffixes=("", "_g"))
            if f"{dest}_g" in out.columns:
                out[dest] = out[f"{dest}_g"].fillna(out[dest])
                out = out.drop(columns=[f"{dest}_g"])

    out["home_field_pts"] = out["home_field_pts"].replace(0, 1.75).fillna(1.75)
    return out


def build_projection_master(games: pd.DataFrame, odds: pd.DataFrame, score_baselines: pd.DataFrame, n_sims: int = 50000, sim_seed: int = 42) -> pd.DataFrame:
    g = prep_games(games)

    g["event_id"] = first_existing(g, ["event_id", "EventID", "eventid"], default="").astype(str).str.strip()
    out = g[["season", "week", "game_id", "event_id", "game_date", "commence_time", "away_team", "home_team", "matchup"]].copy()
    out = out.merge(odds, on="event_id", how="left", suffixes=("", "_odds"))

    for c in ["home_team", "away_team", "matchup"]:
        if f"{c}_odds" in out.columns:
            out[c] = out[c].fillna(out[f"{c}_odds"])
            out = out.drop(columns=[f"{c}_odds"])

    out["market_spread_away"] = -1 * to_num(out["market_spread_home"])
    out["market_home_implied"] = implied_prob_from_american_series(out["market_home_ml"])
    out["market_away_implied"] = implied_prob_from_american_series(out["market_away_ml"])

    out = add_game_features_from_games(out, g)

    if not score_baselines.empty:
        home_base = score_baselines.rename(
            columns={
                "team": "home_team",
                "home_team_avg_scored": "home_avg_scored",
                "home_team_avg_allowed": "home_avg_allowed",
                "away_team_avg_scored": "home_road_avg_scored",
                "away_team_avg_allowed": "home_road_avg_allowed",
            }
        )
        away_base = score_baselines.rename(
            columns={
                "team": "away_team",
                "home_team_avg_scored": "away_home_avg_scored",
                "home_team_avg_allowed": "away_home_avg_allowed",
                "away_team_avg_scored": "away_avg_scored",
                "away_team_avg_allowed": "away_avg_allowed",
            }
        )

        out = out.merge(home_base, on="home_team", how="left")
        out = out.merge(away_base, on="away_team", how="left")
    else:
        for c in [
            "home_avg_scored", "home_avg_allowed", "home_road_avg_scored", "home_road_avg_allowed",
            "away_home_avg_scored", "away_home_avg_allowed", "away_avg_scored", "away_avg_allowed",
        ]:
            out[c] = np.nan

    hist_home_scoring = (
        0.65 * to_num(out["home_avg_scored"]).fillna(21.5)
        + 0.35 * to_num(out["away_avg_allowed"]).fillna(21.5)
    )
    hist_away_scoring = (
        0.65 * to_num(out["away_avg_scored"]).fillna(20.5)
        + 0.35 * to_num(out["home_avg_allowed"]).fillna(20.5)
    )

    market_total = to_num(out["market_total"])
    market_spread = to_num(out["market_spread_home"])

    out["home_score_mean_raw"] = np.where(
        market_total.notna() & market_spread.notna(),
        ((market_total + market_spread) / 2.0),
        hist_home_scoring,
    )
    out["away_score_mean_raw"] = np.where(
        market_total.notna() & market_spread.notna(),
        ((market_total - market_spread) / 2.0),
        hist_away_scoring,
    )

    out["proj_margin_raw"] = (
        0.70 * market_spread.fillna(0.0)
        + 0.45 * to_num(out["power_diff_home"]).fillna(0.0)
        + 0.30 * to_num(out["recent_form_diff_home"]).fillna(0.0)
        + 0.20 * to_num(out["qb_diff_home"]).fillna(0.0)
        + 0.15 * to_num(out["offense_diff_home"]).fillna(0.0)
        - 0.10 * to_num(out["defense_diff_home"]).fillna(0.0)
        + 1.00 * to_num(out["home_field_pts"]).fillna(1.75)
        + 0.10 * to_num(out["rest_diff_home"]).fillna(0.0)
        - 0.10 * to_num(out["injury_diff_home"]).fillna(0.0)
    )

    hist_total = out["home_score_mean_raw"] + out["away_score_mean_raw"]
    out["proj_total_raw"] = np.where(
        market_total.notna(),
        market_total
        + 0.60 * to_num(out["pace_diff_home"]).fillna(0.0)
        + 0.12 * to_num(out["recent_form_diff_home"]).fillna(0.0).abs()
        - to_num(out["weather_total_penalty"]).fillna(0.0),
        hist_total
        + 0.60 * to_num(out["pace_diff_home"]).fillna(0.0)
        + 0.12 * to_num(out["recent_form_diff_home"]).fillna(0.0).abs()
        - to_num(out["weather_total_penalty"]).fillna(0.0),
    )

    out["proj_margin_home"] = np.where(
        market_spread.notna(),
        0.75 * market_spread + 0.25 * out["proj_margin_raw"],
        out["proj_margin_raw"],
    )

    out["proj_total"] = np.where(
        market_total.notna(),
        0.78 * market_total + 0.22 * out["proj_total_raw"],
        out["proj_total_raw"],
    )

    out["proj_home_points"] = np.maximum(10, (out["proj_total"] + out["proj_margin_home"]) / 2.0)
    out["proj_away_points"] = np.maximum(10, (out["proj_total"] - out["proj_margin_home"]) / 2.0)

    out["median_home_points"] = np.round(out["proj_home_points"]).astype(int)
    out["median_away_points"] = np.round(out["proj_away_points"]).astype(int)

    margin_sd = 6.5
    total_sd = 10.5
    erf = np.vectorize(math.erf)

    def norm_cdf(x, mean, sd):
        z = (x - mean) / (sd * np.sqrt(2))
        return 0.5 * (1 + erf(z))

    out["sim_win_prob_home"] = norm_cdf(0.0, out["proj_margin_home"], margin_sd)
    out["sim_win_prob_away"] = 1.0 - out["sim_win_prob_home"]

    spread_anchor = market_spread.fillna(0.0)
    total_anchor = market_total.fillna(out["proj_total"])

    out["sim_cover_prob_home"] = 1.0 - norm_cdf(-spread_anchor, out["proj_margin_home"], margin_sd)
    out["sim_cover_prob_away"] = 1.0 - out["sim_cover_prob_home"]

    out["sim_over_prob"] = 1.0 - norm_cdf(total_anchor, out["proj_total"], total_sd)
    out["sim_under_prob"] = 1.0 - out["sim_over_prob"]

    out["home_ml_edge"] = out["sim_win_prob_home"] - out["market_home_implied"]
    out["away_ml_edge"] = out["sim_win_prob_away"] - out["market_away_implied"]
    out["home_spread_edge"] = out["sim_cover_prob_home"] - 0.5238
    out["away_spread_edge"] = out["sim_cover_prob_away"] - 0.5238
    out["over_edge"] = out["sim_over_prob"] - 0.5238
    out["under_edge"] = out["sim_under_prob"] - 0.5238

    out["projection_winner"] = np.where(
        out["sim_win_prob_home"].fillna(0.5) >= out["sim_win_prob_away"].fillna(0.5),
        out["home_team"],
        out["away_team"],
    )
    out["projection_winner_prob"] = np.maximum(out["sim_win_prob_home"], out["sim_win_prob_away"])
    out["value_ml_side"] = np.where(
        out["home_ml_edge"].fillna(-999) >= out["away_ml_edge"].fillna(-999),
        out["home_team"] + " ML",
        out["away_team"] + " ML",
    )

    out["home_ml_decision_score"] = blended_side_score(out["sim_win_prob_home"], out["home_ml_edge"])
    out["away_ml_decision_score"] = blended_side_score(out["sim_win_prob_away"], out["away_ml_edge"])
    out["home_spread_decision_score"] = blended_side_score(out["sim_cover_prob_home"], out["home_spread_edge"], 0.68, 0.32)
    out["away_spread_decision_score"] = blended_side_score(out["sim_cover_prob_away"], out["away_spread_edge"], 0.68, 0.32)
    out["over_decision_score"] = blended_side_score(out["sim_over_prob"], out["over_edge"], 0.68, 0.32)
    out["under_decision_score"] = blended_side_score(out["sim_under_prob"], out["under_edge"], 0.68, 0.32)

    out["best_ml_side"] = np.where(
        out["home_ml_decision_score"].fillna(-999) >= out["away_ml_decision_score"].fillna(-999),
        out["home_team"] + " ML",
        out["away_team"] + " ML",
    )
    out["best_ml_edge"] = np.where(
        out["home_ml_decision_score"].fillna(-999) >= out["away_ml_decision_score"].fillna(-999),
        out["home_ml_edge"],
        out["away_ml_edge"],
    )
    out["best_ml_decision_score"] = np.maximum(out["home_ml_decision_score"].fillna(-999), out["away_ml_decision_score"].fillna(-999))
    out["best_ml_confidence"] = confidence_from_signal(out["best_ml_decision_score"] - 0.50, base=52, scale=120)

    out["best_spread_side"] = np.where(
        out["home_spread_decision_score"].fillna(-999) >= out["away_spread_decision_score"].fillna(-999),
        out["home_team"],
        out["away_team"],
    )
    out["best_spread_line"] = np.where(
        out["home_spread_decision_score"].fillna(-999) >= out["away_spread_decision_score"].fillna(-999),
        out["market_spread_home"],
        out["market_spread_away"],
    )
    out["best_spread_edge"] = np.where(
        out["home_spread_decision_score"].fillna(-999) >= out["away_spread_decision_score"].fillna(-999),
        out["home_spread_edge"],
        out["away_spread_edge"],
    )
    out["best_spread_decision_score"] = np.maximum(out["home_spread_decision_score"].fillna(-999), out["away_spread_decision_score"].fillna(-999))
    out["best_spread_confidence"] = confidence_from_signal(out["best_spread_decision_score"] - 0.50, base=50, scale=115)

    out["best_total_side"] = np.where(
        out["over_decision_score"].fillna(-999) >= out["under_decision_score"].fillna(-999),
        "Over",
        "Under",
    )
    out["best_total_line"] = out["market_total"]
    out["best_total_edge"] = np.where(
        out["over_decision_score"].fillna(-999) >= out["under_decision_score"].fillna(-999),
        out["over_edge"],
        out["under_edge"],
    )
    out["best_total_decision_score"] = np.maximum(out["over_decision_score"].fillna(-999), out["under_decision_score"].fillna(-999))
    out["best_total_confidence"] = confidence_from_signal(out["best_total_decision_score"] - 0.50, base=49, scale=110)

    out["best_overall_market"] = np.select(
        [
            (out["best_ml_decision_score"] >= out["best_spread_decision_score"]) & (out["best_ml_decision_score"] >= out["best_total_decision_score"]),
            (out["best_spread_decision_score"] >= out["best_ml_decision_score"]) & (out["best_spread_decision_score"] >= out["best_total_decision_score"]),
        ],
        ["moneyline", "spread"],
        default="total",
    )

    spread_label = out["best_spread_side"] + " " + out["best_spread_line"].map(
        lambda x: f"{float(x):+0.1f}" if pd.notna(x) else ""
    )
    total_label = out["best_total_side"] + " " + out["best_total_line"].map(
        lambda x: f"{float(x):0.1f}" if pd.notna(x) else ""
    )

    out["best_overall_label"] = np.where(
        out["best_overall_market"] == "moneyline",
        out["best_ml_side"],
        np.where(
            out["best_overall_market"] == "spread",
            spread_label,
            total_label,
        ),
    )

    out["best_overall_edge"] = np.select(
        [out["best_overall_market"] == "moneyline", out["best_overall_market"] == "spread"],
        [out["best_ml_edge"], out["best_spread_edge"]],
        default=out["best_total_edge"],
    )
    out["best_overall_decision_score"] = np.select(
        [out["best_overall_market"] == "moneyline", out["best_overall_market"] == "spread"],
        [out["best_ml_decision_score"], out["best_spread_decision_score"]],
        default=out["best_total_decision_score"],
    )

    out["projection_alignment_bonus"] = np.select(
        [
            (out["best_overall_market"] == "moneyline") & (out["best_ml_side"] == (out["projection_winner"] + " ML")),
            (out["best_overall_market"] == "spread") & (
                ((out["proj_margin_home"] >= 0) & (out["best_spread_side"] == out["home_team"])) |
                ((out["proj_margin_home"] < 0) & (out["best_spread_side"] == out["away_team"]))
            ),
            (out["best_overall_market"] == "total") & (
                ((out["proj_total"] >= out["market_total"]) & (out["best_total_side"] == "Over")) |
                ((out["proj_total"] < out["market_total"]) & (out["best_total_side"] == "Under"))
            ),
        ],
        [3.0, 2.0, 2.0],
        default=-2.0,
    )

    out["best_overall_confidence"] = confidence_from_signal(out["best_overall_decision_score"] - 0.50, base=50, scale=120)
    out["fort_knox_score"] = clip(
        out["best_overall_confidence"]
        + 120 * out["best_overall_edge"].fillna(0.0)
        + out["projection_alignment_bonus"],
        25,
        95,
    )

    out = add_simulation_summaries(out, n_sims=n_sims, seed=sim_seed)

    desired = [
        "season","week","game_id","event_id","game_date","commence_time","away_team","home_team","matchup",
        "market_spread_home","market_spread_away","market_total","market_home_ml","market_away_ml",
        "close_spread_home","close_spread_away","close_total","close_home_ml","close_away_ml",
        "market_home_implied","market_away_implied","close_home_implied","close_away_implied",
        "home_spread_clv","away_spread_clv","over_clv","under_clv","home_ml_clv","away_ml_clv",
        "power_diff_home","recent_form_diff_home","qb_diff_home","offense_diff_home","defense_diff_home",
        "pace_diff_home","home_field_pts","rest_diff_home","injury_diff_home","weather_penalty_pts",
        "weather_total_penalty",
        "home_score_mean_raw","away_score_mean_raw","proj_margin_raw","proj_total_raw","proj_margin_home",
        "proj_total","proj_home_points","proj_away_points","median_home_points","median_away_points",
        "sim_win_prob_home","sim_win_prob_away","sim_cover_prob_home","sim_cover_prob_away",
        "sim_over_prob","sim_under_prob",
        "home_ml_edge","away_ml_edge","home_spread_edge","away_spread_edge","over_edge","under_edge",
        "projection_winner","projection_winner_prob","value_ml_side",
        "home_ml_decision_score","away_ml_decision_score","home_spread_decision_score","away_spread_decision_score",
        "over_decision_score","under_decision_score",
        "best_ml_side","best_ml_edge","best_ml_decision_score","best_ml_confidence",
        "best_spread_side","best_spread_line","best_spread_edge","best_spread_decision_score","best_spread_confidence",
        "best_total_side","best_total_line","best_total_edge","best_total_decision_score","best_total_confidence",
        "best_overall_market","best_overall_label","best_overall_edge","best_overall_decision_score","best_overall_confidence",
        "projection_alignment_bonus","fort_knox_score",
        "sim_home_points_avg","sim_home_points_min","sim_home_points_max","sim_home_points_p10","sim_home_points_p90",
        "sim_away_points_avg","sim_away_points_min","sim_away_points_max","sim_away_points_p10","sim_away_points_p90",
        "sim_total_points_avg","sim_total_points_min","sim_total_points_max","sim_total_points_p10","sim_total_points_p90",
        "sim_margin_avg","sim_margin_min","sim_margin_max","sim_margin_p10","sim_margin_p90",
        "market_snapshot_timestamp","close_snapshot_timestamp","market_books","close_books"
    ]

    for c in desired:
        if c not in out.columns:
            out[c] = np.nan

    result = out[desired].sort_values(
        ["week", "game_date", "away_team", "home_team"]
    ).reset_index(drop=True)

    return result


# ------------------------------------------------------------
# Simulation summaries / blended decisioning
# ------------------------------------------------------------

def add_simulation_summaries(out: pd.DataFrame, n_sims: int = 50000, seed: int = 42) -> pd.DataFrame:
    if out.empty:
        return out

    rng = np.random.default_rng(seed)
    n = len(out)
    margin_mean = to_num(out["proj_margin_home"]).fillna(0.0).to_numpy(dtype=float)[:, None]
    total_mean = to_num(out["proj_total"]).fillna(0.0).to_numpy(dtype=float)[:, None]

    margin_sd = 6.5
    total_sd = 10.5

    sim_margin = rng.normal(loc=margin_mean, scale=margin_sd, size=(n, n_sims))
    sim_total = rng.normal(loc=total_mean, scale=total_sd, size=(n, n_sims))

    sim_home = np.maximum(0.0, (sim_total + sim_margin) / 2.0)
    sim_away = np.maximum(0.0, (sim_total - sim_margin) / 2.0)
    sim_game_total = sim_home + sim_away
    sim_margin_pts = sim_home - sim_away

    def add_summary(prefix: str, arr: np.ndarray) -> None:
        out[f"{prefix}_avg"] = arr.mean(axis=1)
        out[f"{prefix}_min"] = arr.min(axis=1)
        out[f"{prefix}_max"] = arr.max(axis=1)
        out[f"{prefix}_p10"] = np.percentile(arr, 10, axis=1)
        out[f"{prefix}_p90"] = np.percentile(arr, 90, axis=1)

    add_summary("sim_home_points", sim_home)
    add_summary("sim_away_points", sim_away)
    add_summary("sim_total_points", sim_game_total)
    add_summary("sim_margin", sim_margin_pts)
    return out


def blended_side_score(prob: pd.Series, edge: pd.Series, projection_weight: float = 0.72, edge_weight: float = 0.28) -> pd.Series:
    p = to_num(prob).fillna(0.5)
    e = to_num(edge).fillna(0.0).clip(-0.20, 0.20)
    return projection_weight * p + edge_weight * e


def confidence_from_signal(signal: pd.Series, base: float = 50.0, scale: float = 100.0) -> pd.Series:
    return clip(base + scale * signal, 25, 95)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build game_projection_master using historical odds and real score baselines.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-sims", type=int, default=20000)
    parser.add_argument("--sim-seed", type=int, default=42)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else find_root(script_path)

    games, weekly_markets = load_week_games(root, args.season, args.week)
    if games.empty:
        raise SystemExit(f"Missing cr_games for season={args.season} week={args.week}")

    games = attach_event_id_from_markets(games, weekly_markets)
    if games["event_id"].astype(str).str.len().eq(0).all():
        raise SystemExit("Weekly games still have no usable event_id values after merging from cr_markets.")

    odds_raw = load_primary_odds(root)
    if odds_raw.empty:
        raise SystemExit("No historical odds source found under exports/historical_odds")

    odds = prep_odds_for_merge(odds_raw, args.season, args.week, games)
    if odds.empty:
        raise SystemExit(f"No matching odds rows found for season={args.season} week={args.week}")

    scores_hist = load_scores_history(root)
    score_baselines = prep_score_baselines(scores_hist, args.season)

    out = build_projection_master(games, odds, score_baselines)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = root / out_path
    else:
        out_dir = root / "exports" / "canonical"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"game_projection_master_{args.season}_w{args.week}.csv"

    out.to_csv(out_path, index=False)

    print(f"[OK] wrote {out_path}")
    print(f"[OK] rows: {len(out)}")
    print("[OK] sample:")
    print(out[[
        "matchup", "market_spread_home", "market_total",
        "proj_home_points", "proj_away_points", "best_overall_label"
    ]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()