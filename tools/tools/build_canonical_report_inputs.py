from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
CANONICAL = EXPORTS / "canonical"


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        hit = lower_map.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _week_series(df: pd.DataFrame) -> pd.Series:
    col = _find_col(df, ["Week", "week", "week_num", "week_number"])
    if col is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    s = df[col].astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _season_series(df: pd.DataFrame) -> pd.Series:
    col = _find_col(df, ["Season", "season", "year"])
    if col is not None:
        return pd.to_numeric(df[col], errors="coerce").astype("Int64")

    date_col = _find_col(df, ["game_date", "Date", "date", "commence_time"])
    if date_col is not None:
        return pd.to_datetime(df[date_col], errors="coerce").dt.year.astype("Int64")

    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")


def _filter_season_week(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_season"] = _season_series(out)
    out["_week"] = _week_series(out)
    return out[(out["_season"] == season) & (out["_week"] == week)].copy()


def _ensure_home_away(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    away_col = _find_col(
        out,
        [
            "away_team", "away", "AwayTeam", "visitor_team", "visitor",
            "team_away", "away_team_bet", "awayName"
        ],
    )
    home_col = _find_col(
        out,
        [
            "home_team", "home", "HomeTeam", "host",
            "team_home", "home_team_bet", "homeName"
        ],
    )

    if away_col is not None and away_col != "away_team":
        out["away_team"] = out[away_col]
    if home_col is not None and home_col != "home_team":
        out["home_team"] = out[home_col]

    if "away_team" in out.columns:
        out["away_team"] = out["away_team"].astype(str).str.strip()
    if "home_team" in out.columns:
        out["home_team"] = out["home_team"].astype(str).str.strip()

    return out


def _ensure_matchup_game_id(df: pd.DataFrame, season: int | None = None, week: int | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    out = _ensure_home_away(df.copy())

    if "matchup" not in out.columns and {"away_team", "home_team"}.issubset(out.columns):
        out["matchup"] = out["away_team"].astype(str) + " @ " + out["home_team"].astype(str)

    if "game_id" not in out.columns and {"away_team", "home_team"}.issubset(out.columns):
        season_part = out["_season"] if "_season" in out.columns else season
        week_part = out["_week"] if "_week" in out.columns else week

        if not isinstance(season_part, pd.Series):
            season_part = pd.Series([season] * len(out), index=out.index)
        if not isinstance(week_part, pd.Series):
            week_part = pd.Series([week] * len(out), index=out.index)

        out["game_id"] = (
            season_part.astype("Int64").astype(str).fillna("")
            + "_w"
            + week_part.astype("Int64").astype(str).fillna("")
            + "_"
            + out["away_team"].astype(str).str.lower().str.replace(r"\W+", "_", regex=True)
            + "_at_"
            + out["home_team"].astype(str).str.lower().str.replace(r"\W+", "_", regex=True)
        )

    return out


def _to_num(s: pd.Series | object) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _american_implied_prob(odds: pd.Series) -> pd.Series:
    x = _to_num(odds)
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    pos = x > 0
    neg = x < 0
    out.loc[pos] = 100.0 / (x.loc[pos] + 100.0)
    out.loc[neg] = (-x.loc[neg]) / ((-x.loc[neg]) + 100.0)
    return out


def _derive_confidence_from_edge(edge: pd.Series) -> pd.Series:
    e = _to_num(edge).fillna(0.0)
    conf = 50.0 + (e * 100.0 * 35.0)
    return conf.clip(lower=35.0, upper=95.0)


def build_games(season: int, week: int) -> pd.DataFrame:
    candidates = [
        EXPORTS / "games_master_recent_form_market_regime.csv",
        EXPORTS / "games_master_recent_form_market.csv",
        EXPORTS / "games_master_recent_form.csv",
        EXPORTS / "games_master.csv",
    ]

    df = pd.DataFrame()
    chosen = None
    for src in candidates:
        trial = read_csv_safe(src)
        if trial.empty:
            continue
        trial = _filter_season_week(trial, season, week)
        if not trial.empty:
            df = trial
            chosen = src
            break

    print(f"[INFO] build_games source: {chosen if chosen else 'NONE'}")
    if df.empty:
        return df

    # explicit team mapping from your actual source columns
    if "away_team" not in df.columns:
        away_col = _find_col(df, ["away_team_x", "away_team_y", "AwayTeam", "away"])
        if away_col is not None:
            df["away_team"] = df[away_col]

    if "home_team" not in df.columns:
        home_col = _find_col(df, ["home_team_x", "home_team_y", "HomeTeam", "home"])
        if home_col is not None:
            df["home_team"] = df[home_col]

    df = _ensure_matchup_game_id(df, season, week)
    if df.empty:
        return df

    def col_or_nan(name_list):
        c = _find_col(df, name_list)
        return _to_num(df[c]) if c is not None else pd.Series(np.nan, index=df.index)

    # offense and defense baselines
    home_pf = pd.concat(
        [
            col_or_nan(["home_points_for_avg_last3"]),
            col_or_nan(["home_points_for_avg_last5"]),
            col_or_nan(["home_points_for_avg_season_prior"]),
        ],
        axis=1,
    ).mean(axis=1)

    away_pf = pd.concat(
        [
            col_or_nan(["away_points_for_avg_last3"]),
            col_or_nan(["away_points_for_avg_last5"]),
            col_or_nan(["away_points_for_avg_season_prior"]),
        ],
        axis=1,
    ).mean(axis=1)

    home_pa = pd.concat(
        [
            col_or_nan(["home_points_against_avg_last3"]),
            col_or_nan(["home_points_against_avg_last5"]),
            col_or_nan(["home_points_against_avg_season_prior"]),
        ],
        axis=1,
    ).mean(axis=1)

    away_pa = pd.concat(
        [
            col_or_nan(["away_points_against_avg_last3"]),
            col_or_nan(["away_points_against_avg_last5"]),
            col_or_nan(["away_points_against_avg_season_prior"]),
        ],
        axis=1,
    ).mean(axis=1)

    # trend / team-strength adjustments
    home_win_form = pd.concat(
        [
            col_or_nan(["home_win_avg_last3"]),
            col_or_nan(["home_win_avg_last5"]),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.5)

    away_win_form = pd.concat(
        [
            col_or_nan(["away_win_avg_last3"]),
            col_or_nan(["away_win_avg_last5"]),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.5)

    home_trend = col_or_nan(["home_points_for_trend_last3_vs_season"]).fillna(0.0)
    away_trend = col_or_nan(["away_points_for_trend_last3_vs_season"]).fillna(0.0)

    # give home side a real HFA so scores are not mirror-equal
    home_field_adv = 1.5

    base_margin = (
        (home_pf - away_pa) - (away_pf - home_pa)
    ) / 2.0

    form_margin = (
        (home_win_form - away_win_form) * 6.0
        + (home_trend - away_trend) * 0.75
    )

    projected_margin = base_margin + form_margin + home_field_adv

    projected_total = (
        ((home_pf + away_pa) / 2.0) + ((away_pf + home_pa) / 2.0)
    )
    # shrink raw model margin toward market
    market_margin = (-1.0 * market_spread).fillna(0.0)

    # raw projected_margin should already exist above this line
    projected_margin = (0.60 * market_margin) + (0.40 * projected_margin)

    model_home_score = ((projected_total + projected_margin) / 2.0).clip(lower=10).round(1)
    model_away_score = ((projected_total - projected_margin) / 2.0).clip(lower=10).round(1)
    model_home_score = ((projected_total + projected_margin) / 2.0).clip(lower=10).round(1)
    model_away_score = ((projected_total - projected_margin) / 2.0).clip(lower=10).round(1)

    # if explicit projected scores exist, use them where present
    home_score_model_col = _find_col(df, ["model_home_score", "pred_home_score", "home_score_model"])
    away_score_model_col = _find_col(df, ["model_away_score", "pred_away_score", "away_score_model"])

    if home_score_model_col is not None:
        explicit_home = _to_num(df[home_score_model_col])
        model_home_score = explicit_home.fillna(model_home_score)

    if away_score_model_col is not None:
        explicit_away = _to_num(df[away_score_model_col])
        model_away_score = explicit_away.fillna(model_away_score)

    # pull market lines from lines_master
    market_src = read_csv_safe(EXPORTS / "lines_master.csv")
    market_src = _filter_season_week(market_src, season, week)
    market_src = _ensure_matchup_game_id(market_src, season, week)

    market_spread = pd.Series(np.nan, index=df.index)
    market_total = pd.Series(np.nan, index=df.index)

    if not market_src.empty:
        # explicit team mapping in market file
        if "away_team" not in market_src.columns:
            away_m = _find_col(market_src, ["away_team", "away_team_x", "away_team_y", "AwayTeam", "away"])
            if away_m is not None:
                market_src["away_team"] = market_src[away_m]

        if "home_team" not in market_src.columns:
            home_m = _find_col(market_src, ["home_team", "home_team_x", "home_team_y", "HomeTeam", "home"])
            if home_m is not None:
                market_src["home_team"] = market_src[home_m]

        spread_col = _find_col(market_src, ["market_spread", "spread", "spread_home", "closing_spread_home"])
        total_col = _find_col(market_src, ["market_total", "total", "total_close", "closing_total"])

        ts_col = _find_col(market_src, ["snapshot_timestamp", "requested_snapshot", "commence_time", "game_date"])
        if ts_col is not None:
            market_src["_sort_ts"] = pd.to_datetime(market_src[ts_col], errors="coerce")
            market_src = market_src.sort_values(["game_id", "_sort_ts"]).groupby("game_id", as_index=False).tail(1)
        else:
            market_src = market_src.drop_duplicates(subset=["game_id"], keep="last")

        # 1) try merge on game_id
        merge_cols = ["game_id"]
        if spread_col is not None:
            merge_cols.append(spread_col)
        if total_col is not None:
            merge_cols.append(total_col)

        merged = df[["game_id"]].merge(market_src[merge_cols], on="game_id", how="left")

        if spread_col is not None:
            market_spread = _to_num(merged[spread_col])
        if total_col is not None:
            market_total = _to_num(merged[total_col])

        # 2) fallback merge on away/home if game_id merge misses
        if market_spread.isna().all() or market_total.isna().all():
            team_merge_cols = ["away_team", "home_team"]
            if spread_col is not None:
                team_merge_cols.append(spread_col)
            if total_col is not None:
                team_merge_cols.append(total_col)

            merged2 = df[["away_team", "home_team"]].merge(
                market_src[team_merge_cols].drop_duplicates(subset=["away_team", "home_team"]),
                on=["away_team", "home_team"],
                how="left",
            )

            if spread_col is not None and market_spread.isna().all():
                market_spread = _to_num(merged2[spread_col])
            if total_col is not None and market_total.isna().all():
                market_total = _to_num(merged2[total_col])

    # fallback only if market lines still missing
    model_margin = (model_home_score - model_away_score).round(1)
    implied_total = (model_home_score + model_away_score).round(1)

    market_total = market_total.fillna(implied_total)
    market_spread = market_spread.fillna((-1.0 * model_margin).round(1))

    out = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "away_team": df["away_team"].astype(str).str.strip(),
            "home_team": df["home_team"].astype(str).str.strip(),
            "matchup": df["away_team"].astype(str).str.strip() + " @ " + df["home_team"].astype(str).str.strip(),
            "market_spread": market_spread,
            "market_total": market_total,
            "model_away_score": model_away_score,
            "model_home_score": model_home_score,
            "Season": df["_season"],
            "Week": df["_week"],
            "game_date": df[_find_col(df, ["game_date", "Date", "date", "commence_time"])] if _find_col(df, ["game_date", "Date", "date", "commence_time"]) else pd.NA,
            "game_datetime_utc": df[_find_col(df, ["game_datetime_utc", "commence_time"])] if _find_col(df, ["game_datetime_utc", "commence_time"]) else pd.NA,
        }
    )

    return out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)


def build_edges(season: int, week: int) -> pd.DataFrame:
    src = EXPORTS / "edges_master.csv"
    df = read_csv_safe(src)
    df = _filter_season_week(df, season, week)
    df = _ensure_matchup_game_id(df, season, week)

    if df.empty:
        return pd.DataFrame()

    if "away_team" not in df.columns:
        away_col = _find_col(df, ["away_team", "away_team_bet", "away", "AwayTeam"])
        if away_col is not None:
            df["away_team"] = df[away_col]

    if "home_team" not in df.columns:
        home_col = _find_col(df, ["home_team", "home_team_bet", "home", "HomeTeam"])
        if home_col is not None:
            df["home_team"] = df[home_col]

    df = _ensure_matchup_game_id(df, season, week)

    market_col = _find_col(df, ["market", "market_type"])
    side_col = _find_col(df, ["side", "selected_team", "selection", "pick"])
    odds_col = _find_col(df, ["odds", "closing_odds", "price", "true_moneyline"])
    line_col = _find_col(df, ["line", "market_spread", "market_total"])
    model_prob_col = _find_col(df, ["model_prob", "p_win", "win_prob", "pred_prob"])
    implied_prob_col = _find_col(df, ["implied_prob", "market_prob", "close_prob", "bet_prob"])

    if model_prob_col is None:
        return pd.DataFrame()

    if implied_prob_col is None and odds_col is not None:
        df["implied_prob"] = _american_implied_prob(df[odds_col])
        implied_prob_col = "implied_prob"

    if implied_prob_col is None:
        df["implied_prob"] = 110.0 / 210.0
        implied_prob_col = "implied_prob"

    df["model_prob"] = _to_num(df[model_prob_col])
    df["implied_prob"] = _to_num(df[implied_prob_col])
    df["edge"] = df["model_prob"] - df["implied_prob"]

    if market_col is None:
        df["market"] = "moneyline"
        market_col = "market"

    if side_col is None:
        df["side"] = "home"
        side_col = "side"

    if "label" not in df.columns:
        def make_label(row):
            side = str(row[side_col]).strip()
            market = str(row[market_col]).strip().lower()
            line = row[line_col] if line_col is not None else None

            if market == "moneyline":
                return f"{side} ML"

            if market == "spread":
                try:
                    return f"{side} {float(line):+.1f}"
                except Exception:
                    return f"{side} spread"

            if market == "total":
                try:
                    if side.lower() in ["over", "under"]:
                        return f"{side.title()} {float(line):.1f}"
                except Exception:
                    pass
                return f"{side} total"

            if "team_total" in market:
                try:
                    return f"{side} Team Total {float(line):.1f}"
                except Exception:
                    return f"{side} team total"

            return f"{side} {market}".strip()

        df["label"] = df.apply(make_label, axis=1)

    if "confidence" not in df.columns:
        df["confidence"] = (55.0 + (_to_num(df["edge"]).fillna(0.0) * 180.0)).clip(40.0, 95.0)

    keep = [
        "game_id",
        "_season",
        "_week",
        "game_date",
        "away_team",
        "home_team",
        "matchup",
        "market",
        "side",
        "label",
        "odds",
        "model_prob",
        "implied_prob",
        "edge",
        "confidence",
    ]
    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()
    out = out.rename(columns={"_season": "Season", "_week": "Week"})
    out = out.sort_values("edge", ascending=False).reset_index(drop=True)
    return out

def build_markets(season: int, week: int) -> pd.DataFrame:
    src = EXPORTS / "lines_master.csv"
    df = read_csv_safe(src)
    df = _filter_season_week(df, season, week)
    df = _ensure_matchup_game_id(df, season, week)
    if df.empty:
        return df

    ts_col = _find_col(df, ["snapshot_timestamp", "requested_snapshot", "commence_time", "game_date"])
    if ts_col:
        df["_sort_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.sort_values(["game_id", "_sort_ts"])
        df = df.groupby("game_id", as_index=False).tail(1)

    out = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "event_id": df[_find_col(df, ["event_id"])] if _find_col(df, ["event_id"]) else pd.NA,
            "requested_snapshot": df[_find_col(df, ["requested_snapshot"])] if _find_col(df, ["requested_snapshot"]) else pd.NA,
            "snapshot_timestamp": df[_find_col(df, ["snapshot_timestamp"])] if _find_col(df, ["snapshot_timestamp"]) else pd.NA,
            "commence_time": df[_find_col(df, ["commence_time"])] if _find_col(df, ["commence_time"]) else pd.NA,
            "game_date": df[_find_col(df, ["game_date", "Date", "date"])] if _find_col(df, ["game_date", "Date", "date"]) else pd.NA,
            "season": df["_season"],
            "week": df["_week"],
            "away_team": df["away_team"],
            "home_team": df["home_team"],
            "book_key": df[_find_col(df, ["book_key", "book"])] if _find_col(df, ["book_key", "book"]) else pd.NA,
            "book_title": df[_find_col(df, ["book_title"])] if _find_col(df, ["book_title"]) else pd.NA,
            "market_spread": df[_find_col(df, ["market_spread", "spread", "spread_home", "closing_spread_home"])] if _find_col(df, ["market_spread", "spread", "spread_home", "closing_spread_home"]) else pd.NA,
            "market_total": df[_find_col(df, ["market_total", "total", "total_close", "closing_total"])] if _find_col(df, ["market_total", "total", "total_close", "closing_total"]) else pd.NA,
            "matchup": df["matchup"],
        }
    )

    out["market_spread"] = _to_num(out["market_spread"])
    out["market_total"] = _to_num(out["market_total"])

    return out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)


def build_scores(season: int, week: int) -> pd.DataFrame:
    src = EXPORTS / "scores_master.csv"
    df = read_csv_safe(src)
    df = _filter_season_week(df, season, week)
    df = _ensure_matchup_game_id(df, season, week)
    if df.empty:
        return df

    home_score_col = _find_col(df, ["home_score", "HomeScore", "score_home"])
    away_score_col = _find_col(df, ["away_score", "AwayScore", "score_away"])

    home_score = _to_num(df[home_score_col]) if home_score_col else pd.Series(np.nan, index=df.index)
    away_score = _to_num(df[away_score_col]) if away_score_col else pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "Season": df["_season"],
            "week_num": df["_week"],
            "Date": df[_find_col(df, ["Date", "game_date", "date"])] if _find_col(df, ["Date", "game_date", "date"]) else pd.NA,
            "AwayTeam": df["away_team"],
            "HomeTeam": df["home_team"],
            "home_score": home_score,
            "away_score": away_score,
            "margin": home_score - away_score,
            "total_points": home_score + away_score,
            "home_win": (home_score > away_score).astype("Int64"),
            "away_win": (away_score > home_score).astype("Int64"),
            "matchup": df["matchup"],
        }
    )
    return out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)


def build_parlays(season: int, week: int) -> pd.DataFrame:
    candidates = [
        EXPORTS / "data" / "parlay_scores.csv",
        EXPORTS / "parlay_scores.csv",
    ]

    df = pd.DataFrame()
    for src in candidates:
        trial = read_csv_safe(src)
        if trial.empty:
            continue
        trial = _filter_season_week(trial, season, week)
        if not trial.empty:
            df = trial
            break

    if df.empty:
        return df

    df = _ensure_matchup_game_id(df, season, week)

    def parse_matchup_from_game_id(g):
        try:
            parts = str(g).split("_")
            if len(parts) >= 4:
                return f"{parts[-2]} @ {parts[-1]}"
        except Exception:
            pass
        return "Unknown matchup"

    if "matchup" not in df.columns:
        df["matchup"] = df["game_id"].map(parse_matchup_from_game_id)
    else:
        bad = df["matchup"].isna() | df["matchup"].astype(str).str.contains("nan @ nan", na=False)
        df.loc[bad, "matchup"] = df.loc[bad, "game_id"].map(parse_matchup_from_game_id)

    market_col = _find_col(df, ["market"])
    side_col = _find_col(df, ["side"])
    line_col = _find_col(df, ["line"])
    pwin_col = _find_col(df, ["p_win", "model_prob"])

    def make_parlay_label(row):
        matchup = row["matchup"]
        market = str(row[market_col]).lower() if market_col is not None and pd.notna(row[market_col]) else "moneyline"
        side = str(row[side_col]).strip() if side_col is not None and pd.notna(row[side_col]) else ""
        line = row[line_col] if line_col is not None else None

        if market == "moneyline" and side:
            return f"{matchup} | {side} ML"
        if market == "spread" and side:
            try:
                return f"{matchup} | {side} {float(line):+.1f}"
            except Exception:
                return f"{matchup} | {side} spread"
        if market == "total" and side:
            try:
                return f"{matchup} | {side.title()} {float(line):.1f}"
            except Exception:
                return f"{matchup} | {side} total"
        if side:
            return f"{matchup} | {side}"
        return f"{matchup} | Parlay leg"

    df["label"] = df.apply(make_parlay_label, axis=1)

    if "parlay_score" not in df.columns:
        if pwin_col is not None:
            df["parlay_score"] = (_to_num(df[pwin_col]).fillna(0.55) * 100.0).round(2)
        else:
            df["parlay_score"] = 55.0
    else:
        df["parlay_score"] = _to_num(df["parlay_score"]).fillna(55.0)

    out = pd.DataFrame(
        {
            "ts": df[_find_col(df, ["ts", "timestamp"])] if _find_col(df, ["ts", "timestamp"]) else pd.NA,
            "sport": df[_find_col(df, ["sport"])] if _find_col(df, ["sport"]) else "NFL",
            "league": df[_find_col(df, ["league"])] if _find_col(df, ["league"]) else "NFL",
            "game_id": df["game_id"],
            "season": df["_season"] if "_season" in df.columns else season,
            "week": df["_week"] if "_week" in df.columns else week,
            "market": df[market_col] if market_col is not None else "moneyline",
            "ref": df[_find_col(df, ["ref"])] if _find_col(df, ["ref"]) else pd.NA,
            "side": df[side_col] if side_col is not None else pd.NA,
            "line": df[line_col] if line_col is not None else pd.NA,
            "odds": df[_find_col(df, ["odds", "price"])] if _find_col(df, ["odds", "price"]) else pd.NA,
            "p_win": df[pwin_col] if pwin_col is not None else pd.NA,
            "label": df["label"],
            "parlay_score": df["parlay_score"],
            "matchup": df["matchup"],
        }
    )
    return out.reset_index(drop=True)
    # derive matchup from game_id if needed
    if ("matchup" not in df.columns) or df.get("matchup", pd.Series(dtype="object")).astype(str).str.contains("nan @ nan", na=False).all():
        gid = df["game_id"].astype(str)

        def parse_matchup(g):
            parts = g.split("_")
            if len(parts) >= 4:
                away = parts[-2]
                home = parts[-1]
                return f"{away} @ {home}"
            return "Parlay matchup"

        df["matchup"] = gid.map(parse_matchup)

    # better labels for edges
    if "label" not in df.columns:
        side_col = _find_col(df, ["side", "selection", "selected_team", "pick"])
        market_col = _find_col(df, ["market", "market_type"])

        if side_col is not None and market_col is not None:
            side_vals = df[side_col].astype(str).str.strip()
            market_vals = df[market_col].astype(str).str.strip().str.lower()

            line_col = _find_col(df, ["line", "market_spread", "market_total"])
            line_vals = df[line_col] if line_col is not None else pd.Series([pd.NA] * len(df), index=df.index)

            def make_edge_label(row):
                side = str(row["_side"]).strip()
                market = str(row["_market"]).strip().lower()
                line = row["_line"]

                if market == "moneyline":
                    return f"{side} ML"

                if market == "spread":
                    try:
                        return f"{side} {float(line):+.1f}"
                    except Exception:
                        return f"{side} spread"

                if market == "total":
                    side_low = side.lower()
                    try:
                        if side_low in ["over", "under"]:
                            return f"{side.title()} {float(line):.1f}"
                    except Exception:
                        pass
                    return f"{side} total"

                if "team_total" in market:
                    try:
                        return f"{side} Team Total {float(line):.1f}"
                    except Exception:
                        return f"{side} team total"

                return f"{side} {market}".strip()

            temp = pd.DataFrame({
                "_side": side_vals,
                "_market": market_vals,
                "_line": line_vals,
            })
            df["label"] = temp.apply(make_edge_label, axis=1)
        else:
            df["label"] = df["matchup"].astype(str)

    # derive parlay_score
    if "parlay_score" not in df.columns:
        pwin_col = _find_col(df, ["p_win", "model_prob"])
        if pwin_col is not None:
            df["parlay_score"] = (_to_num(df[pwin_col]).fillna(0.55) * 100.0).round(2)
        else:
            df["parlay_score"] = 55.0
    else:
        df["parlay_score"] = _to_num(df["parlay_score"]).fillna(55.0)

    out = pd.DataFrame(
        {
            "ts": df[_find_col(df, ["ts", "timestamp"])] if _find_col(df, ["ts", "timestamp"]) else pd.NA,
            "sport": df[_find_col(df, ["sport"])] if _find_col(df, ["sport"]) else "NFL",
            "league": df[_find_col(df, ["league"])] if _find_col(df, ["league"]) else "NFL",
            "game_id": df["game_id"] if "game_id" in df.columns else pd.NA,
            "season": df["_season"] if "_season" in df.columns else season,
            "week": df["_week"] if "_week" in df.columns else week,
            "market": df[_find_col(df, ["market"])] if _find_col(df, ["market"]) else "moneyline",
            "ref": df[_find_col(df, ["ref"])] if _find_col(df, ["ref"]) else pd.NA,
            "side": df[_find_col(df, ["side"])] if _find_col(df, ["side"]) else pd.NA,
            "line": df[_find_col(df, ["line"])] if _find_col(df, ["line"]) else pd.NA,
            "odds": df[_find_col(df, ["odds", "price"])] if _find_col(df, ["odds", "price"]) else pd.NA,
            "p_win": df[_find_col(df, ["p_win", "model_prob"])] if _find_col(df, ["p_win", "model_prob"]) else pd.NA,
            "label": df["label"],
            "parlay_score": df["parlay_score"],
            "matchup": df["matchup"],
        }
    )

    return out.reset_index(drop=True)

def save(df: pd.DataFrame, name: str, season: int, week: int) -> None:
    CANONICAL.mkdir(parents=True, exist_ok=True)
    path = CANONICAL / f"{name}_{season}_w{week}.csv"
    df.to_csv(path, index=False)
    print(f"[OK] {path} ({len(df)} rows)")
    if not df.empty:
        print(f"     columns: {list(df.columns[:12])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical weekly report inputs.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    season = args.season
    week = args.week

    print("[INFO] Building canonical report inputs")
    print(f"[INFO] exports dir: {EXPORTS}")

    games = build_games(season, week)
    edges = build_edges(season, week)
    markets = build_markets(season, week)
    scores = build_scores(season, week)
    parlays = build_parlays(season, week)

    save(games, "cr_games", season, week)
    save(edges, "cr_edges", season, week)
    save(markets, "cr_markets", season, week)
    save(scores, "cr_scores", season, week)
    save(parlays, "cr_parlay_scores", season, week)

    print("[DONE] Canonical report inputs built.")


if __name__ == "__main__":
    main()