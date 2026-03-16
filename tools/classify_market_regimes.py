from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEASON_CANDIDATES = ["season", "Season", "season_ui", "year"]
WEEK_CANDIDATES = ["week", "Week", "week_num", "week_number"]
GAME_ID_CANDIDATES = ["game_id", "event_id", "gameid", "eventId"]
SIDE_CANDIDATES = ["side", "selection", "team_side", "pick_side"]
MARKET_CANDIDATES = ["market", "market_type", "bet_type"]
OPEN_AM_CANDIDATES = ["open_odds", "open_american", "open_price", "odds_open", "book_open_odds"]
MID_AM_CANDIDATES = ["mid_odds", "mid_american", "mid_price", "odds_mid", "book_mid_odds"]
CLOSE_AM_CANDIDATES = ["close_odds", "close_american", "close_price", "odds_close", "book_close_odds", "closing_odds"]
MODEL_PROB_CANDIDATES = ["model_prob", "raw_model_prob"]
CAL_PROB_CANDIDATES = ["cal_prob", "calibrated_prob", "model_prob_calibrated"]
EDGE_OPEN_CANDIDATES = ["model_edge_vs_open", "edge_vs_open", "true_edge_open", "ev_edge_open"]
EDGE_CLOSE_CANDIDATES = ["model_edge_vs_close", "edge_vs_close", "true_edge_close", "ev_edge_close"]
OPEN_MID_MOVE_CANDIDATES = ["open_to_mid_prob_move", "prob_move_open_to_mid", "open_mid_prob_move"]
MID_CLOSE_MOVE_CANDIDATES = ["mid_to_close_prob_move", "prob_move_mid_to_close", "mid_close_prob_move"]
RESULT_CANDIDATES = ["result", "won", "win", "hit_bool", "outcome", "is_win"]
PAYOUT_ODDS_CANDIDATES = ["odds", "american_odds", "price", "bet_odds", "offered_odds", "picked_odds"]
DATE_CANDIDATES = ["game_date", "date", "commence_time", "event_date", "start_time"]


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def american_to_prob(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.where(s > 0, 100.0 / (s + 100.0), np.where(s < 0, np.abs(s) / (np.abs(s) + 100.0), np.nan))


def american_profit_per_1(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.where(s > 0, s / 100.0, np.where(s < 0, 100.0 / np.abs(s), np.nan))


def normalize_result(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        out = np.where(out > 0, 1.0, np.where(out == 0, 0.0, np.nan))
        return pd.Series(out, index=s.index, dtype="float64")
    t = s.astype(str).str.strip().str.lower()
    wins = {"1", "true", "w", "win", "won", "cash", "hit", "yes"}
    losses = {"0", "false", "l", "loss", "lost", "miss", "no"}
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[t.isin(wins)] = 1.0
    out[t.isin(losses)] = 0.0
    return out


def ensure_column(df: pd.DataFrame, name: str, candidates: list[str], default=np.nan) -> pd.Series:
    col = first_existing(df, candidates)
    if col is None:
        return pd.Series(default, index=df.index)
    return df[col]


def build_join_key(df: pd.DataFrame) -> pd.Series:
    season = ensure_column(df, "season", SEASON_CANDIDATES, default="").astype(str)
    week = ensure_column(df, "week", WEEK_CANDIDATES, default="").astype(str)
    game_id = ensure_column(df, "game_id", GAME_ID_CANDIDATES, default="").astype(str)
    side = ensure_column(df, "side", SIDE_CANDIDATES, default="").astype(str).str.lower().str.strip()
    market = ensure_column(df, "market", MARKET_CANDIDATES, default="").astype(str).str.lower().str.strip()
    key = season + "|" + week + "|" + game_id + "|" + market + "|" + side
    date_col = first_existing(df, DATE_CANDIDATES)
    if date_col and key.eq("||||").all():
        date_s = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        key = date_s + "|" + market + "|" + side
    return key


def add_movement_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "open_prob" not in df:
        df["open_prob"] = american_to_prob(ensure_column(df, "open_prob", OPEN_AM_CANDIDATES))
    if "mid_prob" not in df:
        df["mid_prob"] = american_to_prob(ensure_column(df, "mid_prob", MID_AM_CANDIDATES))
    if "close_prob" not in df:
        df["close_prob"] = american_to_prob(ensure_column(df, "close_prob", CLOSE_AM_CANDIDATES))

    raw_open_mid = ensure_column(df, "open_to_mid_prob_move", OPEN_MID_MOVE_CANDIDATES)
    raw_mid_close = ensure_column(df, "mid_to_close_prob_move", MID_CLOSE_MOVE_CANDIDATES)

    if pd.to_numeric(raw_open_mid, errors="coerce").notna().any():
        df["open_to_mid_prob_move"] = pd.to_numeric(raw_open_mid, errors="coerce")
    else:
        df["open_to_mid_prob_move"] = df["mid_prob"] - df["open_prob"]

    if pd.to_numeric(raw_mid_close, errors="coerce").notna().any():
        df["mid_to_close_prob_move"] = pd.to_numeric(raw_mid_close, errors="coerce")
    else:
        df["mid_to_close_prob_move"] = df["close_prob"] - df["mid_prob"]

    df["open_to_close_prob_move"] = df["close_prob"] - df["open_prob"]
    return df


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_move = df["open_to_close_prob_move"]
    first_leg = df["open_to_mid_prob_move"]
    last_leg = df["mid_to_close_prob_move"]
    edge_close = pd.to_numeric(df["model_edge_vs_close"], errors="coerce")

    df["move_size_bucket"] = pd.cut(
        total_move.abs(),
        bins=[-np.inf, 0.01, 0.025, 0.05, np.inf],
        labels=["flat", "small", "medium", "large"],
    ).astype("string")

    same_dir = np.sign(first_leg.fillna(0.0)) == np.sign(last_leg.fillna(0.0))
    reversal = np.sign(first_leg.fillna(0.0)) == -np.sign(last_leg.fillna(0.0))
    df["move_path"] = np.select(
        [
            first_leg.abs().fillna(0).lt(1e-9) & last_leg.abs().fillna(0).lt(1e-9),
            same_dir & total_move.gt(0),
            same_dir & total_move.lt(0),
            reversal,
        ],
        ["flat", "steam_up", "steam_down", "reversal"],
        default="mixed",
    )

    df["close_edge_bucket"] = pd.cut(
        edge_close,
        bins=[-np.inf, 0.0, 0.02, 0.05, 0.10, np.inf],
        labels=["negative", "thin", "good", "strong", "elite"],
    ).astype("string")

    df["market_confirmation"] = np.select(
        [
            edge_close.ge(0.02) & total_move.ge(0.01),
            edge_close.ge(0.02) & total_move.le(-0.01),
            edge_close.lt(0.02) & total_move.ge(0.01),
            edge_close.lt(0.0),
        ],
        ["confirmed", "faded_by_market", "market_only", "negative_close_edge"],
        default="neutral",
    )

    df["late_drift_flag"] = np.where(last_leg.abs().ge(0.015), "high_late_drift", "normal_late_drift")
    return df


def roi_from_result_and_odds(result: pd.Series, odds: pd.Series) -> pd.Series:
    profit = american_profit_per_1(odds)
    return np.where(result == 1, profit, np.where(result == 0, -1.0, np.nan))


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)
    out = g.agg(
        bets=("result", "size"),
        win_rate=("result", "mean"),
        roi=("roi_per_1", "mean"),
        avg_model_prob=("model_prob", "mean"),
        avg_cal_prob=("cal_prob", "mean"),
        avg_edge_open=("model_edge_vs_open", "mean"),
        avg_edge_close=("model_edge_vs_close", "mean"),
        avg_open_mid_move=("open_to_mid_prob_move", "mean"),
        avg_mid_close_move=("mid_to_close_prob_move", "mean"),
        pct_positive_close_edge=("positive_close_edge", "mean"),
    ).reset_index()
    out["win_rate"] = (100 * out["win_rate"]).round(2)
    out["roi"] = (100 * out["roi"]).round(2)
    out["pct_positive_close_edge"] = (100 * out["pct_positive_close_edge"]).round(2)
    return out.sort_values(["bets"], ascending=[False])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True, help="Path to fort_knox_market_joined_moneyline.csv")
    ap.add_argument("--odds", required=True, help="Path to nfl_open_mid_close_odds.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    joined = pd.read_csv(args.joined, low_memory=False)
    odds = pd.read_csv(args.odds, low_memory=False)

    joined["join_key"] = build_join_key(joined)
    odds["join_key"] = build_join_key(odds)

    merged = joined.merge(
        odds[[c for c in odds.columns if c not in joined.columns or c == "join_key"]],
        on="join_key",
        how="left",
        suffixes=("", "_odds"),
    )

    merged["season"] = pd.to_numeric(ensure_column(merged, "season", SEASON_CANDIDATES), errors="coerce").astype("Int64")
    merged["week"] = pd.to_numeric(ensure_column(merged, "week", WEEK_CANDIDATES), errors="coerce").astype("Int64")
    merged["model_prob"] = pd.to_numeric(ensure_column(merged, "model_prob", MODEL_PROB_CANDIDATES), errors="coerce")
    merged["cal_prob"] = pd.to_numeric(ensure_column(merged, "cal_prob", CAL_PROB_CANDIDATES), errors="coerce")
    merged["model_edge_vs_open"] = pd.to_numeric(ensure_column(merged, "model_edge_vs_open", EDGE_OPEN_CANDIDATES), errors="coerce")
    merged["model_edge_vs_close"] = pd.to_numeric(ensure_column(merged, "model_edge_vs_close", EDGE_CLOSE_CANDIDATES), errors="coerce")
    merged["result"] = normalize_result(ensure_column(merged, "result", RESULT_CANDIDATES))

    picked_odds = ensure_column(merged, "picked_odds", PAYOUT_ODDS_CANDIDATES)
    if picked_odds.isna().all():
        picked_odds = ensure_column(merged, "picked_odds", OPEN_AM_CANDIDATES)
    merged["picked_odds"] = pd.to_numeric(picked_odds, errors="coerce")

    merged = add_movement_fields(merged)
    merged = add_regime_labels(merged)
    merged["positive_close_edge"] = pd.to_numeric(merged["model_edge_vs_close"], errors="coerce").gt(0).astype(int)
    merged["roi_per_1"] = roi_from_result_and_odds(merged["result"], merged["picked_odds"])

    merged.to_csv(outdir / "market_regime_enriched.csv", index=False)

    season_summary = summarize(merged.dropna(subset=["result"]), ["season"])
    season_week_summary = summarize(merged.dropna(subset=["result"]), ["season", "week"])
    regime_summary = summarize(merged.dropna(subset=["result"]), ["season", "market_confirmation", "move_path", "move_size_bucket", "late_drift_flag"])
    close_edge_summary = summarize(merged.dropna(subset=["result"]), ["season", "close_edge_bucket"])

    season_summary.to_csv(outdir / "season_summary.csv", index=False)
    season_week_summary.to_csv(outdir / "season_week_summary.csv", index=False)
    regime_summary.to_csv(outdir / "regime_summary.csv", index=False)
    close_edge_summary.to_csv(outdir / "close_edge_summary.csv", index=False)

    season_2024 = merged[merged["season"] == 2024]
    season_2025 = merged[merged["season"] == 2025]
    drift_rows = []
    metrics = [
        "model_prob", "cal_prob", "model_edge_vs_open", "model_edge_vs_close",
        "open_to_mid_prob_move", "mid_to_close_prob_move", "open_to_close_prob_move", "roi_per_1", "result"
    ]
    for m in metrics:
        a = pd.to_numeric(season_2024[m], errors="coerce")
        b = pd.to_numeric(season_2025[m], errors="coerce")
        drift_rows.append({
            "metric": m,
            "season_2024_mean": a.mean(),
            "season_2025_mean": b.mean(),
            "delta_2025_minus_2024": b.mean() - a.mean(),
        })
    pd.DataFrame(drift_rows).to_csv(outdir / "season_drift_2024_vs_2025.csv", index=False)

    print(f"[OK] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
