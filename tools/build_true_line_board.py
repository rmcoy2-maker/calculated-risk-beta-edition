from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def american_to_implied_prob(odds):
    if pd.isna(odds):
        return np.nan
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return abs(o) / (abs(o) + 100.0)
    return np.nan


def prob_to_american(p):
    if pd.isna(p):
        return np.nan
    try:
        p = float(p)
    except Exception:
        return np.nan
    if p <= 0 or p >= 1:
        return np.nan
    if p > 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def american_to_decimal(odds):
    if pd.isna(odds):
        return np.nan
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 1.0 + o / 100.0
    if o < 0:
        return 1.0 + 100.0 / abs(o)
    return np.nan


def first_present(df: pd.DataFrame, candidates: list[str], default=np.nan) -> pd.Series:
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return df[lookup[c.lower()]]
    return pd.Series([default] * len(df), index=df.index)


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clip_prob(series: pd.Series) -> pd.Series:
    return ensure_numeric(series).clip(lower=0.0001, upper=0.9999)


def assign_value_tier(edge_prob):
    if pd.isna(edge_prob):
        return "None"
    if edge_prob >= 0.050:
        return "Elite"
    if edge_prob >= 0.030:
        return "Strong"
    if edge_prob >= 0.015:
        return "Playable"
    if edge_prob >= 0.005:
        return "Thin"
    return "Negative"


def assign_governor(edge_prob, open_to_mid=None, mid_to_close=None):
    if pd.isna(edge_prob):
        return "None"

    adverse_late = pd.notna(mid_to_close) and mid_to_close < -0.01
    adverse_early = pd.notna(open_to_mid) and open_to_mid < -0.01

    if edge_prob >= 0.030 and not adverse_late:
        return "Tight"
    if edge_prob >= 0.020 and not adverse_late:
        return "Normal"
    if edge_prob >= 0.010 and not (adverse_late and adverse_early):
        return "Loose"
    return "Block"


def odds_bucket(odds):
    if pd.isna(odds):
        return "Unknown"
    o = float(odds)
    if o <= -300:
        return "Heavy Favorite"
    if o <= -150:
        return "Medium Favorite"
    if o < -110:
        return "Light Favorite"
    if o <= 110:
        return "Pickem"
    if o <= 180:
        return "Small Dog"
    return "Big Dog"


def get_week_governor_weight(week):
    if pd.isna(week):
        return 0.65
    try:
        w = int(float(week))
    except Exception:
        return 0.65

    if w <= 3:
        return 0.50
    if w <= 8:
        return 0.65
    return 0.75


def disagreement_bucket(disagreement_abs):
    if pd.isna(disagreement_abs):
        return "Unknown"
    if disagreement_abs < 0.03:
        return "Low"
    if disagreement_abs < 0.06:
        return "Moderate"
    if disagreement_abs < 0.10:
        return "High"
    return "Extreme"


def disagreement_penalty_from_abs(disagreement_abs):
    if pd.isna(disagreement_abs):
        return 0.00
    if disagreement_abs < 0.03:
        return 0.00
    if disagreement_abs < 0.06:
        return 0.10
    if disagreement_abs < 0.10:
        return 0.20
    return 0.30


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    g = df.groupby(group_cols, dropna=False, observed=False)
    summary = g.agg(
        bets=("profit", "size"),
        wins=("actual_win", "sum"),
        roi=("profit", "mean"),
        hit_rate=("actual_win", "mean"),
        avg_true_prob=("true_prob_cal", "mean"),
        avg_market_prob=("market_prob", "mean"),
        avg_edge_market=("edge_prob_market", "mean"),
        avg_edge_open=("model_edge_vs_open", "mean"),
        avg_edge_close=("model_edge_vs_close", "mean"),
        avg_edge_governed=("edge_prob_governed", "mean"),
        avg_otm_move=("open_to_mid_prob_move", "mean"),
        avg_mtc_move=("mid_to_close_prob_move", "mean"),
        avg_ev=("ev_per_1_cal", "mean"),
        avg_ev_governed=("ev_per_1_governed", "mean"),
        avg_hybrid_score=("hybrid_score", "mean"),
        avg_confidence_score=("confidence_score", "mean"),
        avg_edge_shrunk_35=("edge_prob_shrunk_35", "mean"),
        avg_edge_shrunk_40=("edge_prob_shrunk_40", "mean"),
        avg_edge_shrunk_50=("edge_prob_shrunk_50", "mean"),
        avg_disagreement_abs=("model_market_gap_abs", "mean"),
        avg_governed_weight=("governed_model_weight", "mean"),
    ).reset_index()

    return summary.sort_values("roi", ascending=False, na_position="last")


def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    recs = [
        ("rows", len(df)),
        ("rows_with_model_prob", int(df["model_prob"].notna().sum())),
        ("rows_with_cal_prob", int(df["cal_prob"].notna().sum())),
        ("rows_with_real_cal_prob", int((df["cal_prob_source"] == "cal_prob").sum())),
        ("rows_with_cal_fallback", int((df["cal_prob_source"] == "model_prob_fallback").sum())),
        ("rows_with_market_odds", int(df["market_odds"].notna().sum())),
        ("rows_with_profit", int(df["profit"].notna().sum())),
        ("rows_with_actual_win", int(df["actual_win"].notna().sum())),
        ("rows_with_edge_prob_governed", int(df["edge_prob_governed"].notna().sum())),
        ("rows_with_confidence_score", int(df["confidence_score"].notna().sum())),
    ]
    return pd.DataFrame(recs, columns=["metric", "value"])


def build_true_line_board(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Season"] = first_present(out, ["Season", "season"])
    out["Week"] = ensure_numeric(first_present(out, ["Week", "week"]))
    out["game_id"] = first_present(out, ["game_id", "Game_ID", "gameid"])
    out["selected_team"] = first_present(out, ["selected_team", "selection", "team_pick", "pick_team", "team"])
    out["profit"] = ensure_numeric(first_present(out, ["profit", "roi_per_1", "unit_profit", "profit_per_1"], default=np.nan))
    out["actual_win"] = ensure_numeric(first_present(out, ["actual_win", "result", "hit", "won"], default=np.nan))

    out["model_prob_raw"] = clip_prob(first_present(out, ["model_prob", "Model Prob", "prob", "p_win"]))

    cal_raw = first_present(out, ["cal_prob", "calibrated_prob", "prob_cal", "calibrated_probability"], default=np.nan)
    out["cal_prob"] = ensure_numeric(cal_raw)
    out["cal_prob"] = out["cal_prob"].where(out["cal_prob"].notna(), out["model_prob_raw"])
    out["cal_prob"] = clip_prob(out["cal_prob"])

    out["cal_prob_source"] = np.where(
        ensure_numeric(cal_raw).notna(),
        "cal_prob",
        "model_prob_fallback"
    )

    out["picked_odds"] = ensure_numeric(first_present(out, ["picked_odds", "american_odds", "odds", "price", "American"]))
    out["open_odds"] = ensure_numeric(first_present(out, ["open_odds", "open_american", "open_price", "odds_open"]))
    out["mid_odds"] = ensure_numeric(first_present(out, ["mid_odds", "mid_american", "mid_price", "odds_mid"]))
    out["close_odds"] = ensure_numeric(first_present(out, ["close_odds", "close_american", "close_price", "odds_close"]))

    out["market_odds"] = out["picked_odds"]
    out["market_odds"] = out["market_odds"].where(out["market_odds"].notna(), out["close_odds"])
    out["market_odds"] = out["market_odds"].where(out["market_odds"].notna(), out["open_odds"])

    out["market_prob"] = out["market_odds"].map(american_to_implied_prob)
    out["open_prob"] = out["open_odds"].map(american_to_implied_prob)
    out["mid_prob"] = out["mid_odds"].map(american_to_implied_prob)
    out["close_prob"] = out["close_odds"].map(american_to_implied_prob)

    # Overconfidence control:
    # shrink probabilities toward 0.5 before true-line and edge calculations.
    out["model_prob"] = clip_prob(0.5 + 0.85 * (out["model_prob_raw"] - 0.5))
    out["cal_prob_adjusted"] = clip_prob(0.5 + 0.85 * (out["cal_prob"] - 0.5))

    out["true_prob_raw"] = out["model_prob"]
    out["true_prob_cal"] = out["cal_prob_adjusted"]

    out["true_prob_shrunk_35"] = clip_prob(0.35 * out["true_prob_cal"] + 0.65 * out["market_prob"])
    out["true_prob_shrunk_40"] = clip_prob(0.40 * out["true_prob_cal"] + 0.60 * out["market_prob"])
    out["true_prob_shrunk_50"] = clip_prob(0.50 * out["true_prob_cal"] + 0.50 * out["market_prob"])

    out["true_line_raw"] = out["true_prob_raw"].map(prob_to_american)
    out["true_line_cal"] = out["true_prob_cal"].map(prob_to_american)
    out["true_line_shrunk_35"] = out["true_prob_shrunk_35"].map(prob_to_american)
    out["true_line_shrunk_40"] = out["true_prob_shrunk_40"].map(prob_to_american)
    out["true_line_shrunk_50"] = out["true_prob_shrunk_50"].map(prob_to_american)

    out["edge_prob_market"] = out["true_prob_cal"] - out["market_prob"]
    out["edge_prob_open"] = out["true_prob_cal"] - out["open_prob"]
    out["edge_prob_mid"] = out["true_prob_cal"] - out["mid_prob"]
    out["edge_prob_close"] = out["true_prob_cal"] - out["close_prob"]

    out["edge_prob_shrunk_35"] = out["true_prob_shrunk_35"] - out["market_prob"]
    out["edge_prob_shrunk_40"] = out["true_prob_shrunk_40"] - out["market_prob"]
    out["edge_prob_shrunk_50"] = out["true_prob_shrunk_50"] - out["market_prob"]

    upstream_open = ensure_numeric(first_present(out, ["model_edge_vs_open"], default=np.nan))
    upstream_close = ensure_numeric(first_present(out, ["model_edge_vs_close"], default=np.nan))
    out["model_edge_vs_open"] = upstream_open.where(upstream_open.notna(), out["edge_prob_open"])
    out["model_edge_vs_close"] = upstream_close.where(upstream_close.notna(), out["edge_prob_close"])

    upstream_otm = ensure_numeric(first_present(out, ["open_to_mid_prob_move"], default=np.nan))
    upstream_mtc = ensure_numeric(first_present(out, ["mid_to_close_prob_move"], default=np.nan))
    calc_otm = out["mid_prob"] - out["open_prob"]
    calc_mtc = out["close_prob"] - out["mid_prob"]
    out["open_to_mid_prob_move"] = upstream_otm.where(upstream_otm.notna(), calc_otm)
    out["mid_to_close_prob_move"] = upstream_mtc.where(upstream_mtc.notna(), calc_mtc)

    out["market_confirmation"] = np.where(
        (out["model_edge_vs_open"] > 0) & (out["open_to_mid_prob_move"] >= 0),
        "confirmed",
        np.where(
            (out["model_edge_vs_open"] > 0) & (out["mid_to_close_prob_move"] < 0),
            "faded_late",
            np.where(
                (out["model_edge_vs_open"] <= 0) & (out["open_to_mid_prob_move"] > 0),
                "market_only",
                "mixed"
            )
        )
    )

    out["is_market_favorite"] = out["market_prob"] > 0.5

    favorite_penalty = 0.04
    out["true_prob_adj_fav"] = out["true_prob_shrunk_40"]
    out.loc[out["is_market_favorite"], "true_prob_adj_fav"] = (
        out.loc[out["is_market_favorite"], "true_prob_shrunk_40"] - favorite_penalty
    ).clip(lower=0.0001, upper=0.9999)

    out["edge_prob_adj_fav"] = out["true_prob_adj_fav"] - out["market_prob"]
    out["true_line_adj_fav"] = out["true_prob_adj_fav"].map(prob_to_american)

    out["market_decimal"] = out["market_odds"].map(american_to_decimal)
    out["ev_per_1_cal"] = out["true_prob_cal"] * out["market_decimal"] - 1.0
    out["ev_per_1_shrunk_35"] = out["true_prob_shrunk_35"] * out["market_decimal"] - 1.0
    out["ev_per_1_shrunk_40"] = out["true_prob_shrunk_40"] * out["market_decimal"] - 1.0
    out["ev_per_1_shrunk_50"] = out["true_prob_shrunk_50"] * out["market_decimal"] - 1.0

    out["value_tier"] = out["edge_prob_market"].map(assign_value_tier)
    out["value_tier_shrunk_35"] = out["edge_prob_shrunk_35"].map(assign_value_tier)
    out["value_tier_shrunk_40"] = out["edge_prob_shrunk_40"].map(assign_value_tier)
    out["value_tier_shrunk_50"] = out["edge_prob_shrunk_50"].map(assign_value_tier)

    # Market disagreement signal
    out["model_market_gap"] = out["true_prob_cal"] - out["market_prob"]
    out["model_market_gap_abs"] = out["model_market_gap"].abs()
    out["disagreement_bucket"] = out["model_market_gap_abs"].map(disagreement_bucket)
    out["disagreement_penalty"] = out["model_market_gap_abs"].map(disagreement_penalty_from_abs)

    # Dynamic week-based governor + disagreement control
    out["week_governor_weight"] = out["Week"].map(get_week_governor_weight)
    out["governed_model_weight"] = (out["week_governor_weight"] - out["disagreement_penalty"]).clip(
        lower=0.35,
        upper=0.80,
    )

    out["true_prob_governed"] = clip_prob(
        out["governed_model_weight"] * out["true_prob_cal"]
        + (1.0 - out["governed_model_weight"]) * out["market_prob"]
    )
    out["true_line_governed"] = out["true_prob_governed"].map(prob_to_american)
    out["edge_prob_governed"] = out["true_prob_governed"] - out["market_prob"]
    out["ev_per_1_governed"] = out["true_prob_governed"] * out["market_decimal"] - 1.0
    out["value_tier_governed"] = out["edge_prob_governed"].map(assign_value_tier)

    # Confidence score:
    # reward governed edge, real disagreement, and market agreement in movement.
    out["market_movement_support"] = (
        0.60 * out["open_to_mid_prob_move"].fillna(0)
        + 0.40 * out["mid_to_close_prob_move"].fillna(0)
    )

    out["confidence_score"] = (
        0.55 * out["edge_prob_governed"].fillna(0)
        + 0.25 * out["model_market_gap_abs"].fillna(0)
        + 0.10 * out["model_edge_vs_close"].fillna(0)
        + 0.10 * out["market_movement_support"].fillna(0)
    )

    out["governor"] = [
        assign_governor(e, otm, mtc)
        for e, otm, mtc in zip(
            out["edge_prob_governed"],
            out["open_to_mid_prob_move"],
            out["mid_to_close_prob_move"],
        )
    ]

    out["odds_bucket"] = out["market_odds"].map(odds_bucket)

    out["hybrid_score"] = (
        0.15 * out["true_prob_governed"].fillna(0)
        + 0.15 * out["model_edge_vs_open"].fillna(0)
        + 0.15 * out["model_edge_vs_close"].fillna(0)
        + 0.20 * out["edge_prob_governed"].fillna(0)
        + 0.10 * out["open_to_mid_prob_move"].fillna(0)
        - 0.10 * (-out["mid_to_close_prob_move"].clip(upper=0).fillna(0))
        - 0.10 * out["model_market_gap_abs"].fillna(0)
        + 0.25 * out["confidence_score"].fillna(0)
    )

    out["close_edge_bucket"] = pd.cut(
        out["model_edge_vs_close"],
        bins=[-10, 0.0, 0.02, 0.05, 10],
        labels=["<0", "0-0.02", "0.02-0.05", ">0.05"],
        include_lowest=True,
    )

    dedupe_cols = [c for c in ["game_id", "selected_team"] if c in out.columns]
    if len(dedupe_cols) == 2:
        out = out.sort_values(
            ["Season", "Week", "confidence_score", "hybrid_score"],
            ascending=[True, True, False, False],
            na_position="last"
        ).drop_duplicates(dedupe_cols, keep="first")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False, encoding="utf-8-sig")
    board = build_true_line_board(df)

    board.to_csv(outdir / "true_line_board.csv", index=False)

    candidates = board[
        (board["governor"] != "Block") &
        (board["odds_bucket"].isin(["Medium Favorite", "Light Favorite", "Pickem", "Small Dog"])) &
        (~board["market_confirmation"].isin(["market_only"])) &
        (board["model_edge_vs_close"] > 0)
    ].copy()
    candidates.sort_values(
        ["Season", "Week", "confidence_score", "hybrid_score"],
        ascending=[True, True, False, False],
        inplace=True,
    )
    candidates.to_csv(outdir / "true_line_candidates.csv", index=False)

    summarize(board, ["Season"]).to_csv(outdir / "true_line_season_summary.csv", index=False)
    summarize(board, ["Season", "market_confirmation"]).to_csv(outdir / "true_line_confirmation_summary.csv", index=False)
    summarize(board, ["Season", "odds_bucket"]).to_csv(outdir / "true_line_odds_bucket_summary.csv", index=False)
    summarize(board, ["Season", "value_tier_shrunk_35"]).to_csv(outdir / "true_line_value_tier_shrunk_35_summary.csv", index=False)
    summarize(board, ["Season", "value_tier_shrunk_40"]).to_csv(outdir / "true_line_value_tier_shrunk_40_summary.csv", index=False)
    summarize(board, ["Season", "value_tier_shrunk_50"]).to_csv(outdir / "true_line_value_tier_shrunk_50_summary.csv", index=False)
    summarize(board, ["Season", "value_tier_governed"]).to_csv(outdir / "true_line_value_tier_governed_summary.csv", index=False)
    summarize(board, ["Season", "disagreement_bucket"]).to_csv(outdir / "true_line_disagreement_summary.csv", index=False)
    summarize(board, ["Season", "close_edge_bucket"]).to_csv(outdir / "true_line_close_edge_summary.csv", index=False)
    qc_summary(board).to_csv(outdir / "true_line_qc_summary.csv", index=False)

    print(f"Saved: {outdir / 'true_line_board.csv'}")
    print(f"Saved: {outdir / 'true_line_candidates.csv'}")
    print(f"Saved summaries to: {outdir}")


if __name__ == "__main__":
    main()