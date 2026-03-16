from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def max_drawdown_from_profit(profit: pd.Series) -> float:
    s = pd.to_numeric(profit, errors="coerce").fillna(0.0)
    equity = s.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()) if len(dd) else 0.0


def longest_losing_streak(actual_win: pd.Series) -> int:
    vals = pd.to_numeric(actual_win, errors="coerce").fillna(0)
    streak = 0
    best = 0
    for v in vals:
        if v >= 1:
            streak = 0
        else:
            streak += 1
            best = max(best, streak)
    return int(best)


def summarize_block(df: pd.DataFrame, label: str) -> dict:
    out = {"segment": label, "bets": len(df)}
    if df.empty:
        out.update({
            "hit_rate": np.nan,
            "roi": np.nan,
            "avg_edge_market": np.nan,
            "avg_edge_close": np.nan,
            "avg_ev": np.nan,
            "max_drawdown": np.nan,
            "longest_losing_streak": np.nan,
        })
        return out

    out.update({
        "hit_rate": pd.to_numeric(df.get("actual_win"), errors="coerce").mean(),
        "roi": pd.to_numeric(df.get("profit"), errors="coerce").mean(),
        "avg_edge_market": pd.to_numeric(df.get("edge_prob_market"), errors="coerce").mean(),
        "avg_edge_close": pd.to_numeric(df.get("model_edge_vs_close"), errors="coerce").mean(),
        "avg_ev": pd.to_numeric(df.get("ev_per_1_cal"), errors="coerce").mean(),
        "max_drawdown": max_drawdown_from_profit(df.get("profit")),
        "longest_losing_streak": longest_losing_streak(df.get("actual_win")),
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--oos-season", type=int, default=2025)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False, encoding="utf-8-sig")

    if "Season" in df.columns:
        season = pd.to_numeric(df["Season"], errors="coerce")
    else:
        season = pd.Series([np.nan] * len(df), index=df.index)

    report = pd.DataFrame([
        summarize_block(df, "all"),
        summarize_block(df[season == 2024], "2024"),
        summarize_block(df[season == 2025], "2025"),
        summarize_block(df[season == args.oos_season], f"oos_{args.oos_season}"),
    ])
    report.to_csv(outdir / "selector_diagnostic_report.csv", index=False)

    if "Week" in df.columns:
        weekly = (
            df.groupby(["Season", "Week"], dropna=False)
            .agg(
                bets=("profit", "size"),
                hit_rate=("actual_win", "mean"),
                roi=("profit", "mean"),
                avg_edge_market=("edge_prob_market", "mean"),
                avg_edge_close=("model_edge_vs_close", "mean"),
                avg_ev=("ev_per_1_cal", "mean"),
            )
            .reset_index()
        )
        weekly.to_csv(outdir / "selector_weekly_report.csv", index=False)

    if "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"], errors="coerce")
        if dates.notna().any():
            work = df.copy()
            work["month"] = dates.dt.to_period("M").astype(str)
            monthly = (
                work.groupby(["Season", "month"], dropna=False)
                .agg(
                    bets=("profit", "size"),
                    hit_rate=("actual_win", "mean"),
                    roi=("profit", "mean"),
                    avg_edge_market=("edge_prob_market", "mean"),
                    avg_edge_close=("model_edge_vs_close", "mean"),
                    avg_ev=("ev_per_1_cal", "mean"),
                )
                .reset_index()
            )
            monthly.to_csv(outdir / "selector_monthly_report.csv", index=False)

    print(f"Saved diagnostics to: {outdir}")


if __name__ == "__main__":
    main()