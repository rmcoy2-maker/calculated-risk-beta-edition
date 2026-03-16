from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "rule_name": label,
            "bets": 0,
            "wins": 0,
            "hit_rate": np.nan,
            "roi": np.nan,
            "avg_edge_shrunk_40": np.nan,
            "avg_edge_close": np.nan,
            "avg_hybrid_score": np.nan,
            "season_2024_roi": np.nan,
            "season_2025_roi": np.nan,
            "season_gap_abs": np.nan,
        }

    s2024 = df[pd.to_numeric(df["Season"], errors="coerce") == 2024]
    s2025 = df[pd.to_numeric(df["Season"], errors="coerce") == 2025]

    roi2024 = pd.to_numeric(s2024["profit"], errors="coerce").mean() if not s2024.empty else np.nan
    roi2025 = pd.to_numeric(s2025["profit"], errors="coerce").mean() if not s2025.empty else np.nan

    return {
        "rule_name": label,
        "bets": len(df),
        "wins": int(pd.to_numeric(df["actual_win"], errors="coerce").fillna(0).sum()),
        "hit_rate": pd.to_numeric(df["actual_win"], errors="coerce").mean(),
        "roi": pd.to_numeric(df["profit"], errors="coerce").mean(),
        "avg_edge_shrunk_40": pd.to_numeric(df["edge_prob_shrunk_40"], errors="coerce").mean(),
        "avg_edge_close": pd.to_numeric(df["model_edge_vs_close"], errors="coerce").mean(),
        "avg_hybrid_score": pd.to_numeric(df["hybrid_score"], errors="coerce").mean(),
        "season_2024_roi": roi2024,
        "season_2025_roi": roi2025,
        "season_gap_abs": abs(roi2024 - roi2025) if pd.notna(roi2024) and pd.notna(roi2025) else np.nan,
    }


def top_n_per_week(df: pd.DataFrame, n: int) -> pd.DataFrame:
    work = df.copy()
    work["Season"] = pd.to_numeric(work["Season"], errors="coerce")
    work["Week"] = pd.to_numeric(work["Week"], errors="coerce")
    work = work.sort_values(["Season", "Week", "hybrid_score"], ascending=[True, True, False])
    return work.groupby(["Season", "Week"], dropna=False).head(n).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False, encoding="utf-8-sig")

    base = df.copy()
    base = base[
        base["odds_bucket"].isin(["Medium Favorite", "Light Favorite"]) &
        base["market_confirmation"].isin(["confirmed", "faded_late"]) &
        (pd.to_numeric(base["model_edge_vs_close"], errors="coerce") > 0) &
        (~base["odds_bucket"].isin(["Big Dog", "Unknown"])) &
        (base["governor"] != "Block")
    ].copy()

    candidates = []

    for edge_min in [0.005, 0.010, 0.015, 0.020]:
        for weekly_n in [1, 2, 3, 4]:
            rule = base[pd.to_numeric(base["edge_prob_shrunk_40"], errors="coerce") >= edge_min].copy()
            rule = top_n_per_week(rule, weekly_n)

            if 40 <= len(rule) <= 120:
                label = f"shr40_edge>={edge_min:.3f}_top{weekly_n}"
                candidates.append((label, rule))

    for edge_min in [0.005, 0.010, 0.015]:
        rule = base[pd.to_numeric(base["edge_prob_shrunk_35"], errors="coerce") >= edge_min].copy()
        if 40 <= len(rule) <= 120:
            label = f"shr35_edge>={edge_min:.3f}"
            candidates.append((label, rule))

    for edge_min in [0.005, 0.010, 0.015]:
        rule = base[pd.to_numeric(base["edge_prob_shrunk_50"], errors="coerce") >= edge_min].copy()
        if 40 <= len(rule) <= 120:
            label = f"shr50_edge>={edge_min:.3f}"
            candidates.append((label, rule))

    summary_rows = [summarize(rule_df, label) for label, rule_df in candidates]
    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        summary.to_csv(outdir / "shrunk_selector_summary.csv", index=False)
        print("No shrunk selectors landed in the 40-120 bet range. Widen thresholds.")
        return

    summary["score"] = (
        summary["roi"].fillna(-999)
        - 0.50 * summary["season_gap_abs"].fillna(999)
    )
    summary = summary.sort_values(["score", "roi", "bets"], ascending=[False, False, False])
    summary.to_csv(outdir / "shrunk_selector_summary.csv", index=False)

    best_name = summary.iloc[0]["rule_name"]
    best_df = next(df_ for name, df_ in candidates if name == best_name)

    best_df.to_csv(outdir / "best_shrunk_selector_bets.csv", index=False)

    by_week = (
        best_df.groupby(["Season", "Week"], dropna=False)
        .agg(
            bets=("profit", "size"),
            wins=("actual_win", "sum"),
            roi=("profit", "mean"),
            avg_edge_shrunk_40=("edge_prob_shrunk_40", "mean"),
            avg_edge_close=("model_edge_vs_close", "mean"),
            avg_hybrid_score=("hybrid_score", "mean"),
        )
        .reset_index()
    )
    by_week.to_csv(outdir / "best_shrunk_selector_by_week.csv", index=False)

    print(f"Saved: {outdir / 'shrunk_selector_summary.csv'}")
    print(f"Saved: {outdir / 'best_shrunk_selector_bets.csv'}")
    print(f"Saved: {outdir / 'best_shrunk_selector_by_week.csv'}")
    print(f"Best rule: {best_name}")


if __name__ == "__main__":
    main()