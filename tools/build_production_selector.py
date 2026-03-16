from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def american_profit_per_1(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.where(s > 0, s / 100.0, np.where(s < 0, 100.0 / np.abs(s), np.nan))


def normalize_result(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        return pd.Series(np.where(out > 0, 1.0, np.where(out == 0, 0.0, np.nan)), index=s.index)
    t = s.astype(str).str.strip().str.lower()
    wins = {"1", "true", "w", "win", "won", "cash", "hit", "yes"}
    losses = {"0", "false", "l", "loss", "lost", "miss", "no"}
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[t.isin(wins)] = 1.0
    out[t.isin(losses)] = 0.0
    return out


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure(df: pd.DataFrame, candidates: list[str], default=np.nan) -> pd.Series:
    c = first_existing(df, candidates)
    if c is None:
        return pd.Series(default, index=df.index)
    return df[c]


def score_hybrid(df: pd.DataFrame) -> pd.Series:
    z = pd.DataFrame(index=df.index)
    for col in [
        "model_prob", "cal_prob", "model_edge_vs_open", "model_edge_vs_close",
        "open_to_mid_prob_move", "mid_to_close_prob_move"
    ]:
        s = pd.to_numeric(df[col], errors="coerce")
        z[col] = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) not in [0, np.nan] else 1.0)
        z[col] = z[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return (
        0.10 * z["model_prob"]
        + 0.25 * z["cal_prob"]
        + 0.15 * z["model_edge_vs_open"]
        + 0.30 * z["model_edge_vs_close"]
        + 0.10 * z["open_to_mid_prob_move"]
        + 0.10 * z["mid_to_close_prob_move"]
    )


def apply_top_n_per_week(df: pd.DataFrame, n: int, rank_col: str) -> pd.DataFrame:
    return (
        df.sort_values(["season", "week", rank_col], ascending=[True, True, False])
          .groupby(["season", "week"], dropna=False)
          .head(n)
          .copy()
    )


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "selector": label, "bets": 0, "hit_rate": np.nan, "roi": np.nan,
            "roi_2024": np.nan, "roi_2025": np.nan, "season_gap_abs": np.nan,
            "avg_close_edge": np.nan, "pct_positive_close_edge": np.nan,
        }
    season_roi = df.groupby("season")["roi_per_1"].mean()
    return {
        "selector": label,
        "bets": int(len(df)),
        "hit_rate": round(100 * df["result"].mean(), 2),
        "roi": round(100 * df["roi_per_1"].mean(), 2),
        "roi_2024": round(100 * season_roi.get(2024, np.nan), 2),
        "roi_2025": round(100 * season_roi.get(2025, np.nan), 2),
        "season_gap_abs": round(100 * abs(season_roi.get(2025, np.nan) - season_roi.get(2024, np.nan)), 2),
        "avg_close_edge": round(df["model_edge_vs_close"].mean(), 4),
        "pct_positive_close_edge": round(100 * df["model_edge_vs_close"].gt(0).mean(), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to market_regime_enriched.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    df["season"] = pd.to_numeric(ensure(df, ["season", "Season", "year"]), errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(ensure(df, ["week", "Week", "week_num", "week_number"]), errors="coerce").astype("Int64")
    df["result"] = normalize_result(ensure(df, ["result", "won", "win", "hit_bool", "outcome", "is_win"]))
    df["picked_odds"] = pd.to_numeric(ensure(df, ["picked_odds", "odds", "american_odds", "price", "bet_odds", "offered_odds", "open_odds"]), errors="coerce")
    df["roi_per_1"] = np.where(df["result"] == 1, american_profit_per_1(df["picked_odds"]), np.where(df["result"] == 0, -1.0, np.nan))

    needed = [
        "model_prob", "cal_prob", "model_edge_vs_open", "model_edge_vs_close",
        "open_to_mid_prob_move", "mid_to_close_prob_move"
    ]
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    work = df.dropna(subset=["result", "season", "week"] + needed + ["roi_per_1"]).copy()
    work["hybrid_score"] = score_hybrid(work)

    candidate_rows = []
    selected_exports: list[pd.DataFrame] = []

    for cal_min in [0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]:
        for edge_close_min in [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
            for edge_open_min in [0.00, 0.01, 0.03, 0.05]:
                for open_mid_min in [0.00, 0.005, 0.01, 0.015]:
                    for mid_close_min in [-0.005, 0.0, 0.005, 0.01]:
                        subset = work[
                            (work["cal_prob"] >= cal_min)
                            & (work["model_edge_vs_close"] >= edge_close_min)
                            & (work["model_edge_vs_open"] >= edge_open_min)
                            & (work["open_to_mid_prob_move"] >= open_mid_min)
                            & (work["mid_to_close_prob_move"] >= mid_close_min)
                        ].copy()
                        if 50 <= len(subset) <= 100:
                            label = f"rule_cal{cal_min:.2f}_ec{edge_close_min:.3f}_eo{edge_open_min:.3f}_om{open_mid_min:.3f}_mc{mid_close_min:.3f}"
                            row = summarize(subset, label)
                            row["family"] = "market_confirmed_rule"
                            row["cal_min"] = cal_min
                            row["edge_close_min"] = edge_close_min
                            row["edge_open_min"] = edge_open_min
                            row["open_mid_min"] = open_mid_min
                            row["mid_close_min"] = mid_close_min
                            candidate_rows.append(row)

    for n in [1, 2, 3, 4, 5, 6]:
        subset = work[work["cal_prob"] >= 0.60].copy()
        subset = apply_top_n_per_week(subset, n=n, rank_col="hybrid_score")
        if 50 <= len(subset) <= 100:
            row = summarize(subset, f"top_{n}_per_week_hybrid")
            row["family"] = "top_n_per_week"
            row["top_n"] = n
            candidate_rows.append(row)

    for edge_close_min in [0.03, 0.05, 0.08, 0.10, 0.12]:
        subset = work[(work["model_edge_vs_close"] >= edge_close_min) & (work["cal_prob"] >= 0.58)].copy()
        subset = subset.sort_values("model_edge_vs_close", ascending=False).head(100)
        if 50 <= len(subset) <= 100:
            row = summarize(subset, f"price_edge_close_{edge_close_min:.2f}")
            row["family"] = "price_based_ev"
            row["edge_close_min"] = edge_close_min
            candidate_rows.append(row)

    for thresh in [0.40, 0.60, 0.80, 1.00, 1.20]:
        subset = work[work["hybrid_score"] >= thresh].copy()
        if 50 <= len(subset) <= 100:
            row = summarize(subset, f"hybrid_score_{thresh:.2f}")
            row["family"] = "hybrid_weighted"
            row["hybrid_threshold"] = thresh
            candidate_rows.append(row)

    summary_df = pd.DataFrame(candidate_rows)
    if summary_df.empty:
        raise SystemExit("No selectors landed in the 50-100 bet target range. Widen thresholds.")

    summary_df["passes_basic"] = (
        summary_df["roi"].gt(0)
        & summary_df["roi_2024"].gt(0)
        & summary_df["roi_2025"].gt(0)
        & summary_df["season_gap_abs"].lt(15)
    )
    summary_df["stability_score"] = (
        summary_df["roi"].fillna(-999)
        - 0.50 * summary_df["season_gap_abs"].fillna(999)
        + 0.10 * summary_df["pct_positive_close_edge"].fillna(0)
    )

    summary_df = summary_df.sort_values(
        ["passes_basic", "stability_score", "roi", "bets"],
        ascending=[False, False, False, False],
    )
    summary_df.to_csv(outdir / "selector_summary.csv", index=False)

    best = summary_df.iloc[0].to_dict()
    best_family = best["family"]
    if best_family == "top_n_per_week":
        best_df = apply_top_n_per_week(work[work["cal_prob"] >= 0.60].copy(), int(best["top_n"]), "hybrid_score")
    elif best_family == "price_based_ev":
        best_df = work[(work["model_edge_vs_close"] >= float(best["edge_close_min"])) & (work["cal_prob"] >= 0.58)].copy()
        best_df = best_df.sort_values("model_edge_vs_close", ascending=False).head(100)
    elif best_family == "hybrid_weighted":
        best_df = work[work["hybrid_score"] >= float(best["hybrid_threshold"])].copy()
    else:
        best_df = work[
            (work["cal_prob"] >= float(best.get("cal_min", 0.0)))
            & (work["model_edge_vs_close"] >= float(best.get("edge_close_min", 0.0)))
            & (work["model_edge_vs_open"] >= float(best.get("edge_open_min", 0.0)))
            & (work["open_to_mid_prob_move"] >= float(best.get("open_mid_min", 0.0)))
            & (work["mid_to_close_prob_move"] >= float(best.get("mid_close_min", -999.0)))
        ].copy()

    best_df = best_df.sort_values(["season", "week", "hybrid_score"], ascending=[True, True, False])
    best_df.to_csv(outdir / "best_selector_bets.csv", index=False)

    by_season_week = best_df.groupby(["season", "week"], dropna=False).agg(
        bets=("result", "size"),
        hit_rate=("result", "mean"),
        roi=("roi_per_1", "mean"),
        avg_close_edge=("model_edge_vs_close", "mean"),
    ).reset_index()
    by_season_week["hit_rate"] = (100 * by_season_week["hit_rate"]).round(2)
    by_season_week["roi"] = (100 * by_season_week["roi"]).round(2)
    by_season_week.to_csv(outdir / "best_selector_by_week.csv", index=False)

    print(f"[OK] wrote selector outputs to {outdir}")
    print(f"[BEST] {best['selector']}")


if __name__ == "__main__":
    main()
