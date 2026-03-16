import numpy as np
import pandas as pd

INPUT_PATH = "exports/fort_knox_market_joined_moneyline_scored_all_seasons.csv"

OUTPUT_BETS = "analysis_out/uncertainty_selector_bets_banded.csv"
OUTPUT_SUMMARY = "analysis_out/uncertainty_selector_summary_banded.csv"
OUTPUT_WEEKLY_2025 = "analysis_out/uncertainty_selector_weekly_2025_banded.csv"
OUTPUT_WEEKLY_ALL = "analysis_out/uncertainty_selector_weekly_all_banded.csv"
OUTPUT_MARKET_BUCKETS = "analysis_out/uncertainty_selector_market_prob_buckets_banded.csv"


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ensure_numeric(
        df,
        [
            "Season",
            "Week",
            "market_prob",
            "close_prob",
            "edge_prob_governed",
            "model_edge_vs_close",
            "hybrid_score",
            "confidence_score",
            "profit",
            "actual_win",
            "open_to_mid_prob_move",
            "mid_to_close_prob_move",
            "model_prob",
            "implied_prob",
            "edge",
            "closing_odds",
        ],
    )

    print("\nInitial rows:", len(df))

    # ---------- fallback construction for lean scored files ----------
    if "market_prob" not in df.columns:
        if "implied_prob" in df.columns:
            df["market_prob"] = pd.to_numeric(df["implied_prob"], errors="coerce")
        else:
            df["market_prob"] = np.nan

    if "close_prob" not in df.columns:
        if "implied_prob" in df.columns:
            df["close_prob"] = pd.to_numeric(df["implied_prob"], errors="coerce")
        else:
            df["close_prob"] = df["market_prob"]

    if "model_edge_vs_close" not in df.columns:
        if "edge" in df.columns:
            df["model_edge_vs_close"] = pd.to_numeric(df["edge"], errors="coerce")
        elif "model_prob" in df.columns and "close_prob" in df.columns:
            df["model_edge_vs_close"] = pd.to_numeric(df["model_prob"], errors="coerce") - pd.to_numeric(df["close_prob"], errors="coerce")
        else:
            df["model_edge_vs_close"] = np.nan

    if "edge_prob_governed" not in df.columns:
        if "edge" in df.columns:
            df["edge_prob_governed"] = pd.to_numeric(df["edge"], errors="coerce")
        else:
            df["edge_prob_governed"] = df["model_edge_vs_close"]

    if "hybrid_score" not in df.columns:
        # Simple fallback hybrid score for scored-bet files
        df["hybrid_score"] = (
            0.60 * df["edge_prob_governed"].fillna(0)
            + 0.40 * df["model_edge_vs_close"].fillna(0)
        )

    if "confidence_score" not in df.columns:
        df["confidence_score"] = (
            0.70 * df["edge_prob_governed"].fillna(0)
            + 0.30 * df["model_edge_vs_close"].fillna(0)
        )

    if "open_to_mid_prob_move" not in df.columns:
        df["open_to_mid_prob_move"] = 0.0

    if "mid_to_close_prob_move" not in df.columns:
        df["mid_to_close_prob_move"] = 0.0

    # If no game_id exists, create a fallback
    if "game_id" not in df.columns:
        date_part = df["game_date"].astype(str) if "game_date" in df.columns else "NA"
        team_part = df["selected_team"].astype(str) if "selected_team" in df.columns else df.index.astype(str)
        df["game_id"] = df["Season"].astype(str) + "_" + df["Week"].astype(str) + "_" + date_part + "_" + team_part

    required_cols = [
        "market_prob",
        "close_prob",
        "edge_prob_governed",
        "model_edge_vs_close",
        "hybrid_score",
        "profit",
        "actual_win",
        "Week",
        "Season",
    ]
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required).copy()
    print("After NA filter:", len(df))

    if "uncertainty_bucket" in df.columns:
        df = df[df["uncertainty_bucket"].isin(["Medium", "High"])].copy()
        print("After uncertainty filter:", len(df))
    else:
        print("After uncertainty filter: skipped (uncertainty_bucket not found)")

    if "market_confirmation" in df.columns:
        df = df[df["market_confirmation"].isin(["confirmed", "faded_late"])].copy()
        print("After confirmation filter:", len(df))
    else:
        print("After confirmation filter: skipped (market_confirmation not found)")

    df["close_minus_governed_gap"] = df["model_edge_vs_close"] - df["edge_prob_governed"]
    df["close_to_governed_ratio"] = np.where(
        df["model_edge_vs_close"] > 0,
        df["edge_prob_governed"] / df["model_edge_vs_close"],
        np.nan,
    )

    # Diagnostic-only metric; model-relative, not external CLV
    df["clv"] = df["model_edge_vs_close"]
    df["beat_close"] = np.where(df["clv"] > 0, 1, 0)

    df["market_prob_bucket"] = pd.cut(
        df["market_prob"],
        bins=[0.0, 0.40, 0.50, 0.60, 0.72, 1.0],
        labels=["<=0.40", "0.40-0.50", "0.50-0.60", "0.60-0.72", ">0.72"],
        include_lowest=True,
    )

    print("After base filters:", len(df))
    return df


def apply_late_season_rule(
    df: pd.DataFrame,
    governed_min_late: float,
    close_min_late: float,
    market_cap_late: float,
    ratio_min_late: float,
    start_week: int = 11,
) -> pd.DataFrame:
    return df[
        (df["Week"] < start_week) |
        (
            (df["Week"] >= start_week)
            & (df["edge_prob_governed"] >= governed_min_late)
            & (df["model_edge_vs_close"] >= close_min_late)
            & (df["market_prob"] <= market_cap_late)
            & (df["edge_prob_governed"] >= ratio_min_late * df["model_edge_vs_close"])
        )
    ].copy()


def sort_for_selection(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = []
    ascending = []

    if "confidence_score" in df.columns:
        sort_cols.append("confidence_score")
        ascending.append(False)

    sort_cols.append("hybrid_score")
    ascending.append(False)

    if "edge_prob_governed" in df.columns:
        sort_cols.append("edge_prob_governed")
        ascending.append(False)

    return df.sort_values(sort_cols, ascending=ascending, kind="stable")


def finalize_band(
    df: pd.DataFrame,
    band_name: str,
    edge_min: float,
    edge_max: float,
    close_edge_cap: float,
    ratio_min: float,
    governed_min_late: float,
    close_min_late: float,
    market_cap_late: float,
    ratio_min_late: float,
    top_per_week: int,
    market_prob_min: float | None = None,
    market_prob_max: float | None = None,
    late_start_week: int = 11,
) -> pd.DataFrame:
    work = df.copy()

    work = work[
        (work["edge_prob_governed"] >= edge_min)
        & (work["edge_prob_governed"] <= edge_max)
    ].copy()
    print(f"{band_name} after edge filter:", len(work))

    work = work[
        (work["model_edge_vs_close"] <= close_edge_cap)
        & (work["edge_prob_governed"] >= ratio_min * work["model_edge_vs_close"])
    ].copy()
    print(f"{band_name} after consistency filter:", len(work))

    if market_prob_min is not None:
        work = work[work["market_prob"] >= market_prob_min].copy()
    if market_prob_max is not None:
        work = work[work["market_prob"] <= market_prob_max].copy()
    if market_prob_min is not None or market_prob_max is not None:
        print(f"{band_name} after market-prob band:", len(work))

    work = apply_late_season_rule(
        work,
        governed_min_late=governed_min_late,
        close_min_late=close_min_late,
        market_cap_late=market_cap_late,
        ratio_min_late=ratio_min_late,
        start_week=late_start_week,
    )
    print(f"{band_name} after late-season rule:", len(work))

    work = (
        sort_for_selection(work)
        .groupby("game_id", dropna=False)
        .head(1)
        .copy()
    )
    print(f"{band_name} after one-side-per-game:", len(work))

    work = (
        sort_for_selection(work)
        .sort_values(["Season", "Week"], kind="stable")
        .groupby(["Season", "Week"], dropna=False)
        .head(top_per_week)
        .copy()
    )
    print(f"{band_name} after top-{top_per_week}-per-week:", len(work))

    work["bet_band"] = band_name
    return work


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bet_band",
                "Season",
                "bets",
                "wins",
                "hit_rate",
                "roi",
                "avg_market_prob",
                "avg_edge_governed",
                "avg_close_edge",
                "avg_close_minus_governed_gap",
                "avg_close_to_governed_ratio",
                "avg_confidence_score",
                "avg_clv",
                "beat_close_rate",
            ]
        )

    agg_map = {
        "bets": ("profit", "size"),
        "wins": ("actual_win", "sum"),
        "hit_rate": ("actual_win", "mean"),
        "roi": ("profit", "mean"),
        "avg_market_prob": ("market_prob", "mean"),
        "avg_edge_governed": ("edge_prob_governed", "mean"),
        "avg_close_edge": ("model_edge_vs_close", "mean"),
        "avg_close_minus_governed_gap": ("close_minus_governed_gap", "mean"),
        "avg_close_to_governed_ratio": ("close_to_governed_ratio", "mean"),
        "avg_clv": ("clv", "mean"),
        "beat_close_rate": ("beat_close", "mean"),
    }

    if "confidence_score" in df.columns:
        agg_map["avg_confidence_score"] = ("confidence_score", "mean")

    summary = (
        df.groupby(["bet_band", "Season"], dropna=False)
        .agg(**agg_map)
        .reset_index()
        .sort_values(["bet_band", "Season"])
    )
    return summary


def build_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bet_band",
                "Season",
                "Week",
                "bets",
                "wins",
                "hit_rate",
                "roi",
                "avg_market_prob",
                "avg_edge_governed",
                "avg_close_edge",
                "avg_confidence_score",
                "avg_clv",
                "beat_close_rate",
            ]
        )

    agg_map = {
        "bets": ("profit", "size"),
        "wins": ("actual_win", "sum"),
        "hit_rate": ("actual_win", "mean"),
        "roi": ("profit", "mean"),
        "avg_market_prob": ("market_prob", "mean"),
        "avg_edge_governed": ("edge_prob_governed", "mean"),
        "avg_close_edge": ("model_edge_vs_close", "mean"),
        "avg_clv": ("clv", "mean"),
        "beat_close_rate": ("beat_close", "mean"),
    }

    if "confidence_score" in df.columns:
        agg_map["avg_confidence_score"] = ("confidence_score", "mean")

    weekly = (
        df.groupby(["bet_band", "Season", "Week"], dropna=False)
        .agg(**agg_map)
        .reset_index()
        .sort_values(["bet_band", "Season", "Week"])
    )
    return weekly


def build_weekly_2025(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bet_band",
                "Week",
                "bets",
                "wins",
                "hit_rate",
                "roi",
                "avg_market_prob",
                "avg_edge_governed",
                "avg_close_edge",
                "avg_confidence_score",
                "avg_clv",
                "beat_close_rate",
            ]
        )

    agg_map = {
        "bets": ("profit", "size"),
        "wins": ("actual_win", "sum"),
        "hit_rate": ("actual_win", "mean"),
        "roi": ("profit", "mean"),
        "avg_market_prob": ("market_prob", "mean"),
        "avg_edge_governed": ("edge_prob_governed", "mean"),
        "avg_close_edge": ("model_edge_vs_close", "mean"),
        "avg_clv": ("clv", "mean"),
        "beat_close_rate": ("beat_close", "mean"),
    }

    if "confidence_score" in df.columns:
        agg_map["avg_confidence_score"] = ("confidence_score", "mean")

    weekly = (
        df[df["Season"] == 2025]
        .groupby(["bet_band", "Week"], dropna=False)
        .agg(**agg_map)
        .reset_index()
        .sort_values(["bet_band", "Week"])
    )
    return weekly


def build_market_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bet_band",
                "Season",
                "market_prob_bucket",
                "bets",
                "wins",
                "hit_rate",
                "roi",
                "avg_market_prob",
                "avg_edge_governed",
                "avg_close_edge",
                "avg_close_to_governed_ratio",
                "avg_confidence_score",
                "avg_clv",
                "beat_close_rate",
            ]
        )

    agg_map = {
        "bets": ("profit", "size"),
        "wins": ("actual_win", "sum"),
        "hit_rate": ("actual_win", "mean"),
        "roi": ("profit", "mean"),
        "avg_market_prob": ("market_prob", "mean"),
        "avg_edge_governed": ("edge_prob_governed", "mean"),
        "avg_close_edge": ("model_edge_vs_close", "mean"),
        "avg_close_to_governed_ratio": ("close_to_governed_ratio", "mean"),
        "avg_clv": ("clv", "mean"),
        "beat_close_rate": ("beat_close", "mean"),
    }

    if "confidence_score" in df.columns:
        agg_map["avg_confidence_score"] = ("confidence_score", "mean")

    out = (
        df.groupby(
            ["bet_band", "Season", "market_prob_bucket"],
            dropna=False,
            observed=False,
        )
        .agg(**agg_map)
        .reset_index()
        .sort_values(["bet_band", "Season", "market_prob_bucket"])
    )
    return out


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    base = prepare_base_df(df)

    print("\n--- HIGH CONFIDENCE BAND ---")
    high_conf = finalize_band(
        df=base,
        band_name="High Confidence",
        edge_min=0.018,
        edge_max=0.060,
        close_edge_cap=0.10,
        ratio_min=0.30,
        governed_min_late=0.025,
        close_min_late=0.030,
        market_cap_late=0.55,
        ratio_min_late=0.35,
        top_per_week=2,
        market_prob_min=0.40,
        market_prob_max=0.55,
        late_start_week=11,
    )

    print("\n--- BALANCED BAND ---")
    balanced = finalize_band(
        df=base,
        band_name="Balanced",
        edge_min=0.010,
        edge_max=0.060,
        close_edge_cap=0.12,
        ratio_min=0.22,
        governed_min_late=0.018,
        close_min_late=0.025,
        market_cap_late=0.58,
        ratio_min_late=0.28,
        top_per_week=3,
        market_prob_min=None,
        market_prob_max=0.60,
        late_start_week=11,
    )

    all_bets = pd.concat([high_conf, balanced], ignore_index=True)

    if not all_bets.empty:
        sort_cols = ["bet_band", "Season", "Week"]
        ascending = [True, True, True]

        if "confidence_score" in all_bets.columns:
            sort_cols.append("confidence_score")
            ascending.append(False)

        sort_cols.append("hybrid_score")
        ascending.append(False)

        all_bets = all_bets.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    summary = build_summary(all_bets)
    weekly_2025 = build_weekly_2025(all_bets)
    weekly_all = build_weekly(all_bets)
    market_buckets = build_market_bucket_summary(all_bets)

    print("\nBANDED SELECTOR SUMMARY\n")
    if summary.empty:
        print("No bets qualified.")
    else:
        print(summary.to_string(index=False))

    print("\n2025 WEEKLY BY BAND\n")
    if weekly_2025.empty:
        print("No 2025 bets qualified.")
    else:
        print(weekly_2025.to_string(index=False))

    print("\nMARKET PROB BUCKET SUMMARY\n")
    if market_buckets.empty:
        print("No market bucket summary available.")
    else:
        print(market_buckets.to_string(index=False))

    all_bets.to_csv(OUTPUT_BETS, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    weekly_2025.to_csv(OUTPUT_WEEKLY_2025, index=False)
    weekly_all.to_csv(OUTPUT_WEEKLY_ALL, index=False)
    market_buckets.to_csv(OUTPUT_MARKET_BUCKETS, index=False)

    print("\nSaved:")
    print(OUTPUT_BETS)
    print(OUTPUT_SUMMARY)
    print(OUTPUT_WEEKLY_2025)
    print(OUTPUT_WEEKLY_ALL)
    print(OUTPUT_MARKET_BUCKETS)


if __name__ == "__main__":
    main()