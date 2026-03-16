from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}. Found: {df.columns.tolist()}")
    return None


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


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_from_scores(scores_path: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_path, low_memory=False, encoding="utf-8-sig")

    season_col = find_col(df, ["Season", "season"])
    week_col = find_col(df, ["Week", "week"])
    home_team_col = find_col(df, ["HomeTeam", "home_team", "home"])
    away_team_col = find_col(df, ["AwayTeam", "away_team", "away"])
    home_score_col = find_col(df, ["HomeScore", "home_score", "home_points", "home_pts"])
    away_score_col = find_col(df, ["AwayScore", "away_score", "away_points", "away_pts"])

    spread_col = find_col(df, ["Spread", "spread", "closing_spread", "close_spread"], required=False)
    total_col = find_col(df, ["Total", "total", "closing_total", "close_total"], required=False)

    work = df.copy()
    work["Season"] = safe_numeric(work[season_col]).astype("Int64")
    work["Week"] = safe_numeric(work[week_col]).astype("Int64")
    work["home_score"] = safe_numeric(work[home_score_col])
    work["away_score"] = safe_numeric(work[away_score_col])
    work["combined_points"] = work["home_score"] + work["away_score"]
    work["margin_abs"] = (work["home_score"] - work["away_score"]).abs()

    if total_col is not None:
        work["closing_total"] = safe_numeric(work[total_col])
    else:
        work["closing_total"] = np.nan

    if spread_col is not None:
        work["closing_spread"] = safe_numeric(work[spread_col]).abs()
        raw_spread = safe_numeric(work[spread_col])

        # Negative spread means home favorite in common conventions.
        work["favorite_side"] = np.where(raw_spread < 0, "home", np.where(raw_spread > 0, "away", "pickem"))
        work["favorite_win"] = np.where(
            work["favorite_side"] == "home",
            (work["home_score"] > work["away_score"]).astype(float),
            np.where(
                work["favorite_side"] == "away",
                (work["away_score"] > work["home_score"]).astype(float),
                np.nan,
            ),
        )
        work["underdog_win"] = np.where(pd.notna(work["favorite_win"]), 1.0 - work["favorite_win"], np.nan)
    else:
        work["closing_spread"] = np.nan
        work["favorite_side"] = np.nan
        work["favorite_win"] = np.nan
        work["underdog_win"] = np.nan

    weekly = (
        work.groupby(["Season", "Week"], dropna=False)
        .agg(
            games=("combined_points", "size"),
            league_points_per_game=("combined_points", "mean"),
            league_home_points_per_game=("home_score", "mean"),
            league_away_points_per_game=("away_score", "mean"),
            league_margin_abs=("margin_abs", "mean"),
            league_closing_total_avg=("closing_total", "mean"),
            league_closing_spread_abs_avg=("closing_spread", "mean"),
            league_favorite_win_rate=("favorite_win", "mean"),
            league_underdog_win_rate=("underdog_win", "mean"),
        )
        .reset_index()
        .sort_values(["Season", "Week"])
    )

    # Prior-to-week rolling features within each season
    grp = weekly.groupby("Season", dropna=False)

    for col in [
        "league_points_per_game",
        "league_home_points_per_game",
        "league_away_points_per_game",
        "league_margin_abs",
        "league_closing_total_avg",
        "league_closing_spread_abs_avg",
        "league_favorite_win_rate",
        "league_underdog_win_rate",
    ]:
        weekly[f"{col}_prior"] = grp[col].transform(lambda s: s.shift(1).expanding().mean())
        weekly[f"{col}_last3"] = grp[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # Week 1 priors are empty by construction. Fill with expanding cross-season baseline up to previous season.
    season_baseline = (
        weekly.groupby("Season", dropna=False)[
            [
                "league_points_per_game",
                "league_home_points_per_game",
                "league_away_points_per_game",
                "league_margin_abs",
                "league_closing_total_avg",
                "league_closing_spread_abs_avg",
                "league_favorite_win_rate",
                "league_underdog_win_rate",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("Season")
    )

    baseline_cols = [c for c in season_baseline.columns if c != "Season"]
    for col in baseline_cols:
        season_baseline[f"{col}_prior_seasons"] = season_baseline[col].shift(1).expanding().mean()

    weekly = weekly.merge(
        season_baseline[["Season"] + [f"{c}_prior_seasons" for c in baseline_cols]],
        on="Season",
        how="left",
    )

    for col in baseline_cols:
        weekly[f"{col}_prior"] = weekly[f"{col}_prior"].where(
            weekly[f"{col}_prior"].notna(),
            weekly[f"{col}_prior_seasons"],
        )
        weekly[f"{col}_last3"] = weekly[f"{col}_last3"].where(
            weekly[f"{col}_last3"].notna(),
            weekly[f"{col}_prior_seasons"],
        )

    # Regime indicators
    weekly["scoring_regime_vs_prior"] = weekly["league_points_per_game_prior"]
    weekly["scoring_delta_vs_prior"] = weekly["league_points_per_game"] - weekly["league_points_per_game_prior"]
    weekly["favorite_reliability_delta_vs_prior"] = (
        weekly["league_favorite_win_rate"] - weekly["league_favorite_win_rate_prior"]
    )
    weekly["underdog_reliability_delta_vs_prior"] = (
        weekly["league_underdog_win_rate"] - weekly["league_underdog_win_rate_prior"]
    )
    weekly["volatility_proxy"] = weekly["league_margin_abs_prior"] - weekly["league_margin_abs"]
    weekly["scoring_up_flag"] = (weekly["scoring_delta_vs_prior"] > 1.0).astype(int)
    weekly["favorite_weaker_flag"] = (weekly["favorite_reliability_delta_vs_prior"] < -0.03).astype(int)
    weekly["high_variance_flag"] = (
        (weekly["favorite_weaker_flag"] == 1) | (weekly["scoring_up_flag"] == 1)
    ).astype(int)

    keep_cols = [
        "Season",
        "Week",
        "games",
        "league_points_per_game",
        "league_home_points_per_game",
        "league_away_points_per_game",
        "league_margin_abs",
        "league_closing_total_avg",
        "league_closing_spread_abs_avg",
        "league_favorite_win_rate",
        "league_underdog_win_rate",
        "league_points_per_game_prior",
        "league_points_per_game_last3",
        "league_home_points_per_game_prior",
        "league_home_points_per_game_last3",
        "league_away_points_per_game_prior",
        "league_away_points_per_game_last3",
        "league_margin_abs_prior",
        "league_margin_abs_last3",
        "league_closing_total_avg_prior",
        "league_closing_total_avg_last3",
        "league_closing_spread_abs_avg_prior",
        "league_closing_spread_abs_avg_last3",
        "league_favorite_win_rate_prior",
        "league_favorite_win_rate_last3",
        "league_underdog_win_rate_prior",
        "league_underdog_win_rate_last3",
        "scoring_delta_vs_prior",
        "favorite_reliability_delta_vs_prior",
        "underdog_reliability_delta_vs_prior",
        "volatility_proxy",
        "scoring_up_flag",
        "favorite_weaker_flag",
        "high_variance_flag",
    ]

    return weekly[keep_cols].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        default="2017-2025_scores.csv",
        help="Path to full league scores file",
    )
    parser.add_argument(
        "--out",
        default="analysis_out/weekly_league_regime_features.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weekly = build_from_scores(scores_path)
    weekly.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print("\nLast 20 rows:\n")
    print(weekly.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()