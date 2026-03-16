from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Change these two lines when you want to inspect another fold
FOLD_DIR = EXPORTS_DIR / "walkforward" / "train_to_2023_test_2024"
TEST_SEASON = 2024

MODEL_PROBS_PATH = FOLD_DIR / "model_probs_test_only.csv"
CLOSING_LINES_PATH = EXPORTS_DIR / "closing_lines.csv"
GAMES_MASTER_PATH = EXPORTS_DIR / "games_master.csv"

OUT_DIR = FOLD_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def american_to_implied_prob(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(
            odds > 0,
            100 / (odds + 100),
            np.where(odds < 0, (-odds) / ((-odds) + 100), np.nan),
        ),
        index=odds.index,
    )


def american_profit_per_dollar(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(
            odds > 0,
            odds / 100.0,
            np.where(odds < 0, 100.0 / (-odds), np.nan),
        ),
        index=odds.index,
    )


def build_consensus_closing_moneyline(closing: pd.DataFrame) -> pd.DataFrame:
    ml = closing.loc[closing["market"].astype(str).str.lower() == "moneyline"].copy()
    ml["side"] = ml["side"].astype(str).str.lower().str.strip()
    ml["odds"] = pd.to_numeric(ml["odds"], errors="coerce")

    ml = ml[
        ml["side"].isin(["home", "away"])
        & ml["odds"].notna()
        & (ml["odds"] != 0)
        & (ml["odds"].abs() >= 100)
        & (ml["odds"].abs() <= 5000)
    ].copy()

    if ml.empty:
        raise ValueError("No usable moneyline rows found in closing_lines.csv")

    consensus = (
        ml.groupby(["game_id", "side"], as_index=False)
        .agg(
            closing_odds=("odds", "median"),
            books=("book_key", "nunique"),
        )
    )

    home = (
        consensus.loc[consensus["side"] == "home", ["game_id", "closing_odds", "books"]]
        .rename(columns={"closing_odds": "closing_home_ml", "books": "home_books"})
        .copy()
    )

    away = (
        consensus.loc[consensus["side"] == "away", ["game_id", "closing_odds", "books"]]
        .rename(columns={"closing_odds": "closing_away_ml", "books": "away_books"})
        .copy()
    )

    return home.merge(away, on="game_id", how="outer")


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        out = pd.DataFrame(
            {
                "bets": [df["game_id"].count()],
                "wins": [df["actual_win"].sum()],
                "profit_units": [df["profit"].sum()],
                "avg_model_prob": [df["model_prob"].mean()],
                "avg_implied_prob": [df["implied_prob"].mean()],
                "avg_edge": [df["edge"].mean()],
                "avg_odds": [df["closing_odds"].mean()],
                "avg_books": [df["books_used"].mean()],
            }
        )
    else:
        out = (
            df.groupby(group_cols, observed=True, dropna=False)
            .agg(
                bets=("game_id", "count"),
                wins=("actual_win", "sum"),
                profit_units=("profit", "sum"),
                avg_model_prob=("model_prob", "mean"),
                avg_implied_prob=("implied_prob", "mean"),
                avg_edge=("edge", "mean"),
                avg_odds=("closing_odds", "mean"),
                avg_books=("books_used", "mean"),
            )
            .reset_index()
        )

    out["win_rate"] = out["wins"] / out["bets"]
    out["roi"] = out["profit_units"] / out["bets"]
    out["calibration_gap"] = out["win_rate"] - out["avg_model_prob"]
    return out


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["odds_bucket"] = pd.cut(
        out["closing_odds"],
        bins=[-np.inf, -300, -200, -150, -110, 110, 150, 200, 300, np.inf],
        labels=[
            "<=-300",
            "-300 to -200",
            "-200 to -150",
            "-150 to -110",
            "-110 to +110",
            "+110 to +150",
            "+150 to +200",
            "+200 to +300",
            ">=+300",
        ],
        right=False,
    )

    out["model_prob_bucket"] = pd.cut(
        out["model_prob"],
        bins=[0.00, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.00],
        right=False,
        include_lowest=True,
    )

    out["edge_bucket"] = pd.cut(
        out["edge"],
        bins=[-1.0, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 1.0],
        right=False,
        include_lowest=True,
    )

    out["favorite_or_dog"] = np.where(out["closing_odds"] < 0, "favorite", "underdog")

    return out


def main() -> None:
    print(f"Loading fold model probabilities: {MODEL_PROBS_PATH}")
    mp = pd.read_csv(MODEL_PROBS_PATH, low_memory=False)

    print(f"Loading closing lines: {CLOSING_LINES_PATH}")
    closing = pd.read_csv(CLOSING_LINES_PATH, low_memory=False)

    print(f"Loading game results: {GAMES_MASTER_PATH}")
    gm = pd.read_csv(GAMES_MASTER_PATH, low_memory=False)

    needed_mp = [
        "game_id",
        "Season",
        "Week",
        "game_date",
        "away_team",
        "home_team",
        "p_home_win",
        "p_away_win",
    ]
    missing_mp = [c for c in needed_mp if c not in mp.columns]
    if missing_mp:
        raise ValueError(f"model_probs_test_only.csv missing columns: {missing_mp}")

    needed_gm = ["game_id", "home_win", "away_win"]
    missing_gm = [c for c in needed_gm if c not in gm.columns]
    if missing_gm:
        raise ValueError(f"games_master.csv missing columns: {missing_gm}")

    mp["Season"] = pd.to_numeric(mp["Season"], errors="coerce").astype("Int64")
    mp = mp[mp["Season"] == TEST_SEASON].copy()

    if mp.empty:
        raise ValueError(f"No rows found in model_probs for TEST_SEASON={TEST_SEASON}")

    cl = build_consensus_closing_moneyline(closing)
    df = mp[needed_mp].merge(cl, on="game_id", how="inner")
    df = df.merge(gm[needed_gm], on="game_id", how="left")

    home = df.copy()
    home["side"] = "home"
    home["closing_odds"] = pd.to_numeric(home["closing_home_ml"], errors="coerce")
    home["model_prob"] = pd.to_numeric(home["p_home_win"], errors="coerce")
    home["actual_win"] = pd.to_numeric(home["home_win"], errors="coerce")
    home["books_used"] = pd.to_numeric(home["home_books"], errors="coerce")

    away = df.copy()
    away["side"] = "away"
    away["closing_odds"] = pd.to_numeric(away["closing_away_ml"], errors="coerce")
    away["model_prob"] = pd.to_numeric(away["p_away_win"], errors="coerce")
    away["actual_win"] = pd.to_numeric(away["away_win"], errors="coerce")
    away["books_used"] = pd.to_numeric(away["away_books"], errors="coerce")

    bets = pd.concat([home, away], ignore_index=True)

    bets = bets[
        bets["closing_odds"].notna()
        & (bets["closing_odds"].abs() >= 100)
        & (bets["closing_odds"].abs() <= 5000)
        & bets["model_prob"].between(0.001, 0.999)
        & bets["actual_win"].isin([0, 1])
    ].copy()

    bets["implied_prob"] = american_to_implied_prob(bets["closing_odds"])
    bets["edge"] = bets["model_prob"] - bets["implied_prob"]
    bets["profit_per_dollar_win"] = american_profit_per_dollar(bets["closing_odds"])
    bets["profit"] = np.where(bets["actual_win"] == 1, bets["profit_per_dollar_win"], -1.0)

    bets = add_buckets(bets)

    overall = summarize(bets, [])
    by_side = summarize(bets, ["side"])
    by_favdog = summarize(bets, ["favorite_or_dog"])
    by_odds = summarize(bets, ["odds_bucket"])
    by_model_prob = summarize(bets, ["model_prob_bucket"])
    by_edge = summarize(bets, ["edge_bucket"])
    by_side_odds = summarize(bets, ["side", "odds_bucket"])

    top_edges = bets.sort_values(["edge", "game_date"], ascending=[False, True], kind="stable").reset_index(drop=True)
    bottom_edges = bets.sort_values(["edge", "game_date"], ascending=[True, True], kind="stable").reset_index(drop=True)

    overall.to_csv(OUT_DIR / "overall.csv", index=False)
    by_side.to_csv(OUT_DIR / "by_side.csv", index=False)
    by_favdog.to_csv(OUT_DIR / "by_favorite_or_dog.csv", index=False)
    by_odds.to_csv(OUT_DIR / "by_odds_bucket.csv", index=False)
    by_model_prob.to_csv(OUT_DIR / "by_model_prob_bucket.csv", index=False)
    by_edge.to_csv(OUT_DIR / "by_edge_bucket.csv", index=False)
    by_side_odds.to_csv(OUT_DIR / "by_side_and_odds_bucket.csv", index=False)
    top_edges.head(50).to_csv(OUT_DIR / "top_50_edges.csv", index=False)
    bottom_edges.head(50).to_csv(OUT_DIR / "bottom_50_edges.csv", index=False)

    print("\nOVERALL")
    print(overall.to_string(index=False))

    print("\nBY FAVORITE / DOG")
    print(by_favdog.to_string(index=False))

    print("\nBY SIDE")
    print(by_side.to_string(index=False))

    print("\nBY ODDS BUCKET")
    print(by_odds.to_string(index=False))

    print("\nBY MODEL PROB BUCKET")
    print(by_model_prob.to_string(index=False))

    print("\nBY EDGE BUCKET")
    print(by_edge.to_string(index=False))

    print("\nTOP 20 EDGES")
    print(
        top_edges[
            [
                "game_id", "side", "closing_odds", "model_prob", "implied_prob",
                "edge", "actual_win", "profit"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nBOTTOM 20 EDGES")
    print(
        bottom_edges[
            [
                "game_id", "side", "closing_odds", "model_prob", "implied_prob",
                "edge", "actual_win", "profit"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(f"\nSaved analysis files to: {OUT_DIR}")


if __name__ == "__main__":
    main()
