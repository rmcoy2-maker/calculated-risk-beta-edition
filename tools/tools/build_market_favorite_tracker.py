from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "exports"

CLOSING_LINES_PATH = EXPORTS_DIR / "closing_lines.csv"
MARKET_PRIORS_PATH = EXPORTS_DIR / "market_priors.csv"
OUT_PATH = EXPORTS_DIR / "market_favorite_tracker.csv"


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


def side_from_odds(home_odds: float, away_odds: float) -> str:
    if pd.isna(home_odds) or pd.isna(away_odds):
        return np.nan
    if home_odds < away_odds:
        return "home"
    if away_odds < home_odds:
        return "away"
    return "pick"


def favorite_strength(home_odds: float, away_odds: float) -> float:
    if pd.isna(home_odds) or pd.isna(away_odds):
        return np.nan

    home_prob = american_to_implied_prob(pd.Series([home_odds])).iloc[0]
    away_prob = american_to_implied_prob(pd.Series([away_odds])).iloc[0]
    return abs(home_prob - away_prob)


def favorite_odds(home_odds: float, away_odds: float) -> float:
    if pd.isna(home_odds) or pd.isna(away_odds):
        return np.nan
    return min(home_odds, away_odds)


def dog_odds(home_odds: float, away_odds: float) -> float:
    if pd.isna(home_odds) or pd.isna(away_odds):
        return np.nan
    return max(home_odds, away_odds)


def build_moneyline_snapshot(df: pd.DataFrame, label_prefix: str) -> pd.DataFrame:
    ml = df[df["market"].astype(str).str.lower() == "moneyline"].copy()
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
        raise ValueError(f"No usable moneyline rows found for {label_prefix}")

    snap = (
        ml.groupby(["game_id", "side"], as_index=False)
        .agg(
            odds=("odds", "median"),
            books=("book_key", "nunique"),
        )
    )

    home = (
        snap.loc[snap["side"] == "home", ["game_id", "odds", "books"]]
        .rename(columns={"odds": f"{label_prefix}_home_ml", "books": f"{label_prefix}_home_books"})
        .copy()
    )

    away = (
        snap.loc[snap["side"] == "away", ["game_id", "odds", "books"]]
        .rename(columns={"odds": f"{label_prefix}_away_ml", "books": f"{label_prefix}_away_books"})
        .copy()
    )

    out = home.merge(away, on="game_id", how="outer")
    return out


def derive_open_snapshot_from_market_priors(mp: pd.DataFrame) -> pd.DataFrame:
    needed = {"game_id", "market_open_home_ml", "market_open_away_ml"}
    missing = needed - set(mp.columns)
    if missing:
        raise ValueError(f"market_priors.csv missing columns needed for opening favorite tracker: {sorted(missing)}")

    cols = ["game_id", "market_open_home_ml", "market_open_away_ml"]
    if "market_open_home_ml_books" in mp.columns:
        cols.append("market_open_home_ml_books")
    if "market_open_away_ml_books" in mp.columns:
        cols.append("market_open_away_ml_books")

    out = mp[cols].drop_duplicates(subset=["game_id"]).copy()

    rename_map = {
        "market_open_home_ml": "open_home_ml",
        "market_open_away_ml": "open_away_ml",
        "market_open_home_ml_books": "open_home_books",
        "market_open_away_ml_books": "open_away_books",
    }
    out = out.rename(columns=rename_map)

    return out


def main() -> None:
    print("Loading closing lines...")
    closing = pd.read_csv(CLOSING_LINES_PATH, low_memory=False)

    print("Loading market priors...")
    priors = pd.read_csv(MARKET_PRIORS_PATH, low_memory=False)

    print("Building closing snapshot...")
    close_snap = build_moneyline_snapshot(closing, "close")
    close_snap = close_snap.rename(
        columns={
            "close_home_ml": "closing_home_ml",
            "close_away_ml": "closing_away_ml",
            "close_home_books": "closing_home_books",
            "close_away_books": "closing_away_books",
        }
    )

    print("Building opening snapshot...")
    open_snap = derive_open_snapshot_from_market_priors(priors)

    df = open_snap.merge(close_snap, on="game_id", how="outer")

    numeric_cols = [
        "open_home_ml", "open_away_ml",
        "closing_home_ml", "closing_away_ml",
        "open_home_books", "open_away_books",
        "closing_home_books", "closing_away_books",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_favorite_side"] = [
        side_from_odds(h, a) for h, a in zip(df["open_home_ml"], df["open_away_ml"])
    ]
    df["closing_favorite_side"] = [
        side_from_odds(h, a) for h, a in zip(df["closing_home_ml"], df["closing_away_ml"])
    ]

    df["open_favorite_odds"] = [
        favorite_odds(h, a) for h, a in zip(df["open_home_ml"], df["open_away_ml"])
    ]
    df["closing_favorite_odds"] = [
        favorite_odds(h, a) for h, a in zip(df["closing_home_ml"], df["closing_away_ml"])
    ]

    df["open_dog_odds"] = [
        dog_odds(h, a) for h, a in zip(df["open_home_ml"], df["open_away_ml"])
    ]
    df["closing_dog_odds"] = [
        dog_odds(h, a) for h, a in zip(df["closing_home_ml"], df["closing_away_ml"])
    ]

    df["open_favorite_strength"] = [
        favorite_strength(h, a) for h, a in zip(df["open_home_ml"], df["open_away_ml"])
    ]
    df["closing_favorite_strength"] = [
        favorite_strength(h, a) for h, a in zip(df["closing_home_ml"], df["closing_away_ml"])
    ]

    df["favorite_flip"] = np.where(
        df["open_favorite_side"].notna() & df["closing_favorite_side"].notna(),
        df["open_favorite_side"] != df["closing_favorite_side"],
        np.nan,
    )

    df["favorite_strength_delta"] = df["closing_favorite_strength"] - df["open_favorite_strength"]

    df["home_ml_move"] = df["closing_home_ml"] - df["open_home_ml"]
    df["away_ml_move"] = df["closing_away_ml"] - df["open_away_ml"]

    # Negative move means that side got more favored in American odds terms
    df["home_became_more_favored"] = df["home_ml_move"] < 0
    df["away_became_more_favored"] = df["away_ml_move"] < 0

    df["market_pressure_side"] = np.select(
        [
            df["home_became_more_favored"] & ~df["away_became_more_favored"],
            df["away_became_more_favored"] & ~df["home_became_more_favored"],
            df["home_became_more_favored"] & df["away_became_more_favored"],
        ],
        [
            "home",
            "away",
            "both_or_crossed",
        ],
        default="none_or_unclear",
    )

    df["open_home_implied"] = american_to_implied_prob(df["open_home_ml"])
    df["open_away_implied"] = american_to_implied_prob(df["open_away_ml"])
    df["closing_home_implied"] = american_to_implied_prob(df["closing_home_ml"])
    df["closing_away_implied"] = american_to_implied_prob(df["closing_away_ml"])

    df["home_implied_move"] = df["closing_home_implied"] - df["open_home_implied"]
    df["away_implied_move"] = df["closing_away_implied"] - df["open_away_implied"]

    out_cols = [
        "game_id",
        "open_home_ml", "open_away_ml",
        "closing_home_ml", "closing_away_ml",
        "open_home_books", "open_away_books",
        "closing_home_books", "closing_away_books",
        "open_favorite_side", "closing_favorite_side",
        "open_favorite_odds", "closing_favorite_odds",
        "open_dog_odds", "closing_dog_odds",
        "open_favorite_strength", "closing_favorite_strength",
        "favorite_strength_delta",
        "favorite_flip",
        "home_ml_move", "away_ml_move",
        "home_implied_move", "away_implied_move",
        "market_pressure_side",
    ]

    out = df[out_cols].copy()
    out = out.sort_values("game_id", kind="stable").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(out):,}")

    print("\nSample:")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
