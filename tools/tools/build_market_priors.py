from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LINES_PATH = PROJECT_ROOT / "exports" / "lines_master.csv"
OUT_PATH = PROJECT_ROOT / "exports" / "market_priors.csv"


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


def keep_earliest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["requested_snapshot"] = pd.to_datetime(df["requested_snapshot"], errors="coerce", utc=True)
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)

    df = df[df["requested_snapshot"] <= df["commence_time"]].copy()

    df = df.sort_values(
        ["game_id", "book_key", "market", "side", "requested_snapshot"],
        kind="stable",
    )

    out = (
        df.groupby(["game_id", "book_key", "market", "side"], dropna=False, as_index=False)
          .head(1)
          .copy()
    )
    return out


def build_moneyline_priors(df: pd.DataFrame) -> pd.DataFrame:
    ml = df[df["market"].astype(str).str.lower() == "moneyline"].copy()
    ml["side"] = ml["side"].astype(str).str.lower().str.strip()
    ml["odds"] = pd.to_numeric(ml["odds"], errors="coerce")

    ml = ml[
        ml["side"].isin(["home", "away"]) &
        ml["odds"].notna() &
        (ml["odds"].abs() >= 100) &
        (ml["odds"].abs() <= 5000)
    ].copy()

    consensus = (
        ml.groupby(["game_id", "side"], as_index=False)
          .agg(
              opening_ml=("odds", "median"),
              ml_books=("book_key", "nunique"),
          )
    )

    home = consensus[consensus["side"] == "home"][["game_id", "opening_ml", "ml_books"]].rename(
        columns={"opening_ml": "market_open_home_ml", "ml_books": "market_open_home_ml_books"}
    )
    away = consensus[consensus["side"] == "away"][["game_id", "opening_ml", "ml_books"]].rename(
        columns={"opening_ml": "market_open_away_ml", "ml_books": "market_open_away_ml_books"}
    )

    out = home.merge(away, on="game_id", how="outer")

    out["market_open_home_implied_raw"] = american_to_implied_prob(out["market_open_home_ml"])
    out["market_open_away_implied_raw"] = american_to_implied_prob(out["market_open_away_ml"])

    denom = out["market_open_home_implied_raw"] + out["market_open_away_implied_raw"]
    out["market_open_home_implied_novig"] = out["market_open_home_implied_raw"] / denom
    out["market_open_away_implied_novig"] = out["market_open_away_implied_raw"] / denom

    return out


def build_spread_priors(df: pd.DataFrame) -> pd.DataFrame:
    sp = df[df["market"].astype(str).str.lower() == "spread"].copy()
    sp["side"] = sp["side"].astype(str).str.lower().str.strip()
    sp["point"] = pd.to_numeric(sp["point"], errors="coerce")
    sp["odds"] = pd.to_numeric(sp["odds"], errors="coerce")

    sp = sp[sp["side"].isin(["home", "away"])].copy()

    consensus = (
        sp.groupby(["game_id", "side"], as_index=False)
          .agg(
              spread_point=("point", "median"),
              spread_odds=("odds", "median"),
              spread_books=("book_key", "nunique"),
          )
    )

    home = consensus[consensus["side"] == "home"][["game_id", "spread_point", "spread_odds", "spread_books"]].rename(
        columns={
            "spread_point": "market_open_home_spread",
            "spread_odds": "market_open_home_spread_odds",
            "spread_books": "market_open_home_spread_books",
        }
    )
    away = consensus[consensus["side"] == "away"][["game_id", "spread_point", "spread_odds", "spread_books"]].rename(
        columns={
            "spread_point": "market_open_away_spread",
            "spread_odds": "market_open_away_spread_odds",
            "spread_books": "market_open_away_spread_books",
        }
    )

    out = home.merge(away, on="game_id", how="outer")
    return out


def build_total_priors(df: pd.DataFrame) -> pd.DataFrame:
    tot = df[df["market"].astype(str).str.lower() == "total"].copy()
    tot["side"] = tot["side"].astype(str).str.lower().str.strip()
    tot["point"] = pd.to_numeric(tot["point"], errors="coerce")
    tot["odds"] = pd.to_numeric(tot["odds"], errors="coerce")

    tot = tot[tot["side"].isin(["over", "under"])].copy()

    consensus = (
        tot.groupby(["game_id", "side"], as_index=False)
          .agg(
              total_point=("point", "median"),
              total_odds=("odds", "median"),
              total_books=("book_key", "nunique"),
          )
    )

    over = consensus[consensus["side"] == "over"][["game_id", "total_point", "total_odds", "total_books"]].rename(
        columns={
            "total_point": "market_open_total",
            "total_odds": "market_open_over_odds",
            "total_books": "market_open_total_books",
        }
    )
    under = consensus[consensus["side"] == "under"][["game_id", "total_point", "total_odds"]].rename(
        columns={
            "total_point": "market_open_total_under_point",
            "total_odds": "market_open_under_odds",
        }
    )

    out = over.merge(under, on="game_id", how="outer")
    return out


def main() -> None:
    print("Loading lines_master...")
    df = pd.read_csv(LINES_PATH, low_memory=False)

    required = ["game_id", "requested_snapshot", "commence_time", "book_key", "market", "side", "odds", "point"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"lines_master.csv missing columns: {missing}")

    print("Selecting earliest pre-kickoff snapshots...")
    early = keep_earliest_snapshot(df)

    print("Building opening moneyline priors...")
    ml = build_moneyline_priors(early)

    print("Building opening spread priors...")
    sp = build_spread_priors(early)

    print("Building opening total priors...")
    tot = build_total_priors(early)

    out = ml.merge(sp, on="game_id", how="outer").merge(tot, on="game_id", how="outer")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(out):,}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
