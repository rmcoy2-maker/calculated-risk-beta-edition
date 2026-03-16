import pandas as pd
import numpy as np

RAW_PATH = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_combined.csv"
OUT_PATH = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_open_mid_close_odds.csv"

KEY_COLS = [
    "event_id",
    "home_team",
    "away_team",
    "commence_time",
    "book_key",
    "book_title",
    "market_key",
    "outcome_name",
]

MID_HOURS_BEFORE_KICKOFF = 96  # ~4 days before game


def main():
    print("Loading raw odds file...")
    odds = pd.read_csv(RAW_PATH)

    print("Converting timestamps...")
    odds["snapshot_timestamp"] = pd.to_datetime(odds["snapshot_timestamp"], utc=True)
    odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True)

    odds = odds.sort_values("snapshot_timestamp").copy()

    print("Building OPEN snapshot...")
    opening = (
        odds.groupby(KEY_COLS, as_index=False)
        .first()
        .copy()
    )

    opening = opening[KEY_COLS + ["snapshot_timestamp", "outcome_price", "outcome_point"]]
    opening = opening.rename(columns={
        "snapshot_timestamp": "open_snapshot_timestamp",
        "outcome_price": "open_price",
        "outcome_point": "open_point",
    })

    print("Building CLOSE snapshot...")
    closing = (
        odds.groupby(KEY_COLS, as_index=False)
        .last()
        .copy()
    )

    closing = closing[KEY_COLS + ["snapshot_timestamp", "outcome_price", "outcome_point"]]
    closing = closing.rename(columns={
        "snapshot_timestamp": "close_snapshot_timestamp",
        "outcome_price": "close_price",
        "outcome_point": "close_point",
    })

    print("Building MID snapshot (~96 hours before kickoff)...")
    odds["target_mid_timestamp"] = odds["commence_time"] - pd.Timedelta(hours=MID_HOURS_BEFORE_KICKOFF)
    odds["mid_diff_seconds"] = (
        odds["snapshot_timestamp"] - odds["target_mid_timestamp"]
    ).abs().dt.total_seconds()

    mid = (
        odds.sort_values(["mid_diff_seconds", "snapshot_timestamp"])
        .groupby(KEY_COLS, as_index=False)
        .first()
        .copy()
    )

    mid = mid[KEY_COLS + ["snapshot_timestamp", "outcome_price", "outcome_point", "mid_diff_seconds"]]
    mid = mid.rename(columns={
        "snapshot_timestamp": "mid_snapshot_timestamp",
        "outcome_price": "mid_price",
        "outcome_point": "mid_point",
    })

    print("Merging OPEN + MID + CLOSE...")
    out = opening.merge(mid, on=KEY_COLS, how="inner").merge(closing, on=KEY_COLS, how="inner")

    print("Computing movement fields...")
    out["price_move_open_to_mid"] = out["mid_price"] - out["open_price"]
    out["price_move_mid_to_close"] = out["close_price"] - out["mid_price"]
    out["price_move_open_to_close"] = out["close_price"] - out["open_price"]

    out["point_move_open_to_mid"] = out["mid_point"] - out["open_point"]
    out["point_move_mid_to_close"] = out["close_point"] - out["mid_point"]
    out["point_move_open_to_close"] = out["close_point"] - out["open_point"]

    print("Saving...")
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(out):,}")
    print("\nColumns:")
    print(out.columns.tolist())

    print("\nMarket counts:")
    print(out["market_key"].value_counts(dropna=False))

    print("\nMID coverage:")
    print(out["mid_price"].notna().mean())

    print("\nMID timing diagnostics (hours from target):")
    hours_from_target = out["mid_diff_seconds"] / 3600.0
    print(hours_from_target.describe())

    print("\nSample rows:")
    print(out.head(10))


if __name__ == "__main__":
    main()