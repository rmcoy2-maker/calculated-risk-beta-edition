import pandas as pd

file_path = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_combined.csv"
out_path = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_open_close_odds.csv"

odds = pd.read_csv(file_path)

odds["snapshot_timestamp"] = pd.to_datetime(odds["snapshot_timestamp"], utc=True)
odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True)

key_cols = [
    "event_id",
    "home_team",
    "away_team",
    "commence_time",
    "book_key",
    "book_title",
    "market_key",
    "outcome_name",
]

# earliest snapshot = open
opening = (
    odds.sort_values("snapshot_timestamp")
        .groupby(key_cols, as_index=False)
        .first()
        .copy()
)

opening = opening[key_cols + ["snapshot_timestamp", "outcome_price", "outcome_point"]]
opening = opening.rename(columns={
    "snapshot_timestamp": "open_snapshot_timestamp",
    "outcome_price": "open_price",
    "outcome_point": "open_point",
})

# latest snapshot = close
closing = (
    odds.sort_values("snapshot_timestamp")
        .groupby(key_cols, as_index=False)
        .last()
        .copy()
)

closing = closing[key_cols + ["snapshot_timestamp", "outcome_price", "outcome_point"]]
closing = closing.rename(columns={
    "snapshot_timestamp": "close_snapshot_timestamp",
    "outcome_price": "close_price",
    "outcome_point": "close_point",
})

open_close = opening.merge(closing, on=key_cols, how="inner")

# movement helpers
open_close["price_move"] = open_close["close_price"] - open_close["open_price"]
open_close["point_move"] = open_close["close_point"] - open_close["open_point"]

open_close.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(open_close))
print(open_close.head(10))
print("\nMarkets:")
print(open_close["market_key"].value_counts(dropna=False))