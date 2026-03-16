import pandas as pd

raw_path = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_combined.csv"
out_path = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_open_wed_close_odds.csv"

odds = pd.read_csv(raw_path)

odds["snapshot_timestamp"] = pd.to_datetime(odds["snapshot_timestamp"], utc=True)
odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True)

# Convert to Eastern time for Wed 2–4 PM rule
snap_et = odds["snapshot_timestamp"].dt.tz_convert("America/New_York")
odds["snap_et"] = snap_et
odds["weekday_et"] = snap_et.dt.weekday   # Wed = 2
odds["hour_et"] = snap_et.dt.hour
odds["minute_et"] = snap_et.dt.minute

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

# OPEN = earliest snapshot
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

# CLOSE = latest snapshot
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

# WED PM = snapshot closest to Wednesday 3:00 PM ET, restricted to 2–4 PM ET
wed_window = odds[
    (odds["weekday_et"] == 2) &
    (odds["hour_et"] >= 14) &
    (odds["hour_et"] <= 16)
].copy()

wed_window["minutes_from_3pm"] = (
    (wed_window["hour_et"] - 15).abs() * 60 +
    wed_window["minute_et"].sub(0).abs()
)

wed_pm = (
    wed_window.sort_values(["minutes_from_3pm", "snapshot_timestamp"])
    .groupby(key_cols, as_index=False)
    .first()
    .copy()
)
wed_pm = wed_pm[key_cols + ["snapshot_timestamp", "outcome_price", "outcome_point"]]
wed_pm = wed_pm.rename(columns={
    "snapshot_timestamp": "wed_snapshot_timestamp",
    "outcome_price": "wed_price",
    "outcome_point": "wed_point",
})

# Merge all three states
out = opening.merge(wed_pm, on=key_cols, how="left").merge(closing, on=key_cols, how="inner")

# Movement helpers
out["price_move_open_to_wed"] = out["wed_price"] - out["open_price"]
out["price_move_wed_to_close"] = out["close_price"] - out["wed_price"]
out["price_move_open_to_close"] = out["close_price"] - out["open_price"]

out["point_move_open_to_wed"] = out["wed_point"] - out["open_point"]
out["point_move_wed_to_close"] = out["close_point"] - out["wed_point"]
out["point_move_open_to_close"] = out["close_point"] - out["open_point"]

out.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(out))
print("\nColumns:")
print(out.columns.tolist())
print("\nMarket counts:")
print(out["market_key"].value_counts(dropna=False))
print("\nWed snapshot coverage:")
print(out["wed_price"].notna().mean())
print(out.head(10))