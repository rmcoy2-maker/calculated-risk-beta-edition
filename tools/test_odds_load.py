import pandas as pd

file_path = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_combined.csv"

print("Loading:", file_path)

odds = pd.read_csv(file_path)

print("Loaded rows:", len(odds))
print("Columns:", odds.columns.tolist())
print(odds.head())

# convert timestamps
odds["snapshot_timestamp"] = pd.to_datetime(odds["snapshot_timestamp"])
odds["commence_time"] = pd.to_datetime(odds["commence_time"])

# find closing lines
closing = (
    odds.sort_values("snapshot_timestamp")
        .groupby(["event_id","book_key","market_key","outcome_name"])
        .tail(1)
)

print("\nClosing rows:", len(closing))
print(closing.head())