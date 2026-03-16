import pandas as pd

PATH = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_open_mid_close_odds.csv"

df = pd.read_csv(PATH)

print("Rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nMarket counts:")
print(df["market_key"].value_counts(dropna=False))

print("\nExample price columns:")
print(df[[
    "market_key",
    "outcome_name",
    "open_price",
    "mid_price",
    "close_price",
    "open_point",
    "mid_point",
    "close_point"
]].head(20))

print("\nMovement summaries:")
for col in [
    "price_move_open_to_mid",
    "price_move_mid_to_close",
    "price_move_open_to_close",
    "point_move_open_to_mid",
    "point_move_mid_to_close",
    "point_move_open_to_close",
]:
    print(f"\n{col}")
    print(df[col].describe())