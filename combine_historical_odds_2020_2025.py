import pandas as pd

weekly = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_2020_2022_weekly_combined.csv"
odds_2023 = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_2023.csv"
odds_2024 = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_combined.csv"

out_path = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_2020_2025_master.csv"

print("Loading files...")

df1 = pd.read_csv(weekly)
df2 = pd.read_csv(odds_2023)
df3 = pd.read_csv(odds_2024)

print("Rows:")
print("2020-2022:", len(df1))
print("2023:", len(df2))
print("2024+:", len(df3))

common_cols = sorted(set(df1.columns) & set(df2.columns) & set(df3.columns))
print("Common columns:", common_cols)

df1 = df1[common_cols]
df2 = df2[common_cols]
df3 = df3[common_cols]

combined = pd.concat([df1, df2, df3], ignore_index=True)

combined["commence_time"] = pd.to_datetime(combined["commence_time"], utc=True, errors="coerce")

combined = combined[
    combined["commence_time"].dt.year.between(2020, 2025)
]

combined = combined.sort_values("commence_time")

combined.to_csv(out_path, index=False)

print("\nSaved master odds file:")
print(out_path)

print("\nSeason counts:")
print(combined["commence_time"].dt.year.value_counts().sort_index())