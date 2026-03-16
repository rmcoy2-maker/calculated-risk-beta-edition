import pandas as pd

files = [
r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_2020_weekly.csv",
r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_2021_weekly.csv",
r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_historical_odds_2020_2022_weekly_combined.csv"
]

for f in files:
    print("\n"+"="*80)
    print("FILE:", f)

    df = pd.read_csv(f)

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "commence_time" in df.columns:
        ct = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
        print("Commence years:")
        print(ct.dt.year.value_counts().sort_index())

    if "snapshot_timestamp" in df.columns:
        st = pd.to_datetime(df["snapshot_timestamp"], errors="coerce", utc=True)
        print("Snapshot years:")
        print(st.dt.year.value_counts().sort_index())

    print("\nSample:")
    print(df.head(3))