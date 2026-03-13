import pandas as pd

print("Loading files...")

scores = pd.read_csv("exports/scores_1966-2025.csv")
odds = pd.read_csv("exports/games_with_odds.csv", sep="\t")

# ---------------------------
# Normalize score columns
# ---------------------------
scores = scores.rename(columns={
    "Season": "season",
    "_DateISO": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "HomeScore": "home_score",
    "AwayScore": "away_score"
})

scores["date"] = pd.to_datetime(scores["date"], errors="coerce").dt.date

# ---------------------------
# Normalize odds columns
# ---------------------------
odds = odds.rename(columns={
    "game_date": "date",
    "home_team": "home",
    "away_team": "away",
    "spread_close": "spread_home",
    "total_close": "total_close",
})

odds["date"] = pd.to_datetime(odds["date"], errors="coerce").dt.date

# keep only one row per game/date/team combo if duplicates exist
odds = odds.sort_values(["date", "away", "home"]).drop_duplicates(
    subset=["date", "away", "home"],
    keep="first"
)

# ---------------------------
# Merge
# ---------------------------
print("Merging...")

merged = scores.merge(
    odds[["date", "home", "away", "spread_home", "total_close", "ml_home", "ml_away"]],
    on=["date", "home", "away"],
    how="left"
)

# ---------------------------
# Save
# ---------------------------
print("Saving merged dataset...")

merged.to_csv("exports/scores_1966-2025.csv", index=False)

print("Done. File updated with spreads and totals.")
print("Rows:", len(merged))
print("Columns added:", [c for c in ["spread_home", "total_close", "ml_home", "ml_away"] if c in merged.columns])