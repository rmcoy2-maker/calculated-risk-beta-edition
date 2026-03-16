import pandas as pd

df = pd.read_csv("analysis_out/true_line/true_line_board.csv", low_memory=False)

print("\nTotal rows:", len(df))

unique_games = df["game_id"].nunique()

print("Unique games:", unique_games)

dupes = df.groupby("game_id").size().sort_values(ascending=False)

print("\nTop duplicate games:")
print(dupes.head(20))

print("\nRows per game distribution:")
print(dupes.value_counts().sort_index())