import pandas as pd

df = pd.read_csv("analysis_out/true_line/true_line_board.csv", low_memory=False)

counts = df.groupby("game_id")["selected_team"].nunique()

two_sided = (counts == 2).sum()
one_sided = (counts == 1).sum()

print("Games with BOTH sides selected:", two_sided)
print("Games with ONE side selected:", one_sided)

print("\nPercentage two-sided:",
      two_sided / len(counts))