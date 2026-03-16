import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, low_memory=False)

# Find best available team column
team_candidates = [
    "selected_team", "selection", "team", "pick_team", "team_pick",
    "HomeTeam", "home_team", "home", "AwayTeam", "away_team", "away"
]
season_candidates = ["Season", "season"]

team_col = next((c for c in team_candidates if c in df.columns), None)
season_col = next((c for c in season_candidates if c in df.columns), None)

if season_col is None:
    raise ValueError(f"No season column found. Columns: {df.columns.tolist()}")

if team_col is None:
    raise ValueError(f"No team column found. Columns: {df.columns.tolist()}")

out = (
    df.groupby([season_col, team_col], dropna=False)
      .size()
      .reset_index(name="bets")
      .sort_values([season_col, "bets"], ascending=[True, False])
)

print(f"\nUsing season column: {season_col}")
print(f"Using team column: {team_col}\n")
print(out.head(40).to_string(index=False))

out.to_csv("analysis_out/team_exposure_by_season.csv", index=False)
print("\nSaved: analysis_out/team_exposure_by_season.csv")