from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDGES_PATH = PROJECT_ROOT / "exports" / "edges_master.csv"
OUT_PATH = PROJECT_ROOT / "exports" / "top_edges.csv"

df = pd.read_csv(EDGES_PATH, low_memory=False)

# Basic sanity filters
df = df[df["market"].astype(str).str.lower() == "moneyline"].copy()
df = df[df["model_prob"].between(0.05, 0.95)].copy()
df = df[df["edge"].notna()].copy()

# Keep strongest edges
df = df.sort_values("edge", ascending=False, kind="stable")

top = df[[
    "Season",
    "Week",
    "game_date",
    "away_team",
    "home_team",
    "side",
    "odds",
    "model_prob",
    "implied_prob",
    "edge",
    "pred_home_margin",
    "pred_total",
]].copy()

top.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Rows:", len(top))
print(top.head(25).to_string(index=False))