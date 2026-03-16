from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PROBS = PROJECT_ROOT / "exports" / "model_probs.csv"
LINES = PROJECT_ROOT / "exports" / "lines_live.csv"
OUT = PROJECT_ROOT / "exports" / "edges_master.csv"

print("Loading model probabilities...")
mp = pd.read_csv(MODEL_PROBS, low_memory=False)

print("Loading sportsbook lines...")
lines = pd.read_csv(LINES, low_memory=False)

df = lines.merge(mp, on="game_id", how="left", suffixes=("_line", "_model"))


def american_to_prob(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(
        odds > 0,
        100 / (odds + 100),
        np.where(odds < 0, (-odds) / ((-odds) + 100), np.nan)
    )


df["away_team"] = df["away_team_line"]
df["home_team"] = df["home_team_line"]

df["side"] = df["side"].astype(str).str.strip().str.lower()
df["market"] = df["market"].astype(str).str.strip().str.lower()

df["implied_prob"] = american_to_prob(df["odds"])

df["model_prob"] = np.where(
    df["side"] == "home",
    df["p_home_win"],
    np.where(df["side"] == "away", df["p_away_win"], np.nan)
)

df["edge"] = df["model_prob"] - df["implied_prob"]

keep_cols = [
    "game_id",
    "Season",
    "Week",
    "game_date",
    "away_team",
    "home_team",
    "book",
    "market",
    "side",
    "odds",
    "pred_home_margin",
    "pred_total",
    "model_prob",
    "implied_prob",
    "edge",
]

keep_cols = [c for c in keep_cols if c in df.columns]

df_out = df[keep_cols].copy()
df_out = df_out.sort_values("edge", ascending=False, kind="stable")

OUT.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Rows:", len(df_out))
print(df_out.head(10).to_string(index=False))