from pathlib import Path
import json
import pickle

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "exports" / "games_master.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_PATH = PROJECT_ROOT / "exports" / "model_probs.csv"


print("Loading models...")

with open(MODEL_DIR / "spread_model.pkl", "rb") as f:
    spread_model = pickle.load(f)

with open(MODEL_DIR / "total_model.pkl", "rb") as f:
    total_model = pickle.load(f)

with open(MODEL_DIR / "win_model.pkl", "rb") as f:
    win_model = pickle.load(f)

with open(MODEL_DIR / "model_features.json", "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Remove preseason / Hall of Fame games
df = df[
    ~df["Week"].astype(str).str.contains("Preseason|Hall Of Fame", case=False, na=False)
].copy()
missing_features = [c for c in FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(f"Missing model feature columns: {missing_features}")

print("Building predictions...")
X = df[FEATURES]

df["pred_home_margin"] = spread_model.predict(X)
df["pred_total"] = total_model.predict(X)

win_probs = win_model.predict_proba(X)
df["p_home_win"] = win_probs[:, 1]
df["p_away_win"] = win_probs[:, 0]

output_cols = [
    "game_id",
    "Season",
    "Week",
    "game_date",
    "away_team",
    "home_team",
    "pred_home_margin",
    "pred_total",
    "p_home_win",
    "p_away_win",
]

output_cols = [c for c in output_cols if c in df.columns]

df_out = df[output_cols].copy()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Rows:", len(df_out))
print(df_out.head())