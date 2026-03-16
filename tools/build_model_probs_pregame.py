from pathlib import Path
import json
import pickle

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "exports" / "pregame_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_PATH = PROJECT_ROOT / "exports" / "model_probs.csv"


def main() -> None:
    print("Loading models...")

    with open(MODEL_DIR / "spread_model.pkl", "rb") as f:
        spread_model = pickle.load(f)

    with open(MODEL_DIR / "total_model.pkl", "rb") as f:
        total_model = pickle.load(f)

    with open(MODEL_DIR / "win_model.pkl", "rb") as f:
        win_model = pickle.load(f)

    with open(MODEL_DIR / "model_features.json", "r", encoding="utf-8") as f:
        FEATURES = json.load(f)

    print("Loading pregame features...")
    df = pd.read_csv(DATA_PATH, low_memory=False).copy()

    required_meta = [
        "game_id", "Season", "Week", "game_date", "away_team", "home_team"
    ]
    missing_meta = [c for c in required_meta if c not in df.columns]
    if missing_meta:
        raise ValueError(f"pregame_features.csv missing required columns: {missing_meta}")

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing model feature columns: {missing_features[:20]}")

    X = df[FEATURES].copy()
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)

    print("Building predictions...")
    df["pred_home_margin"] = spread_model.predict(X)
    df["pred_total"] = total_model.predict(X)

    win_probs = win_model.predict_proba(X)
    df["p_home_win"] = win_probs[:, 1]
    df["p_away_win"] = win_probs[:, 0]

    out_cols = required_meta + [
        "pred_home_margin",
        "pred_total",
        "p_home_win",
        "p_away_win",
    ]

    df_out = df[out_cols].copy()
    df_out.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(df_out):,}")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()