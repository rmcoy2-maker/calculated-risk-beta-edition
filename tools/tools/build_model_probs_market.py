from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-features",
        type=Path,
        default=PROJECT_ROOT / "exports" / "pregame_features.csv",
    )

    parser.add_argument(
        "--market-priors",
        type=Path,
        default=PROJECT_ROOT / "exports" / "market_priors.csv",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
    )

    parser.add_argument(
        "--out-path",
        type=Path,
        default=PROJECT_ROOT / "exports" / "model_probs.csv",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    test_features_path = args.test_features
    market_path = args.market_priors
    model_dir = args.model_dir
    out_path = args.out_path

    print("Loading models...")

    with open(model_dir / "win_model.pkl", "rb") as f:
        win_model = pickle.load(f)

    with open(model_dir / "spread_model.pkl", "rb") as f:
        spread_model = pickle.load(f)

    with open(model_dir / "total_model.pkl", "rb") as f:
        total_model = pickle.load(f)

    with open(model_dir / "model_features.json", "r") as f:
        feature_cols = json.load(f)

    print("Loading pregame features...")
    pre = pd.read_csv(test_features_path, low_memory=False)

    print("Loading market priors...")
    mk = pd.read_csv(market_path, low_memory=False)

    df = pre.merge(mk, on="game_id", how="left").copy()

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].replace([float("inf"), float("-inf")], 0).fillna(0)

    print("Building predictions...")

    df["pred_home_margin"] = spread_model.predict(X)
    df["pred_total"] = total_model.predict(X)
    df["p_home_win"] = win_model.predict_proba(X)[:, 1]
    df["p_away_win"] = 1 - df["p_home_win"]

    keep_cols = [
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

    out = df[keep_cols].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(out):,}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()