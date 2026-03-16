from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-features",
        type=Path,
        default=PROJECT_ROOT / "exports" / "pregame_features.csv",
    )

    parser.add_argument(
        "--market-priors",
        type=Path,
        default=PROJECT_ROOT / "exports" / "market_priors.csv",
    )

    parser.add_argument(
        "--model-out-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    train_path = args.train_features
    market_path = args.market_priors
    model_dir = args.model_out_dir

    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pregame features...")
    pre = pd.read_csv(train_path, low_memory=False)

    print("Loading market priors...")
    mk = pd.read_csv(market_path, low_memory=False)

    df = pre.merge(mk, on="game_id", how="left")
    df = df[df["home_win"].isin([0, 1])].copy()

    drop_cols = {
        "game_id",
        "Season",
        "Week",
        "game_date",
        "away_team",
        "home_team",
        "home_score",
        "away_score",
        "margin",
        "total_points",
        "home_win",
        "away_win",
    }

    banned_substrings = [
        "score",
        "margin",
        "win",
        "points",
        "total_points",
        "for",
        "against",
    ]

    usable = []

    for c in df.columns:

        cl = c.lower()

        if c in drop_cols:
            continue

        if c.startswith("market_"):
            usable.append(c)
            continue

        if any(b in cl for b in banned_substrings):
            continue

        usable.append(c)

    market_cols = [c for c in usable if c.startswith("market_")]
    diff_cols = [c for c in usable if c.startswith("diff_")]
    away_cols = [c for c in usable if c.startswith("away_")]
    home_cols = [c for c in usable if c.startswith("home_")]
    other_cols = [c for c in usable if c not in market_cols + diff_cols + away_cols + home_cols]

    feature_cols = market_cols + diff_cols + away_cols + home_cols + other_cols

    MAX_FEATURES = 120
    feature_cols = feature_cols[:MAX_FEATURES]

    print("Selected feature sample:")
    print(feature_cols[:40])

    print("market count:", sum(c.startswith("market_") for c in feature_cols))
    print("diff count:", sum(c.startswith("diff_") for c in feature_cols))
    print("away count:", sum(c.startswith("away_") for c in feature_cols))
    print("home count:", sum(c.startswith("home_") for c in feature_cols))

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    y_win = df["home_win"]
    y_spread = df["margin"]
    y_total = df["total_points"]

    seasons = sorted(df["Season"].dropna().unique())
    test_season = seasons[-1]

    train_mask = df["Season"] < test_season
    test_mask = df["Season"] == test_season

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train_win = y_win.loc[train_mask]
    y_test_win = y_win.loc[test_mask]

    y_train_spread = y_spread.loc[train_mask]
    y_test_spread = y_spread.loc[test_mask]

    y_train_total = y_total.loc[train_mask]
    y_test_total = y_total.loc[test_mask]

    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Feature count:", len(feature_cols))

    print("Training win model...")

    win_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1,
    )

    win_model.fit(X_train, y_train_win)

    win_pred = win_model.predict(X_test)
    win_prob = win_model.predict_proba(X_test)[:, 1]

    print("Training spread model...")

    spread_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1,
    )

    spread_model.fit(X_train, y_train_spread)
    spread_pred = spread_model.predict(X_test)

    print("Training total model...")

    total_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1,
    )

    total_model.fit(X_train, y_train_total)
    total_pred = total_model.predict(X_test)

    metrics = {
        "test_season": int(test_season),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(feature_cols)),
        "win_accuracy": float(accuracy_score(y_test_win, win_pred)),
        "win_auc": float(roc_auc_score(y_test_win, win_prob)),
        "spread_mae": float(mean_absolute_error(y_test_spread, spread_pred)),
        "total_mae": float(mean_absolute_error(y_test_total, total_pred)),
    }

    print("\nMetrics:")

    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(model_dir / "win_model.pkl", "wb") as f:
        pickle.dump(win_model, f)

    with open(model_dir / "spread_model.pkl", "wb") as f:
        pickle.dump(spread_model, f)

    with open(model_dir / "total_model.pkl", "wb") as f:
        pickle.dump(total_model, f)

    with open(model_dir / "model_features.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(model_dir / "pregame_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModels saved to:", model_dir)


if __name__ == "__main__":
    main()