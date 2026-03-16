from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "exports" / "games_master.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df


def filter_games(df: pd.DataFrame) -> pd.DataFrame:
    if "Week" in df.columns:
        df = df[
            ~df["Week"].astype(str).str.contains("Preseason|Hall Of Fame", case=False, na=False)
        ].copy()
    return df


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    if "margin" not in df.columns and {"home_score", "away_score"}.issubset(df.columns):
        df["margin"] = df["home_score"] - df["away_score"]

    if "total_points" not in df.columns and {"home_score", "away_score"}.issubset(df.columns):
        df["total_points"] = df["home_score"] + df["away_score"]

    if "home_win" not in df.columns and "margin" in df.columns:
        df["home_win"] = (df["margin"] > 0).astype(int)

    return df


def choose_features(df: pd.DataFrame) -> list[str]:
    exclude = {
        # identifiers / metadata
        "game_id",
        "game_date",
        "game_date_str",
        "Date",
        "Date_std",
        "AwayTeam",
        "HomeTeam",
        "AwayTeam_std",
        "HomeTeam_std",
        "Season_score",
        "Date_std_score",
        "AwayTeam_std_score",
        "HomeTeam_std_score",
        "score_found",
        "season_type",
        "PostSeason",

        # targets / direct outcomes
        "home_score",
        "away_score",
        "margin",
        "total_points",
        "home_win",
        "away_win",

        # obvious leakage / same-game scoring outcomes
        "home_scoring_plays",
        "away_scoring_plays",
        "home_touchdowns",
        "away_touchdowns",
        "combined_scoring_plays",
        "combined_touchdowns",

        # same-game high leakage volume summaries
        "combined_total_plays",
        "combined_offensive_plays",
        "combined_drive_count",
        "combined_scoring_drive_rate",

        # direct same-game side labels
        "home_home_or_away",
        "away_home_or_away",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]

    # require decent coverage
    features = [c for c in features if df[c].notna().sum() >= 100]

    return features


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Season" in df.columns:
        seasons = sorted(df["Season"].dropna().unique())
        if len(seasons) >= 2:
            test_season = seasons[-1]
            train_df = df[df["Season"] < test_season].copy()
            test_df = df[df["Season"] == test_season].copy()
            return train_df, test_df

    split_idx = int(len(df) * 0.8)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_calibrated_classifier():
    # Impute outside calibration so the calibrated classifier sees clean numeric arrays
    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3,
    )
    return calibrated_model


def main() -> None:
    print("Loading games_master...")
    df = load_data()
    df = filter_games(df)
    df = prepare_targets(df)

    required = ["margin", "total_points", "home_win"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    model_df = df.dropna(subset=["margin", "total_points", "home_win"]).copy()

    if "Season" in model_df.columns and "week_sort" in model_df.columns:
        model_df = model_df.sort_values(["Season", "week_sort"], kind="stable")
    elif "Season" in model_df.columns and "Week" in model_df.columns:
        model_df = model_df.sort_values(["Season", "Week"], kind="stable")
    elif "game_date" in model_df.columns:
        model_df = model_df.sort_values(["game_date"], kind="stable")

    features = choose_features(model_df)
    if not features:
        raise ValueError("No usable numeric features found.")

    print(f"Using {len(features)} features")
    print("Feature sample:", features[:20])

    train_df, test_df = time_split(model_df)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    X_train = train_df[features]
    X_test = test_df[features]

    y_margin_train = train_df["margin"]
    y_margin_test = test_df["margin"]

    y_total_train = train_df["total_points"]
    y_total_test = test_df["total_points"]

    y_win_train = train_df["home_win"].astype(int)
    y_win_test = test_df["home_win"].astype(int)

    spread_model = build_regressor()
    total_model = build_regressor()
    win_model = build_calibrated_classifier()

    print("Training spread model...")
    spread_model.fit(X_train, y_margin_train)

    print("Training total model...")
    total_model.fit(X_train, y_total_train)

    print("Training calibrated win model...")
    win_model.fit(X_train, y_win_train)

    pred_margin = spread_model.predict(X_test)
    pred_total = total_model.predict(X_test)
    pred_win = win_model.predict(X_test)
    pred_win_prob = win_model.predict_proba(X_test)[:, 1]

    metrics = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(features)),
        "spread_mae": float(mean_absolute_error(y_margin_test, pred_margin)),
        "total_mae": float(mean_absolute_error(y_total_test, pred_total)),
        "win_accuracy": float(accuracy_score(y_win_test, pred_win)),
        "win_auc": float(roc_auc_score(y_win_test, pred_win_prob)),
        "p_home_win_min": float(np.min(pred_win_prob)),
        "p_home_win_25": float(np.quantile(pred_win_prob, 0.25)),
        "p_home_win_50": float(np.quantile(pred_win_prob, 0.50)),
        "p_home_win_75": float(np.quantile(pred_win_prob, 0.75)),
        "p_home_win_max": float(np.max(pred_win_prob)),
    }

    with open(MODEL_DIR / "spread_model.pkl", "wb") as f:
        pickle.dump(spread_model, f)

    with open(MODEL_DIR / "total_model.pkl", "wb") as f:
        pickle.dump(total_model, f)

    with open(MODEL_DIR / "win_model.pkl", "wb") as f:
        pickle.dump(win_model, f)

    with open(MODEL_DIR / "model_features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel Performance")
    print("-----------------")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nSaved models to: {MODEL_DIR}")


if __name__ == "__main__":
    main()