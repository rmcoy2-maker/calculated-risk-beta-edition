from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "exports"
TOOLS_DIR = PROJECT_ROOT / "tools"

PREGAME_FEATURES_PATH = EXPORTS_DIR / "pregame_features.csv"

TRAIN_SCRIPT = TOOLS_DIR / "train_pregame_models_market.py"
PREDICT_SCRIPT = TOOLS_DIR / "build_model_probs_market.py"
BACKTEST_SCRIPT = TOOLS_DIR / "backtest_edges.py"

OUT_ROOT = EXPORTS_DIR / "walkforward_weekly"

# predictor is currently writing here no matter what
GLOBAL_MODEL_PROBS_PATH = EXPORTS_DIR / "model_probs.csv"

TARGET_SEASONS = [2024, 2025]


def run_cmd(cmd: list[str]) -> None:
    print("\nRUNNING:", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(str(x) for x in cmd)}")


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    print(f"Loading features: {PREGAME_FEATURES_PATH}")
    feats = pd.read_csv(PREGAME_FEATURES_PATH, low_memory=False)

    required = ["game_id", "Season", "Week", "game_date"]
    ensure_columns(feats, required, "pregame_features.csv")

    feats["Season"] = pd.to_numeric(feats["Season"], errors="coerce").astype("Int64")
    feats["Week"] = pd.to_numeric(feats["Week"], errors="coerce").astype("Int64")
    feats["game_date"] = pd.to_datetime(feats["game_date"], errors="coerce")

    feats = feats.dropna(subset=["Season", "Week", "game_date"]).copy()
    feats["Season"] = feats["Season"].astype(int)
    feats["Week"] = feats["Week"].astype(int)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    combined_summary_rows = []
    combined_bets = []

    for season in TARGET_SEASONS:
        season_df = feats.loc[feats["Season"] == season].copy()
        if season_df.empty:
            print(f"\nSkipping season {season}: no rows found.")
            continue

        weeks = sorted(season_df["Week"].dropna().unique().tolist())

        for week in weeks:
            test_df = season_df.loc[season_df["Week"] == week].copy()
            if test_df.empty:
                continue

            train_df = feats.loc[
                (feats["Season"] < season)
                | ((feats["Season"] == season) & (feats["Week"] < week))
            ].copy()

            if train_df.empty:
                print(f"\nSkipping season {season} week {week}: no prior training rows.")
                continue

            fold_dir = OUT_ROOT / f"season_{season}_week_{week:02d}"
            model_dir = fold_dir / "model_artifacts"

            train_path = fold_dir / "train_features.csv"
            test_path = fold_dir / "test_features.csv"
            fold_model_probs_path = fold_dir / "model_probs.csv"
            summary_path = fold_dir / "backtest_summary_moneyline.csv"
            bets_path = fold_dir / "backtest_bets_moneyline.csv"

            print("\n" + "=" * 80)
            print(f"WALK-FORWARD WEEK: train < {season} week {week}, test = {season} week {week}")
            print("=" * 80)

            save_csv(train_df, train_path)
            save_csv(test_df, test_path)

            print(f"Saved train rows: {len(train_df):,} -> {train_path}")
            print(f"Saved test rows:  {len(test_df):,} -> {test_path}")

            run_cmd(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--train-features",
                    str(train_path),
                    "--model-out-dir",
                    str(model_dir),
                ]
            )

            # This script appears to ignore --out-path and writes to exports/model_probs.csv.
            run_cmd(
                [
                    sys.executable,
                    str(PREDICT_SCRIPT),
                    "--test-features",
                    str(test_path),
                    "--model-dir",
                    str(model_dir),
                    "--out-path",
                    str(fold_model_probs_path),
                ]
            )

            if not GLOBAL_MODEL_PROBS_PATH.exists():
                raise FileNotFoundError(f"Missing predictor output: {GLOBAL_MODEL_PROBS_PATH}")

            global_model_probs = pd.read_csv(GLOBAL_MODEL_PROBS_PATH, low_memory=False)
            ensure_columns(global_model_probs, ["game_id", "Season", "Week"], "global model_probs.csv")

            global_model_probs["Season"] = pd.to_numeric(global_model_probs["Season"], errors="coerce")
            global_model_probs["Week"] = pd.to_numeric(global_model_probs["Week"], errors="coerce")

            week_model_probs = global_model_probs.loc[
                (global_model_probs["Season"] == season) & (global_model_probs["Week"] == week)
            ].copy()

            if week_model_probs.empty:
                raise ValueError(
                    f"No predictions found in {GLOBAL_MODEL_PROBS_PATH} for season={season}, week={week}"
                )

            save_csv(week_model_probs, fold_model_probs_path)
            print(f"Saved filtered week model probs: {len(week_model_probs):,} -> {fold_model_probs_path}")

            run_cmd(
                [
                    sys.executable,
                    str(BACKTEST_SCRIPT),
                    "--model-probs-path",
                    str(fold_model_probs_path),
                    "--summary-out-path",
                    str(summary_path),
                    "--bets-out-path",
                    str(bets_path),
                ]
            )

            if not summary_path.exists():
                raise FileNotFoundError(f"Missing expected backtest summary: {summary_path}")

            summary_df = pd.read_csv(summary_path, low_memory=False)
            summary_df["test_season"] = season
            summary_df["test_week"] = week
            summary_df["train_rows"] = len(train_df)
            summary_df["test_rows"] = len(test_df)
            combined_summary_rows.append(summary_df)

            if bets_path.exists():
                bets_df = pd.read_csv(bets_path, low_memory=False)
                if len(bets_df):
                    bets_df["test_season"] = season
                    bets_df["test_week"] = week
                    combined_bets.append(bets_df)

    if combined_summary_rows:
        all_summary = pd.concat(combined_summary_rows, ignore_index=True)
        all_summary_path = OUT_ROOT / "walkforward_weekly_summary_all.csv"
        all_summary.to_csv(all_summary_path, index=False)

        print("\nWALK-FORWARD WEEKLY COMPLETE")
        print(f"Saved combined summary: {all_summary_path}")

        print("\nTOP SUMMARY ROWS")
        print(
            all_summary.sort_values(
                ["roi", "bets"],
                ascending=[False, False],
                kind="stable",
            ).head(40).to_string(index=False)
        )
    else:
        print("\nNo summary rows were produced.")

    if combined_bets:
        all_bets = pd.concat(combined_bets, ignore_index=True)
        all_bets_path = OUT_ROOT / "walkforward_weekly_bets_all.csv"
        all_bets.to_csv(all_bets_path, index=False)
        print(f"\nSaved combined bets: {all_bets_path}")
        print(f"Total bet rows: {len(all_bets):,}")


if __name__ == "__main__":
    main()