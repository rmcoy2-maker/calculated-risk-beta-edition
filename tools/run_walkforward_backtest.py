from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "exports"
WF_DIR = EXPORTS_DIR / "walkforward"
WF_DIR.mkdir(parents=True, exist_ok=True)

PREGAME_FEATURES_PATH = EXPORTS_DIR / "pregame_features.csv"
MARKET_PRIORS_PATH = EXPORTS_DIR / "market_priors.csv"
MODEL_PROBS_PATH = EXPORTS_DIR / "model_probs.csv"
BACKTEST_SUMMARY_PATH = EXPORTS_DIR / "backtest_summary_moneyline.csv"
BACKTEST_BETS_PATH = EXPORTS_DIR / "backtest_bets_moneyline.csv"

TRAIN_SCRIPT = PROJECT_ROOT / "tools" / "train_pregame_models_market.py"
SCORE_SCRIPT = PROJECT_ROOT / "tools" / "build_model_probs_market.py"
BACKTEST_SCRIPT = PROJECT_ROOT / "tools" / "backtest_edges.py"

BACKUP_DIR = WF_DIR / "_original_backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Expanding-window folds
FOLDS = [
    {"train_end_season": 2019, "test_season": 2020},
    {"train_end_season": 2020, "test_season": 2021},
    {"train_end_season": 2021, "test_season": 2022},
    {"train_end_season": 2022, "test_season": 2023},
    {"train_end_season": 2023, "test_season": 2024},
    {"train_end_season": 2024, "test_season": 2025},
]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\nRUNNING:", " ".join(str(x) for x in cmd))
    completed = subprocess.run(cmd, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(map(str, cmd))}")


def backup_file(src: Path, backup_name: str) -> Path:
    backup_path = BACKUP_DIR / backup_name
    shutil.copy2(src, backup_path)
    return backup_path


def restore_file(backup_path: Path, dst: Path) -> None:
    shutil.copy2(backup_path, dst)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading features: {PREGAME_FEATURES_PATH}")
    pf = pd.read_csv(PREGAME_FEATURES_PATH, low_memory=False)

    print(f"Loading priors: {MARKET_PRIORS_PATH}")
    mp = pd.read_csv(MARKET_PRIORS_PATH, low_memory=False)

    if "game_id" not in pf.columns:
        raise ValueError("pregame_features.csv missing game_id")
    if "game_id" not in mp.columns:
        raise ValueError("market_priors.csv missing game_id")

    if "Season" not in pf.columns and "Season" not in mp.columns:
        raise ValueError("Neither pregame_features.csv nor market_priors.csv contains Season")

    if "Season" in pf.columns:
        pf["Season"] = pd.to_numeric(pf["Season"], errors="coerce").astype("Int64")
    if "Season" in mp.columns:
        mp["Season"] = pd.to_numeric(mp["Season"], errors="coerce").astype("Int64")

    return pf, mp


def build_fold_inputs(
    pf: pd.DataFrame,
    mp: pd.DataFrame,
    train_end_season: int,
    test_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    We keep all rows through test_season so the existing training script can
    internally use the max season as holdout/test. That matches its current behavior.
    """
    if "Season" not in pf.columns and "Season" not in mp.columns:
        raise ValueError("Need Season in at least one source")

    if "Season" in pf.columns:
        pf_fold = pf[pf["Season"] <= test_season].copy()
        allowed_game_ids = set(pf_fold["game_id"].astype(str))
        mp_fold = mp[mp["game_id"].astype(str).isin(allowed_game_ids)].copy()
    else:
        mp_fold = mp[mp["Season"] <= test_season].copy()
        allowed_game_ids = set(mp_fold["game_id"].astype(str))
        pf_fold = pf[pf["game_id"].astype(str).isin(allowed_game_ids)].copy()

    # Sanity: ensure test season exists
    test_rows = 0
    if "Season" in pf_fold.columns:
        test_rows = int((pf_fold["Season"] == test_season).sum())
    elif "Season" in mp_fold.columns:
        test_rows = int((mp_fold["Season"] == test_season).sum())

    if test_rows == 0:
        raise ValueError(f"No rows found for test_season={test_season}")

    # Sanity: ensure there are earlier seasons for training
    earlier_rows = 0
    if "Season" in pf_fold.columns:
        earlier_rows = int((pf_fold["Season"] <= train_end_season).sum())
    elif "Season" in mp_fold.columns:
        earlier_rows = int((mp_fold["Season"] <= train_end_season).sum())

    if earlier_rows == 0:
        raise ValueError(f"No rows found for training through season {train_end_season}")

    return pf_fold, mp_fold


def copy_outputs_to_fold_dir(fold_dir: Path, test_season: int) -> pd.DataFrame:
    if not MODEL_PROBS_PATH.exists():
        raise FileNotFoundError(f"Missing model_probs output: {MODEL_PROBS_PATH}")

    model_probs = pd.read_csv(MODEL_PROBS_PATH, low_memory=False)
    if "Season" not in model_probs.columns:
        raise ValueError("model_probs.csv missing Season")
    model_probs["Season"] = pd.to_numeric(model_probs["Season"], errors="coerce").astype("Int64")

    model_probs_test = model_probs[model_probs["Season"] == test_season].copy()
    if model_probs_test.empty:
        raise ValueError(f"No model_probs rows for test_season={test_season}")

    fold_model_probs_path = fold_dir / "model_probs_test_only.csv"
    model_probs_test.to_csv(fold_model_probs_path, index=False)

    print(f"Saved test-only model_probs: {fold_model_probs_path}")
    print(f"Rows: {len(model_probs_test):,}")

    # Overwrite default model_probs.csv with test-season-only rows,
    # then run backtest_edges.py so it backtests only that season.
    model_probs_test.to_csv(MODEL_PROBS_PATH, index=False)

    run_cmd([sys.executable, str(BACKTEST_SCRIPT)], PROJECT_ROOT)

    if not BACKTEST_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing expected backtest summary: {BACKTEST_SUMMARY_PATH}")

    fold_summary_path = fold_dir / "backtest_summary_moneyline.csv"
    fold_bets_path = fold_dir / "backtest_bets_moneyline.csv"

    shutil.copy2(BACKTEST_SUMMARY_PATH, fold_summary_path)

    if BACKTEST_BETS_PATH.exists():
        bets = pd.read_csv(BACKTEST_BETS_PATH, low_memory=False)
        if "Season" in bets.columns:
            bets["Season"] = pd.to_numeric(bets["Season"], errors="coerce").astype("Int64")
            bets = bets[bets["Season"] == test_season].copy()
        bets.to_csv(fold_bets_path, index=False)
        print(f"Saved test-only bets: {fold_bets_path}")
        print(f"Rows: {len(bets):,}")

    print(f"Saved fold summary: {fold_summary_path}")
    return pd.read_csv(fold_summary_path, low_memory=False)


def main() -> None:
    pf, mp = load_sources()

    # Backup originals once
    pf_backup = backup_file(PREGAME_FEATURES_PATH, "pregame_features.csv.bak")
    mp_backup = backup_file(MARKET_PRIORS_PATH, "market_priors.csv.bak")

    model_probs_backup = None
    if MODEL_PROBS_PATH.exists():
        model_probs_backup = backup_file(MODEL_PROBS_PATH, "model_probs.csv.bak")

    summary_backup = None
    if BACKTEST_SUMMARY_PATH.exists():
        summary_backup = backup_file(BACKTEST_SUMMARY_PATH, "backtest_summary_moneyline.csv.bak")

    bets_backup = None
    if BACKTEST_BETS_PATH.exists():
        bets_backup = backup_file(BACKTEST_BETS_PATH, "backtest_bets_moneyline.csv.bak")

    summary_rows = []

    try:
        for fold in FOLDS:
            train_end = fold["train_end_season"]
            test_season = fold["test_season"]

            print("\n" + "=" * 80)
            print(f"WALK-FORWARD FOLD: train <= {train_end}, test = {test_season}")
            print("=" * 80)

            fold_dir = WF_DIR / f"train_to_{train_end}_test_{test_season}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            pf_fold, mp_fold = build_fold_inputs(pf, mp, train_end, test_season)

            # Save fold copies for inspection
            pf_fold_path = fold_dir / "pregame_features_fold.csv"
            mp_fold_path = fold_dir / "market_priors_fold.csv"
            pf_fold.to_csv(pf_fold_path, index=False)
            mp_fold.to_csv(mp_fold_path, index=False)

            print(f"Saved fold features: {pf_fold_path} ({len(pf_fold):,} rows)")
            print(f"Saved fold priors:   {mp_fold_path} ({len(mp_fold):,} rows)")

            # Overwrite default inputs so the existing scripts use the fold
            pf_fold.to_csv(PREGAME_FEATURES_PATH, index=False)
            mp_fold.to_csv(MARKET_PRIORS_PATH, index=False)

            # Run existing scripts unchanged
            run_cmd([sys.executable, str(TRAIN_SCRIPT)], PROJECT_ROOT)
            run_cmd([sys.executable, str(SCORE_SCRIPT)], PROJECT_ROOT)

            fold_summary = copy_outputs_to_fold_dir(fold_dir, test_season)
            fold_summary["train_end_season"] = train_end
            fold_summary["test_season"] = test_season
            summary_rows.append(fold_summary)

        all_summary = pd.concat(summary_rows, ignore_index=True)
        all_summary_path = WF_DIR / "walkforward_summary_all_folds.csv"
        all_summary.to_csv(all_summary_path, index=False)

        print("\nWALK-FORWARD COMPLETE")
        print(f"Saved combined summary: {all_summary_path}")
        print("\nCombined results:")
        print(all_summary.to_string(index=False))

    finally:
        # Restore originals no matter what
        restore_file(pf_backup, PREGAME_FEATURES_PATH)
        restore_file(mp_backup, MARKET_PRIORS_PATH)

        if model_probs_backup is not None:
            restore_file(model_probs_backup, MODEL_PROBS_PATH)
        if summary_backup is not None:
            restore_file(summary_backup, BACKTEST_SUMMARY_PATH)
        if bets_backup is not None:
            restore_file(bets_backup, BACKTEST_BETS_PATH)

        print("\nOriginal export files restored.")


if __name__ == "__main__":
    main()
