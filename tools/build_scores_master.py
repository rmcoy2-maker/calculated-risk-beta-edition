from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "raw" / "2017-2025_scores.csv"
OUTPUT_FILE = PROJECT_ROOT / "exports" / "scores_master.csv"


def normalize_week(w):
    w = str(w).strip().lower()

    if "preseason" in w:
        return None
    if "hall" in w:
        return None
    if "wild" in w:
        return 19
    if "divisional" in w:
        return 20
    if "conference" in w:
        return 21
    if "super bowl" in w:
        return 22

    digits = "".join(ch for ch in w if ch.isdigit())
    if digits:
        return int(digits)

    return None


def main():
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["week_num"] = df["Week"].apply(normalize_week)
    df = df[df["week_num"].notna()].copy()
    df["week_num"] = df["week_num"].astype(int)

    df["home_score"] = pd.to_numeric(df["HomeScore"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["AwayScore"], errors="coerce")

    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()

    df["margin"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(int)
    df["away_win"] = (df["margin"] < 0).astype(int)

    df["game_id"] = (
        df["Season"].astype(int).astype(str)
        + "_"
        + df["week_num"].astype(str)
        + "_"
        + df["AwayTeam"].astype(str)
        + "_"
        + df["HomeTeam"].astype(str)
    )

    out = df[
        [
            "game_id",
            "Season",
            "week_num",
            "Date",
            "AwayTeam",
            "HomeTeam",
            "home_score",
            "away_score",
            "margin",
            "total_points",
            "home_win",
            "away_win",
            "PostSeason",
        ]
    ].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(out):,}")
    print(f"Games: {out['game_id'].nunique():,}")


if __name__ == "__main__":
    main()