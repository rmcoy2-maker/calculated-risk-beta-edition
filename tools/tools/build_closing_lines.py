from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LINES_PATH = PROJECT_ROOT / "exports" / "lines_master.csv"
OUT_PATH = PROJECT_ROOT / "exports" / "closing_lines.csv"


def main() -> None:
    print("Loading lines_master...")
    df = pd.read_csv(LINES_PATH, low_memory=False)

    required = [
        "game_id",
        "requested_snapshot",
        "commence_time",
        "book_key",
        "market",
        "side",
        "odds",
        "point",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"lines_master.csv missing required columns: {missing}")

    df["requested_snapshot"] = pd.to_datetime(df["requested_snapshot"], errors="coerce", utc=True)
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)

    # Keep only snapshots at or before kickoff
    df = df[df["requested_snapshot"] <= df["commence_time"]].copy()

    # Sort so the latest pre-kickoff line is last
    df = df.sort_values(
        ["game_id", "book_key", "market", "side", "requested_snapshot"],
        kind="stable",
    )

    closing = (
        df.groupby(["game_id", "book_key", "market", "side"], dropna=False, as_index=False)
          .tail(1)
          .copy()
    )

    closing = closing.sort_values(
        ["game_id", "book_key", "market", "side"],
        kind="stable",
    ).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    closing.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(closing):,}")
    print(closing.head(20).to_string(index=False))


if __name__ == "__main__":
    main()