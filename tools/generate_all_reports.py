from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EDITION_ORDER = [
    "tnf",
    "sunday_morning",
    "sunday_afternoon",
    "snf",
    "monday",
    "tuesday",
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate archive range of 3v1 reports.")
    parser.add_argument("--season-start", type=int, required=True)
    parser.add_argument("--season-end", type=int, required=True)
    parser.add_argument("--week-start", type=int, required=True)
    parser.add_argument("--week-end", type=int, required=True)
    parser.add_argument("--asof", type=str, default=None)
    args = parser.parse_args()

    tools_dir = Path(__file__).resolve().parent
    script = tools_dir / "generate_report.py"
    asof = args.asof or datetime.now().isoformat(timespec="minutes")

    failures = []

    for season in range(args.season_start, args.season_end + 1):
        for week in range(args.week_start, args.week_end + 1):
            for edition in EDITION_ORDER:
                cmd = [
                    sys.executable,
                    str(script),
                    "--season", str(season),
                    "--week", str(week),
                    "--edition", edition,
                    "--asof", asof,
                ]
                print(f"[RUN] season={season} week={week} edition={edition}")
                result = subprocess.run(cmd, cwd=str(tools_dir))
                if result.returncode != 0:
                    failures.append((season, week, edition))

    if failures:
        print("[WARN] Some archive generations failed:")
        for season, week, edition in failures:
            print(f" - season={season} week={week} edition={edition}")
        raise SystemExit(1)

    print("[OK] Archive generation complete.")

if __name__ == "__main__":
    main()