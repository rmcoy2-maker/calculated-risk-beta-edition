from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

EDITIONS = [
    "tnf",
    "sunday_morning",
    "sunday_afternoon",
    "snf",
    "monday",
    "tuesday",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all 3v1 report editions for a week")
    parser.add_argument("--week", required=True, type=int, help="NFL week number")
    parser.add_argument(
        "--asof",
        required=False,
        default=None,
        type=str,
        help="Optional shared as-of timestamp for all editions",
    )

    args = parser.parse_args()
    asof = args.asof or datetime.now().isoformat(timespec="minutes")

    for edition in EDITIONS:
        cmd = [
            sys.executable,
            str(TOOLS / "generate_report.py"),
            "--week",
            str(args.week),
            "--edition",
            edition,
            "--asof",
            asof,
        ]

        print("Running:", " ".join(cmd))
        completed = subprocess.run(cmd, capture_output=True, text=True)

        if completed.stdout:
            print(completed.stdout.strip())

        if completed.stderr:
            print(completed.stderr.strip())

        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()