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

    parser = argparse.ArgumentParser(description="Generate all 3v1 editions for a week")

    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--reports-dir", type=str, default=None)
    parser.add_argument("--write-back-labels", action="store_true")
    parser.add_argument("--asof", type=str, default=None)

    args = parser.parse_args()

    tools_dir = Path(__file__).resolve().parent
    script = tools_dir / "generate_report.py"

    if not script.exists():
        raise SystemExit(f"Missing generate_report.py at {script}")

    asof = args.asof or datetime.now().isoformat(timespec="minutes")

    failures = []

    for edition in EDITION_ORDER:

        cmd = [
            sys.executable,
            str(script),
            "--season",
            str(args.season),
            "--week",
            str(args.week),
            "--edition",
            edition,
            "--asof",
            asof,
        ]

        if args.reports_dir:
            cmd += ["--reports-dir", args.reports_dir]

        if args.write_back_labels:
            cmd.append("--write-back-labels")

        print(f"[RUN] season={args.season} week={args.week} edition={edition}")

        result = subprocess.run(
            cmd,
            cwd=str(tools_dir),
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            failures.append(edition)

    if failures:
        print("[WARN] Some editions failed:")
        for f in failures:
            print(f" - {f}")
        raise SystemExit(1)

    print("[OK] All editions generated successfully.")


if __name__ == "__main__":
    main()
