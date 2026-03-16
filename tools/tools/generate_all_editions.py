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
    parser = argparse.ArgumentParser(description="Generate all 3v1 editions for one season/week.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--asof", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default=None)
    parser.add_argument("--write-back-labels", action="store_true")
    args = parser.parse_args()

    tools_dir = Path(__file__).resolve().parent
    script = tools_dir / "generate_report.py"

    if not script.exists():
        raise SystemExit(f"Missing generate_report.py at {script}")

    asof = args.asof or datetime.now().isoformat(timespec="minutes")

    failed = []
    for edition in EDITION_ORDER:
        cmd = [
            sys.executable,
            str(script),
            "--season", str(args.season),
            "--week", str(args.week),
            "--edition", edition,
        ]

        if args.reports_dir:
            cmd += ["--reports-dir", args.reports_dir]

        if args.write_back_labels:
            cmd.append("--write-back-labels")

        # only include --asof if generate_report.py supports it
        cmd += ["--asof", asof]

        print(f"[RUN] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(tools_dir), capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            failed.append(edition)

    if failed:
        raise SystemExit(f"Failed editions: {', '.join(failed)}")

    print("[OK] All editions generated successfully.")

if __name__ == "__main__":
    main()