from __future__ import annotations

import argparse
from pathlib import Path

from report_data import EDITION_ORDER
from generate_report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate every Fort Knox / 3v1 edition for a week.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    for edition in EDITION_ORDER:
        artifacts = generate_report(args.season, args.week, edition, out_dir)
        print(f"[OK] {edition}: {artifacts.pdf_path.name}")


if __name__ == "__main__":
    main()
