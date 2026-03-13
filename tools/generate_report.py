from datetime import datetime
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
REPORTS = EXPORTS / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)


def build_report_file(week: int, edition: str, asof: str | None = None):

    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M")

    filename = f"week{week}_{edition}_3v1_{stamp}.pdf"
    out = REPORTS / filename

    content = f"""
CALCULATED RISK — 3v1 REPORT

Edition: {edition}
Week: {week}

As Of: {asof}
Generated: {now}

Placeholder generator.

Future sections:
- Top edges
- Heatmap board
- Game scripts
- Parlay portfolio
"""

    out.write_text(content)

    return out


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--edition", required=True)
    parser.add_argument("--asof", default=None)

    args = parser.parse_args()

    path = build_report_file(
        week=args.week,
        edition=args.edition,
        asof=args.asof,
    )

    print("Report created:")
    print(path)


if __name__ == "__main__":
    main()