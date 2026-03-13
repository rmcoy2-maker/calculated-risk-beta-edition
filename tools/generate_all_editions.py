import subprocess
import sys
import argparse

EDITIONS = [
    "tnf",
    "sunday_morning",
    "sunday_afternoon",
    "snf",
    "monday",
    "tuesday",
]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--week", required=True, type=int)

    args = parser.parse_args()

    for edition in EDITIONS:

        cmd = [
            sys.executable,
            "generate_report.py",
            "--week",
            str(args.week),
            "--edition",
            edition,
        ]

        print("Running:", " ".join(cmd))

        subprocess.run(cmd)


if __name__ == "__main__":
    main()