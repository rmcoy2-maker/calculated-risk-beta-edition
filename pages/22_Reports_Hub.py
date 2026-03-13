from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


# ------------------------------------------------------------------
# Root/path detection (cloud-safe)
# ------------------------------------------------------------------
THIS = Path(__file__).resolve()


def find_repo_root() -> Path:
    for p in [THIS.parent] + list(THIS.parents):
        if (p / "streamlit_app.py").exists():
            return p
    return Path.cwd()


ROOT = find_repo_root()
TOOLS_DIR = ROOT / "tools"
EXPORTS_DIR = ROOT / "exports"
REPORTS_DIR = EXPORTS_DIR / "reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if TOOLS_DIR.exists() and str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


# ------------------------------------------------------------------
# TEMP AUTH / ELIGIBILITY SHIMS
# ------------------------------------------------------------------
def require_eligibility(*args, **kwargs):
    return True


def current_user():
    return None


def login():
    return None


def logout():
    return None


def show_logout():
    return None


# ------------------------------------------------------------------
# Ensure only writable repo-local dirs are created
# ------------------------------------------------------------------
for p in (EXPORTS_DIR, REPORTS_DIR):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.warning(f"Could not create directory {p}: {e}")


# ------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Doc Odds Reports Hub (3v1 Schedule)",
    page_icon="📑",
    layout="wide",
)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DEFAULT_SEASON = 2025
DEFAULT_WEEK = 13

EDITION_ORDER = [
    "tnf",
    "sunday_morning",
    "sunday_afternoon",
    "snf",
    "monday",
    "tuesday",
]

EDITION_META: Dict[str, Dict[str, str]] = {
    "tnf": {
        "name": "TNF – Thursday Night Edition",
        "timing": "Thu (day-of) – afternoon + kickoff",
    },
    "sunday_morning": {
        "name": "Sunday Morning – full slate preview",
        "timing": "Sun – early morning > before 1pm window",
    },
    "sunday_afternoon": {
        "name": "Sunday Afternoon – grades + late slate preview",
        "timing": "Sun – 1pm–4:25pm windows rolling",
    },
    "snf": {
        "name": "Sunday Night Football – with Thur–Sun grades",
        "timing": "Sun – after 4pm window > SNF",
    },
    "monday": {
        "name": "Monday Morning Edition – with Thur–Sun grades",
        "timing": "Mon – morning / afternoon before MNF",
    },
    "tuesday": {
        "name": "Tuesday Wrap – full Thur–Mon grades & look-ahead",
        "timing": "Tue – day after MNF – week wrap + early lookahead",
    },
}


@dataclass
class EditionRow:
    key: str
    name: str
    timing: str
    pdf_file: str
    exists: bool


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def guess_current_edition(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now()

    dow = now.weekday()
    hour = now.hour

    if dow == 3:
        return "tnf"
    if dow == 4:
        return "sunday_afternoon"
    if dow == 5:
        return "sunday_morning"
    if dow == 6:
        if hour < 12:
            return "sunday_morning"
        elif hour < 17:
            return "sunday_afternoon"
        else:
            return "snf"
    if dow == 0:
        return "monday"
    if dow == 1:
        return "tuesday"

    return "sunday_morning"


def run_tool_command(args: List[str], description: str, cwd: Path | None = None) -> int:
    if cwd is None:
        cwd = TOOLS_DIR

    st.info(f"Running: `{' '.join(args)}`  \n(cwd={cwd})")

    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
    except Exception as e:
        st.error(f"{description} failed to start: {e}")
        return 1

    if completed.stdout:
        st.code(completed.stdout, language="bash")

    if completed.stderr:
        st.error(completed.stderr)

    if completed.returncode == 0:
        st.success(f"{description} completed successfully.")
    else:
        st.error(f"{description} exited with code {completed.returncode}.")

    return completed.returncode


def build_status_table(week: int) -> pd.DataFrame:
    rows: List[EditionRow] = []

    for key in EDITION_ORDER:
        meta = EDITION_META[key]
        pdf_name = f"week{week}_{key}_3v1.pdf"
        pdf_path = REPORTS_DIR / pdf_name

        rows.append(
            EditionRow(
                key=key,
                name=meta["name"],
                timing=meta["timing"],
                pdf_file=pdf_name,
                exists=pdf_path.exists(),
            )
        )

    return pd.DataFrame(
        [
            {
                "Edition Key": r.key,
                "Edition Name": r.name,
                "Timing Window": r.timing,
                "PDF File": r.pdf_file,
                "Status": "✅ Generated" if r.exists else "❌ Missing",
            }
            for r in rows
        ]
    )


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def app():
    st.title("Doc Odds Reports Hub (3v1 Schedule)")
    st.caption(f"Repo root: {ROOT}")
    st.caption(f"Reports dir: {REPORTS_DIR}")

    st.sidebar.title("Report Controls")

    season = st.sidebar.number_input(
        "Season",
        min_value=2010,
        max_value=2100,
        value=DEFAULT_SEASON,
        step=1,
    )

    week = st.sidebar.number_input(
        "Week",
        min_value=1,
        max_value=25,
        value=DEFAULT_WEEK,
        step=1,
    )

    st.sidebar.caption(
        "Season is mainly for context; current scripts use `--week` "
        "and infer season from your games_master."
    )

    st.sidebar.subheader("Batch Actions")

    run_full_pipeline_btn = st.sidebar.button(
        "Run FULL weekly pipeline\n(ETL + ECE + Playoffs + Reports)"
    )
    run_all_editions_btn = st.sidebar.button(
        "Generate ALL 3v1 editions for this week"
    )

    now = datetime.now()
    suggested_key = guess_current_edition(now)
    suggested_meta = EDITION_META[suggested_key]

    st.caption(
        f"Current slot: **{now:%A %Y-%m-%d %H:%M}** — suggested edition: "
        f"**{suggested_key}** ({suggested_meta['name']})."
    )

    st.markdown("### Run report for NOW / special edition")

    col_now, col_picker = st.columns([2, 2])

    with col_picker:
        edition_for_now = st.selectbox(
            "Edition to run",
            options=EDITION_ORDER,
            format_func=lambda k: f"{k} — {EDITION_META[k]['name']}",
            index=EDITION_ORDER.index(suggested_key),
        )

    with col_now:
        if st.button("Run report for NOW -> generate this edition"):
            script = TOOLS_DIR / "generate_report.py"
            if script.exists():
                run_tool_command(
                    [
                        sys.executable,
                        "generate_report.py",
                        "--week",
                        str(week),
                        "--edition",
                        edition_for_now,
                    ],
                    description=f"Generate single 3v1 edition ({edition_for_now})",
                )
            else:
                st.error(f"{script} not found. Expected at {script}.")

    st.markdown("---")

    st.markdown("### Week-by-Week 3v1 Edition Schedule & File Status")
    df_status = build_status_table(int(week))
    st.dataframe(df_status, hide_index=True, width="stretch")

    st.markdown("#### Run an individual edition")

    col_ed, col_btn = st.columns([3, 1])

    with col_ed:
        edition_choice = st.selectbox(
            "Choose edition to generate",
            options=EDITION_ORDER,
            format_func=lambda k: f"{k} — {EDITION_META[k]['name']}",
        )

    with col_btn:
        if st.button("Generate selected edition"):
            script = TOOLS_DIR / "generate_report.py"
            if script.exists():
                run_tool_command(
                    [
                        sys.executable,
                        "generate_report.py",
                        "--week",
                        str(week),
                        "--edition",
                        edition_choice,
                    ],
                    description=f"Generate 3v1 edition ({edition_choice})",
                )
            else:
                st.error(f"{script} not found. Expected at {script}.")

    st.markdown("---")
    st.caption(
        "This hub reads/writes PDFs under `exports/reports/`. "
        "Edition keys must match those expected by `tools/generate_report.py` "
        "and `tools/generate_all_editions.py`."
    )

    if run_full_pipeline_btn:
        scripts_to_try = [
            ("run_week_update.py", "Run weekly ETL/update (games, odds, ECE)"),
            ("run_playoffs_update.py", "Run playoffs update (if applicable)"),
            ("generate_all_editions.py", "Generate ALL 3v1 editions"),
        ]

        for script_name, label in scripts_to_try:
            script_path = TOOLS_DIR / script_name
            if not script_path.exists():
                st.info(f"Skipping {script_name} (not found in tools).")
                continue

            if script_name.startswith("generate_all_editions"):
                args = [
                    sys.executable,
                    script_name,
                    "--week",
                    str(week),
                ]
            else:
                args = [
                    sys.executable,
                    script_name,
                    "--season",
                    str(season),
                    "--week",
                    str(week),
                ]

            run_tool_command(args, description=label)

    if run_all_editions_btn:
        script = TOOLS_DIR / "generate_all_editions.py"
        if script.exists():
            run_tool_command(
                [
                    sys.executable,
                    "generate_all_editions.py",
                    "--week",
                    str(week),
                ],
                description="Generate ALL 3v1 editions",
            )
        else:
            st.error(f"{script} not found. Expected at {script}.")


app()