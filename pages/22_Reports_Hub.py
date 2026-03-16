from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


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


for p in (EXPORTS_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Doc Odds Reports Hub (3v1 Schedule)",
    page_icon="📑",
    layout="wide",
)


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


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    command: List[str]
    cwd: Path


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
        if hour < 17:
            return "sunday_afternoon"
        return "snf"
    if dow == 0:
        return "monday"
    if dow == 1:
        return "tuesday"

    return "sunday_morning"


def module_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def script_exists(name: str) -> bool:
    return (TOOLS_DIR / name).exists()


def run_tool_command(args: List[str], description: str, cwd: Path | None = None) -> CommandResult:
    if cwd is None:
        cwd = TOOLS_DIR

    st.info(
        f"Python: `{sys.executable}`  \n"
        f"Running: `{' '.join(args)}`  \n"
        f"(cwd={cwd})"
    )

    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
    except Exception as e:
        st.error(f"{description} failed to start: {e}")
        return CommandResult(1, "", str(e), args, cwd)

    if completed.stdout:
        st.code(completed.stdout)

    if completed.stderr:
        st.error(completed.stderr)

    if completed.returncode == 0:
        st.success(f"{description} completed successfully.")
    else:
        if "No module named 'reportlab'" in (completed.stderr or ""):
            st.warning("The report generator needs the `reportlab` package. Use the install button in the Environment Check section, then rerun the report.")
        st.error(f"{description} exited with code {completed.returncode}.")

    return CommandResult(completed.returncode, completed.stdout, completed.stderr, args, cwd)


def build_status_table(week: int) -> pd.DataFrame:
    rows: List[EditionRow] = []

    for key in EDITION_ORDER:
        pattern = f"week{week}_{key}_3v1"
        exists = any(
            p.name.startswith(pattern) and p.suffix.lower() == ".pdf"
            for p in REPORTS_DIR.glob("*.pdf")
        )
        rows.append(
            EditionRow(
                key=key,
                name=EDITION_META[key]["name"],
                timing=EDITION_META[key]["timing"],
                pdf_file=f"{pattern}_YYYYMMDD_HHMM.pdf",
                exists=exists,
            )
        )

    return pd.DataFrame(
        [
            {
                "Edition Key": r.key,
                "Edition Name": r.name,
                "Timing Window": r.timing,
                "Expected File Pattern": r.pdf_file,
                "Status": "✅ Generated" if r.exists else "❌ Missing",
            }
            for r in rows
        ]
    )


def list_report_files(week: int) -> list[Path]:
    prefix = f"week{week}_"
    files = [p for p in REPORTS_DIR.glob("*.pdf") if p.name.startswith(prefix)]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def week_options() -> list[int]:
    return list(range(1, 26))


def season_options() -> list[int]:
    return list(range(2020, 2031))


def environment_table() -> pd.DataFrame:
    rows = [
        {"Check": "Python executable", "Value": sys.executable, "Status": "OK"},
        {"Check": "Repo root", "Value": str(ROOT), "Status": "OK" if ROOT.exists() else "Missing"},
        {"Check": "tools/generate_report.py", "Value": str(TOOLS_DIR / "generate_report.py"), "Status": "OK" if script_exists("generate_report.py") else "Missing"},
        {"Check": "tools/generate_all_editions.py", "Value": str(TOOLS_DIR / "generate_all_editions.py"), "Status": "OK" if script_exists("generate_all_editions.py") else "Missing"},
        {"Check": "reportlab installed", "Value": "reportlab", "Status": "OK" if module_installed("reportlab") else "Missing"},
        {"Check": "exports/reports", "Value": str(REPORTS_DIR), "Status": "OK" if REPORTS_DIR.exists() else "Missing"},
    ]
    return pd.DataFrame(rows)


def latest_report_table() -> pd.DataFrame:
    pdfs = sorted(REPORTS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for p in pdfs[:25]:
        rows.append(
            {
                "File": p.name,
                "Modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "Size KB": round(p.stat().st_size / 1024.0, 1),
            }
        )
    return pd.DataFrame(rows)


def install_reportlab() -> None:
    run_tool_command(
        [sys.executable, "-m", "pip", "install", "reportlab"],
        description="Install reportlab",
        cwd=ROOT,
    )


def app() -> None:
    st.title("Doc Odds Reports Hub (3v1 Schedule)")
    st.caption(f"Repo root: {ROOT}")
    st.caption(f"Reports dir: {REPORTS_DIR}")
    st.caption(f"App Python: {sys.executable}")

    st.sidebar.title("Report Controls")

    season = st.sidebar.selectbox("Season", options=season_options(), index=season_options().index(DEFAULT_SEASON) if DEFAULT_SEASON in season_options() else 0)
    week = st.sidebar.selectbox("Week", options=week_options(), index=week_options().index(DEFAULT_WEEK) if DEFAULT_WEEK in week_options() else 0)

    st.sidebar.subheader("Batch Actions")
    run_all_editions_btn = st.sidebar.button("Generate ALL 3v1 editions for this week")

    now = datetime.now()
    suggested_key = guess_current_edition(now)
    suggested_meta = EDITION_META[suggested_key]

    st.caption(
        f"Current slot: **{now:%A %Y-%m-%d %H:%M}** — suggested edition: **{suggested_key}** ({suggested_meta['name']})."
    )

    with st.expander("Environment Check", expanded=True):
        env_df = environment_table()
        st.dataframe(env_df, hide_index=True, use_container_width=True)
        missing_reportlab = not module_installed("reportlab")
        if missing_reportlab:
            st.error("reportlab is not installed, so PDF generation will fail until it is installed.")
            if st.button("Install reportlab now"):
                install_reportlab()
                st.rerun()
        else:
            st.success("PDF dependency check passed.")

    st.markdown("### Run report for NOW / special edition")
    col_auto, col_picker = st.columns([2, 2])

    with col_picker:
        edition_for_now = st.selectbox(
            "Edition to run",
            options=EDITION_ORDER,
            format_func=lambda k: f"{k} — {EDITION_META[k]['name']}",
            index=EDITION_ORDER.index(suggested_key),
        )

    with col_auto:
        if st.button("Generate report as of NOW"):
            script = TOOLS_DIR / "generate_report.py"
            asof_now = datetime.now().isoformat(timespec="minutes")
            auto_edition = guess_current_edition(datetime.now())
            run_tool_command(
                [
                    sys.executable,
                    str(script),
                    "--season",
                    str(season),
                    "--week",
                    str(week),
                    "--edition",
                    auto_edition,
                    "--asof",
                    asof_now,
                ],
                description=f"Generate report for NOW ({auto_edition})",
            )

    if st.button("Generate selected edition"):
        script = TOOLS_DIR / "generate_report.py"
        asof_now = datetime.now().isoformat(timespec="minutes")
        run_tool_command(
            [
                sys.executable,
                str(script),
                "--season",
                str(season),
                "--week",
                str(week),
                "--edition",
                edition_for_now,
                "--asof",
                asof_now,
            ],
            description=f"Generate selected 3v1 edition ({edition_for_now})",
        )

    quick_cols = st.columns(3)
    if quick_cols[0].button("Generate Monday Edition"):
        script = TOOLS_DIR / "generate_report.py"
        run_tool_command(
            [sys.executable, str(script), "--season", str(season), "--week", str(week), "--edition", "monday", "--asof", datetime.now().isoformat(timespec="minutes")],
            description="Generate Monday edition",
        )
    if quick_cols[1].button("Generate Tuesday Wrap"):
        script = TOOLS_DIR / "generate_report.py"
        run_tool_command(
            [sys.executable, str(script), "--season", str(season), "--week", str(week), "--edition", "tuesday", "--asof", datetime.now().isoformat(timespec="minutes")],
            description="Generate Tuesday wrap",
        )
    if quick_cols[2].button("Open reports folder summary"):
        latest = latest_report_table()
        if latest.empty:
            st.info("No PDF reports found yet in exports/reports.")
        else:
            st.dataframe(latest, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Week-by-Week 3v1 Edition Schedule & File Status")
    df_status = build_status_table(int(week))
    st.dataframe(df_status, hide_index=True, use_container_width=True)

    st.markdown("### Existing generated reports")
    files = list_report_files(int(week))
    if not files:
        st.caption("No generated reports found yet for this week.")
    else:
        for pdf in files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(pdf.name)
            with col2:
                st.download_button(
                    label="Download",
                    data=pdf.read_bytes(),
                    file_name=pdf.name,
                    mime="application/pdf",
                    key=f"dl_{pdf.name}",
                )

    if run_all_editions_btn:
        script = TOOLS_DIR / "generate_all_editions.py"
        asof_now = datetime.now().isoformat(timespec="minutes")
        run_tool_command(
            [
                sys.executable,
                str(script),
                "--season",
                str(season),
                "--week",
                str(week),
                "--asof",
                asof_now,
            ],
            description="Generate ALL 3v1 editions",
        )


app()
