from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


THIS = Path(__file__).resolve()


def find_repo_root() -> Path:
    candidates = [THIS.parent] + list(THIS.parents)
    markers = [
        "streamlit_app.py",
        "00_Home.py",
        "00_Home_download.py",
        ".git",
    ]
    for p in candidates:
        if any((p / marker).exists() for marker in markers):
            return p
    return Path.cwd()


ROOT = find_repo_root()
TOOLS_DIR = ROOT / "tools"
EXPORTS_DIR = ROOT / "exports"
REPORTS_DIR = EXPORTS_DIR / "reports"
PRIMARY_CANONICAL_DIR = ROOT / "tools" / "exports" / "canonical"
FALLBACK_CANONICAL_DIR = ROOT / "exports" / "canonical"
GENERATOR = TOOLS_DIR / "generate_report.py"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if TOOLS_DIR.exists() and str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


st.set_page_config(
    page_title="Calculated Risk Reports Hub",
    page_icon="📑",
    layout="wide",
)

DEFAULT_SEASON = 2025
DEFAULT_WEEK = 1

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
        "timing": "Thu afternoon / kickoff",
    },
    "sunday_morning": {
        "name": "Sunday Morning – full slate preview",
        "timing": "Sun early morning",
    },
    "sunday_afternoon": {
        "name": "Sunday Afternoon – late slate preview",
        "timing": "Sun afternoon",
    },
    "snf": {
        "name": "Sunday Night Football",
        "timing": "Sun evening",
    },
    "monday": {
        "name": "Monday Edition",
        "timing": "Mon morning / pre-MNF wrap",
    },
    "tuesday": {
        "name": "Tuesday Wrap",
        "timing": "Tue wrap / look-ahead",
    },
}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_json_path(season: int, week: int, edition: str) -> Path:
    return REPORTS_DIR / f"{season}_week{week}_{edition}_3v1.json"


def expected_pdf_path(season: int, week: int, edition: str) -> Path:
    return REPORTS_DIR / f"{season}_week{week}_{edition}_3v1.pdf"


def canonical_file_status(season: int, week: int) -> pd.DataFrame:
    names = [
        "cr_games",
        "cr_edges",
        "cr_markets",
        "cr_scores",
        "cr_parlay_scores",
    ]
    rows = []
    for name in names:
        primary = PRIMARY_CANONICAL_DIR / f"{name}_{season}_w{week}.csv"
        fallback = FALLBACK_CANONICAL_DIR / f"{name}_{season}_w{week}.csv"
        chosen = primary if primary.exists() else fallback
        rows.append(
            {
                "Input": name,
                "Primary Path": str(primary),
                "Primary Exists": primary.exists(),
                "Fallback Path": str(fallback),
                "Fallback Exists": fallback.exists(),
                "Chosen Path": str(chosen) if chosen.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def guess_current_edition(now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
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


def run_generator(
    season: int,
    week: int,
    edition: str,
    asof: str,
) -> int:
    args = [
        sys.executable,
        str(GENERATOR),
        "--season",
        str(season),
        "--week",
        str(week),
        "--edition",
        edition,
        "--asof",
        asof,
    ]

    st.info(
        f"Python: {sys.executable}\n\n"
        f"Repo root: {ROOT}\n\n"
        f"Running: {' '.join(args)}\n\n"
        f"cwd={ROOT}"
    )

    try:
        completed = subprocess.run(
            args,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
    except Exception as e:
        st.error(f"Failed to start generator: {e}")
        return 1

    if completed.stdout:
        st.code(completed.stdout, language="text")
    if completed.stderr:
        st.error(completed.stderr)

    if completed.returncode == 0:
        st.success("Report generation completed successfully.")
    else:
        st.error(f"Report generation exited with code {completed.returncode}.")

    return completed.returncode


def artifact_row(path: Path) -> dict:
    stat = path.stat()
    return {
        "File": path.name,
        "Path": str(path),
        "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "Bytes": stat.st_size,
        "SHA256": sha256_of_file(path),
    }


def collect_artifacts(season: int, week: int, edition: str) -> List[Path]:
    files: List[Path] = []
    for p in [
        expected_json_path(season, week, edition),
        expected_pdf_path(season, week, edition),
    ]:
        if p.exists():
            files.append(p)
    return files


def list_report_pdfs_for_week(season: int, week: int) -> List[Path]:
    exact_prefix = f"{season}_week{week}_"
    legacy_prefix = f"week{week}_"

    files = [
        p for p in REPORTS_DIR.glob("*.pdf")
        if p.name.startswith(exact_prefix) or p.name.startswith(legacy_prefix)
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def build_status_table(season: int, week: int) -> pd.DataFrame:
    rows = []
    for edition in EDITION_ORDER:
        pdf_path = expected_pdf_path(season, week, edition)
        json_path = expected_json_path(season, week, edition)
        rows.append(
            {
                "Edition": edition,
                "Edition Name": EDITION_META[edition]["name"],
                "Timing": EDITION_META[edition]["timing"],
                "JSON": "✅" if json_path.exists() else "❌",
                "PDF": "✅" if pdf_path.exists() else "❌",
                "Expected JSON": json_path.name,
                "Expected PDF": pdf_path.name,
            }
        )
    return pd.DataFrame(rows)


def app() -> None:
    st.title("📑 Calculated Risk Reports Hub")

    st.caption(f"Repo root: {ROOT}")
    st.caption(f"Generator: {GENERATOR}")
    st.caption(f"Primary canonical dir: {PRIMARY_CANONICAL_DIR}")
    st.caption(f"Fallback canonical dir: {FALLBACK_CANONICAL_DIR}")
    st.caption(f"Reports dir: {REPORTS_DIR}")
    st.caption(f"Python: {sys.executable}")

    season = st.sidebar.number_input("Season", min_value=2020, max_value=2100, value=DEFAULT_SEASON, step=1)
    week = st.sidebar.number_input("Week", min_value=1, max_value=22, value=DEFAULT_WEEK, step=1)

    suggested = guess_current_edition()
    edition = st.sidebar.selectbox(
        "Edition",
        options=EDITION_ORDER,
        index=EDITION_ORDER.index(suggested),
        format_func=lambda k: f"{k} — {EDITION_META[k]['name']}",
    )

    asof_default = datetime.now().isoformat(timespec="minutes")
    asof_input = st.sidebar.text_input("As Of (YYYY-MM-DDTHH:MM)", value=asof_default)

    st.markdown("### Canonical input status")
    st.dataframe(canonical_file_status(int(season), int(week)), hide_index=True, use_container_width=True)

    st.markdown("### Expected output status")
    st.dataframe(build_status_table(int(season), int(week)), hide_index=True, use_container_width=True)

    st.markdown("### Generate single report")
    if st.button("Generate selected edition", type="primary"):
        if not GENERATOR.exists():
            st.error(f"Generator not found: {GENERATOR}")
        else:
            rc = run_generator(int(season), int(week), edition, asof_input)
            if rc == 0:
                st.rerun()

    st.markdown("### Selected artifact details")
    artifacts = collect_artifacts(int(season), int(week), edition)

    if not artifacts:
        st.info("No JSON/PDF found yet for the selected season/week/edition.")
    else:
        rows = [artifact_row(p) for p in artifacts]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        for p in artifacts:
            st.markdown(f"**{p.name}**")
            st.caption(str(p))
            st.caption(
                f"Modified: {datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Bytes: {p.stat().st_size} | "
                f"SHA256: {sha256_of_file(p)}"
            )

            if p.suffix.lower() == ".json":
                try:
                    payload = json.loads(p.read_text(encoding="utf-8"))
                    with st.expander("Preview JSON"):
                        st.json(payload)
                except Exception as e:
                    st.warning(f"Could not preview JSON: {e}")

            mime = "application/pdf" if p.suffix.lower() == ".pdf" else "application/json"
            st.download_button(
                label=f"Download {p.suffix.upper().replace('.', '')}",
                data=p.read_bytes(),
                file_name=p.name,
                mime=mime,
                key=f"download_{p.name}",
            )

    st.markdown("---")
    st.markdown("### Existing PDFs for selected season/week")

    pdfs = list_report_pdfs_for_week(int(season), int(week))
    if not pdfs:
        st.caption("No PDF reports found for this season/week.")
    else:
        for pdf in pdfs:
            st.markdown(f"**{pdf.name}**")
            st.caption(str(pdf))
            st.caption(
                f"Modified: {datetime.fromtimestamp(pdf.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Bytes: {pdf.stat().st_size} | "
                f"SHA256: {sha256_of_file(pdf)}"
            )
            st.download_button(
                label="Download PDF",
                data=pdf.read_bytes(),
                file_name=pdf.name,
                mime="application/pdf",
                key=f"pdf_{pdf.name}",
            )


app()