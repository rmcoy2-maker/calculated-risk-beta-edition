from __future__ import annotations

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


def run_tool_command(args: List[str], description: str, cwd: Path | None = None) -> int:
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
        return 1

    if completed.stdout:
        st.code(completed.stdout)

    if completed.stderr:
        st.error(completed.stderr)

    if completed.returncode == 0:
        st.success(f"{description} completed successfully.")
    else:
        st.error(f"{description} exited with code {completed.returncode}.")

    return completed.returncode


def build_status_table(season: int, week: int) -> pd.DataFrame:
    rows: List[EditionRow] = []

    for key in EDITION_ORDER:
        pattern = f"{season}_week{week}_{key}_3v1"

        exists = any(
            p.name.startswith(pattern) and p.suffix.lower() == ".pdf"
            for p in REPORTS_DIR.glob("*.pdf")
        )

        rows.append(
            EditionRow(
                key=key,
                name=EDITION_META[key]["name"],
                timing=EDITION_META[key]["timing"],
                pdf_file=f"{pattern}.pdf",
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


def list_report_files(season: int, week: int) -> list[Path]:
    prefixes = [
        f"{season}_week{week}_",
        f"week{week}_",
    ]

    files = [
        p for p in REPORTS_DIR.glob("*.pdf")
        if any(p.name.startswith(prefix) for prefix in prefixes)
    ]

    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def list_all_report_files() -> list[Path]:
    return sorted(
        REPORTS_DIR.glob("*.pdf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def parse_report_filename(path: Path) -> dict:
    stem = path.stem

    parsed = {
        "file": path.name,
        "season": None,
        "week": None,
        "edition": "",
        "edition_name": "",
        "modified": datetime.fromtimestamp(path.stat().st_mtime),
        "path": path,
    }

    parts = stem.split("_")

    if len(parts) >= 4 and parts[0].isdigit() and parts[1].startswith("week"):
        try:
            parsed["season"] = int(parts[0])
            parsed["week"] = int(parts[1].replace("week", ""))
            parsed["edition"] = "_".join(parts[2:-1])
        except Exception:
            pass
    elif len(parts) >= 3 and parts[0].startswith("week"):
        try:
            parsed["week"] = int(parts[0].replace("week", ""))
            parsed["edition"] = "_".join(parts[1:-1])
        except Exception:
            pass

    edition_key = parsed["edition"]
    if edition_key in EDITION_META:
        parsed["edition_name"] = EDITION_META[edition_key]["name"]
    else:
        parsed["edition_name"] = edition_key or "Unknown"

    return parsed


def build_archive_table() -> pd.DataFrame:
    rows = []
    for pdf in list_all_report_files():
        info = parse_report_filename(pdf)
        rows.append(
            {
                "Season": info["season"],
                "Week": info["week"],
                "Edition": info["edition"],
                "Edition Name": info["edition_name"],
                "File": info["file"],
                "Modified": info["modified"].strftime("%Y-%m-%d %H:%M"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Season", "Week", "Edition", "Edition Name", "File", "Modified"])

    return pd.DataFrame(rows)


def app() -> None:
    st.title("Doc Odds Reports Hub (3v1 Schedule)")

    st.caption(f"Repo root: {ROOT}")
    st.caption(f"Reports dir: {REPORTS_DIR}")
    st.caption(f"App Python: {sys.executable}")

    archive_df = build_archive_table()

    season_values = sorted({
        int(x) for x in archive_df["Season"].dropna().tolist()
        if str(x) != "nan"
    }) if not archive_df.empty else []

    if DEFAULT_SEASON not in season_values:
        season_values.append(DEFAULT_SEASON)

    season_values = sorted(set(season_values))

    if not season_values:
        season_values = [DEFAULT_SEASON]

    week_values = list(range(1, 19))
    if not archive_df.empty:
        week_values = sorted(set(week_values) | {
            int(x) for x in archive_df["Week"].dropna().tolist()
            if str(x) != "nan"
        })

    season = st.sidebar.selectbox(
        "Season",
        options=season_values,
        index=season_values.index(DEFAULT_SEASON) if DEFAULT_SEASON in season_values else 0,
    )

    week = st.sidebar.selectbox(
        "Week",
        options=week_values,
        index=week_values.index(DEFAULT_WEEK) if DEFAULT_WEEK in week_values else 0,
    )

    now = datetime.now()
    suggested_key = guess_current_edition(now)
    suggested_meta = EDITION_META[suggested_key]

    st.caption(
        f"Current slot: **{now:%A %Y-%m-%d %H:%M}** — suggested edition: "
        f"**{suggested_key}** ({suggested_meta['name']})."
    )

    st.markdown("### Run report for NOW / special edition")

    col_auto, col_picker = st.columns([2, 2])

    with col_picker:
        edition_for_now = st.selectbox(
            "Edition to run",
            options=EDITION_ORDER,
            format_func=lambda k: f"{k} — {EDITION_META[k]['name']}",
            index=EDITION_ORDER.index(suggested_key),
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

    st.markdown("---")
    st.markdown("### Season/Week 3v1 Edition Schedule & File Status")

    df_status = build_status_table(int(season), int(week))
    st.dataframe(df_status, hide_index=True, use_container_width=True)

    st.markdown("### Existing generated reports for selected season/week")

    files = list_report_files(int(season), int(week))

    if not files:
        st.caption("No generated reports found yet for this season/week.")
    else:
        for pdf in files:
            info = parse_report_filename(pdf)

            col1, col2 = st.columns([4, 1])

            with col1:
                st.write(
                    f"**{pdf.name}**"
                    f"  \nSeason: {info['season'] if info['season'] is not None else 'Unknown'}"
                    f" | Week: {info['week'] if info['week'] is not None else 'Unknown'}"
                    f" | Edition: {info['edition_name']}"
                    f" | Modified: {info['modified'].strftime('%Y-%m-%d %H:%M')}"
                )

            with col2:
                st.download_button(
                    label="Download",
                    data=pdf.read_bytes(),
                    file_name=pdf.name,
                    mime="application/pdf",
                    key=f"dl_{pdf.name}",
                )

    st.markdown("---")
    st.markdown("### Historical Reports Archive (2020 onward if available)")

    archive_df = build_archive_table()

    if archive_df.empty:
        st.caption("No report PDFs found in the archive yet.")
    else:
        valid_seasons = sorted(
            [int(x) for x in archive_df["Season"].dropna().unique().tolist()]
        )
        archive_season_options = ["All"] + valid_seasons

        archive_col1, archive_col2, archive_col3 = st.columns(3)

        with archive_col1:
            archive_season = st.selectbox(
                "Archive Season",
                options=archive_season_options,
                index=0 if len(archive_season_options) > 1 else 0,
            )

        if archive_season == "All":
            week_options = ["All"] + sorted(
                [int(x) for x in archive_df["Week"].dropna().unique().tolist()]
            )
        else:
            week_options = ["All"] + sorted(
                [
                    int(x) for x in archive_df.loc[
                        archive_df["Season"] == archive_season, "Week"
                    ].dropna().unique().tolist()
                ]
            )

        with archive_col2:
            archive_week = st.selectbox(
                "Archive Week",
                options=week_options,
                index=0,
            )

        if archive_season == "All":
            edition_options = ["All"] + sorted(
                archive_df["Edition"].dropna().astype(str).unique().tolist()
            )
        else:
            filt = archive_df["Season"] == archive_season
            if archive_week != "All":
                filt = filt & (archive_df["Week"] == archive_week)
            edition_options = ["All"] + sorted(
                archive_df.loc[filt, "Edition"].dropna().astype(str).unique().tolist()
            )

        with archive_col3:
            archive_edition = st.selectbox(
                "Archive Edition",
                options=edition_options,
                index=0,
            )

        filtered = archive_df.copy()

        if archive_season != "All":
            filtered = filtered[filtered["Season"] == archive_season]

        if archive_week != "All":
            filtered = filtered[filtered["Week"] == archive_week]

        if archive_edition != "All":
            filtered = filtered[filtered["Edition"] == archive_edition]

        st.dataframe(filtered, hide_index=True, use_container_width=True)

        st.markdown("#### Download from archive")

        archive_files = []
        for _, row in filtered.iterrows():
            path = REPORTS_DIR / row["File"]
            if path.exists():
                archive_files.append(path)

        if not archive_files:
            st.caption("No matching archive PDFs available to download.")work
        else:
            for pdf in archive_files[:50]:
                info = parse_report_filename(pdf)
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.write(
                        f"**{pdf.name}**"
                        f"  \nSeason: {info['season'] if info['season'] is not None else 'Unknown'}"
                        f" | Week: {info['week'] if info['week'] is not None else 'Unknown'}"
                        f" | Edition: {info['edition_name']}"
                    )

                with col2:
                    st.download_button(
                        label="Download",
                        data=pdf.read_bytes(),
                        file_name=pdf.name,
                        mime="application/pdf",
                        key=f"archive_dl_{pdf.name}",
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