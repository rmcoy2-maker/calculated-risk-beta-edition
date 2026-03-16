# Calculated Risk Project Architecture

## Core folders

### `app/exports/raw`
Raw external inputs and line files. These are source files and should not be treated as final report inputs.

Examples:
- `lines_snapshots.csv`
- `odds_lines_all_long.csv`
- `props_odds_all.csv`
- `lines_open_*.csv`
- `lines_close_*.csv`

---

### `app/exports/canonical`
Canonical weekly files. These are the single source of truth for reports and live board outputs.

Files:
- `cr_games_{season}_w{week}.csv`
- `cr_edges_{season}_w{week}.csv`
- `cr_markets_{season}_w{week}.csv`
- `cr_parlay_scores_{season}_w{week}.csv`
- `cr_scores_{season}_w{week}.csv`

Rule:
- All report generation should read from this folder only.

---

### `app/exports/reports`
Generated report outputs.

Examples:
- `week11_tnf_3v1.pdf`
- `week11_sunday_morning_3v1.pdf`
- `week11_snf_3v1.pdf`
- `week11_monday_3v1.pdf`
- `week11_tuesday_3v1.pdf`
- `week11_sunday_morning_3v1.json`

---

### `app/data`
Persistent supporting datasets, logs, history, and analytics files.

Examples:
- `bankroll.csv`
- `bets_log.csv`
- `model_probs.csv`
- `parlays.csv`
- `weekly_roi.csv`
- `scores_1966-2025.csv`
- `micro_bets.csv`
- `hitrate.csv`
- `line_bias.csv`

Rule:
- These support models, analytics, and history, but are not direct report outputs.

---

### `app/models`
Saved trained model artifacts.

Examples:
- `game_lr.joblib`
- `parlay_model.joblib`

---

### `app/tools`
Automation and report-generation scripts.

Files:
- `build_canonical_week_files.py`
- `report_data.py`
- `generate_report.py`
- `generate_all_editions.py`

Pipeline:
1. Read raw files
2. Build canonical weekly files
3. Generate reports from canonical files

---

### `app/pages`
Streamlit UI pages.

Examples:
- `10_Edge_Scanner.py`
- `21_Reports.py`
- `22_Reports_Hub.py`
- `23_Doc_Odds_Live_Board.py`
- `24_Doc_Odds_AI.py`
- `25_Bet_Engine.py`

---

### `app/utils`
Shared utility logic.

Examples:
- `clv.py`

---

## Canonical workflow

```text
exports/raw
    ->
tools/build_canonical_week_files.py
    ->
exports/canonical
    ->
tools/generate_report.py
    ->
exports/reports
    ->
Streamlit pages / Reports Hub / Live Board