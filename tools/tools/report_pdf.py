from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def draw_line(c, text, x, y, size=11, bold=False):
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    c.drawString(x, y, str(text))
    return y - (size + 4)


def draw_section(c, title, x, y):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 6
    c.line(x, y, x + 470, y)
    return y - 14


def _new_page(c, width, height):
    c.showPage()
    return 0.75 * inch, height - 0.75 * inch


def render_report_pdf(report: dict[str, Any], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    x = 0.75 * inch
    y = height - 0.75 * inch

    meta = report.get("meta", {})

    y = draw_line(c, "CALCULATED RISK™ · EDGE FACTOR™ · 3v1™", x, y, 16, True)
    y = draw_line(c, meta.get("title", "FORT KNOX REPORT"), x, y, 14, True)

    y = draw_line(c, f"Week: {meta.get('week','')}", x, y)
    y = draw_line(c, f"Edition: {meta.get('edition','')}", x, y)
    y = draw_line(c, f"As Of: {meta.get('asof','')}", x, y)
    y = draw_line(c, "Prepared by Doc Odds — Educational & Entertainment Use Only", x, y, 9)

    y -= 6
    y = draw_section(c, "HOW TO READ THE CONFIDENCE BANDS", x, y)
    y = draw_line(c, "Gold (90-95): highest-confidence board only", x, y, 10)
    y = draw_line(c, "Dark Green (75-89): strong edge with solid support", x, y, 10)
    y = draw_line(c, "Green (60-74): playable edge", x, y, 10)
    y = draw_line(c, "Yellow (45-59): lean / context play", x, y, 10)
    y = draw_line(c, "Amber (25-44): weak support", x, y, 10)
    y = draw_line(c, "Red (<25): avoid / informational only", x, y, 10)
    y = draw_line(c, "Simulation basis: 20,000 Monte Carlo runs per game where model score inputs are available.", x, y, 9)

    y -= 8
    y = draw_section(c, "SECTION 1 — TOP EDGES", x, y)

    top_edges = report.get("top_edges", [])
    if not top_edges:
        y = draw_line(c, "No top edges available for this edition.", x, y, 10)
    else:
        for i, row in enumerate(top_edges, start=1):
            label = row.get("label", "Edge")
            conf = row.get("confidence", "")
            score = round(row.get("fort_knox_score", 0), 2)

            y = draw_line(
                c,
                f"{i}) {label} | Confidence: {conf} | Score: {score}",
                x,
                y,
                11,
                True,
            )
            y = draw_line(c, row.get("blurb", ""), x + 10, y, 10)
            y -= 4

            if y < 80:
                x, y = _new_page(c, width, height)

    y -= 8
    if y < 90:
        x, y = _new_page(c, width, height)

    y = draw_section(c, "SECTION 2 — HEATMAP BOARD", x, y)
    heatmap = report.get("heatmap", [])
    if not heatmap:
        y = draw_line(c, "No heatmap rows available for this edition.", x, y, 10)
    else:
        for row in heatmap:
            label = row.get("label", "")
            band = row.get("confidence_band", "")
            score = round(row.get("fort_knox_score", 0), 2)
            y = draw_line(c, f"{label} | {band} | Score {score}", x, y, 10)

            if y < 80:
                x, y = _new_page(c, width, height)

    y -= 8
    if y < 90:
        x, y = _new_page(c, width, height)

    y = draw_section(c, "SECTION 3 — GAME SCRIPTS", x, y)
    games = report.get("games", [])
    if not games:
        y = draw_line(c, "No game scripts available for this edition.", x, y, 10)
    else:
        for row in games:
            y = draw_line(c, row.get("matchup", "Game"), x, y, 11, True)
            y = draw_line(c, row.get("script", ""), x + 10, y, 10)
            y -= 4

            if y < 80:
                x, y = _new_page(c, width, height)

    parlays = report.get("parlays", [])
    if parlays:
        y -= 8
        if y < 90:
            x, y = _new_page(c, width, height)

        y = draw_section(c, "SECTION 4 — PARLAY PORTFOLIO", x, y)

        for p in parlays:
            y = draw_line(c, p.get("title", "Parlay"), x, y, 11, True)

            for leg in p.get("legs", []):
                y = draw_line(c, f"- {leg}", x + 10, y, 10)

            y = draw_line(c, f"Confidence: {p.get('confidence','')}", x + 10, y, 10)
            y = draw_line(c, p.get("blurb", ""), x + 10, y, 9)
            y -= 6

            if y < 80:
                x, y = _new_page(c, width, height)

    appendix = report.get("appendix", {})
    if appendix:
        y -= 8
        if y < 90:
            x, y = _new_page(c, width, height)

        y = draw_section(c, "APPENDIX", x, y)

        summary = appendix.get("summary", {})
        for k, v in summary.items():
            y = draw_line(c, f"{k}: {v}", x, y, 10)

        for note in appendix.get("notes", []):
            y = draw_line(c, f"- {note}", x, y, 9)

            if y < 80:
                x, y = _new_page(c, width, height)

    c.save()