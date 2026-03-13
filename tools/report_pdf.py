from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def draw_line(c: canvas.Canvas, text: str, x: float, y: float, size: int = 11) -> float:
    c.setFont("Helvetica", size)
    c.drawString(x, y, text)
    return y - 14


def render_report_pdf(report: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    x = 0.7 * inch
    y = height - 0.8 * inch

    meta = report.get("meta", {})
    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "CALCULATED RISK™ · EDGE FACTOR™ · 3v1™")
    y -= 24
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, meta.get("title", "FORT KNOX REPORT"))
    y -= 22

    y = draw_line(c, f"Week: {meta.get('week', '')}", x, y)
    y = draw_line(c, f"Edition: {meta.get('edition', '')}", x, y)
    y = draw_line(c, f"As Of: {meta.get('asof', '')}", x, y)
    y = draw_line(c, "Prepared by Doc Odds — Educational & Entertainment Use Only", x, y)
    y -= 10

    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "Top 5 Edges")
    y -= 18

    for i, row in enumerate(report.get("top_edges", []), start=1):
        line = f"{i}) {row.get('label', 'Play')} | Conf {row.get('confidence', '')} | Score {round(row.get('fort_knox_score', 0), 1)}"
        y = draw_line(c, line, x, y, 10)
        y = draw_line(c, f"   {row.get('blurb', '')}", x, y, 9)

        if y < 80:
            c.showPage()
            y = height - 0.8 * inch

    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "Heatmap Board")
    y -= 18

    for row in report.get("heatmap", []):
        line = f"- {row.get('label', 'Play')} | {row.get('confidence_band', '')} | {round(row.get('fort_knox_score', 0), 1)}"
        y = draw_line(c, line, x, y, 10)

        if y < 80:
            c.showPage()
            y = height - 0.8 * inch

    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "Game Scripts")
    y -= 18

    for row in report.get("games", []):
        y = draw_line(c, row.get("matchup", "Game"), x, y, 10)
        y = draw_line(c, f"   {row.get('script', '')}", x, y, 9)

        if y < 80:
            c.showPage()
            y = height - 0.8 * inch

    c.save()