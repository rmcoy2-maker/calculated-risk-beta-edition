from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from report_data import (
    EDITION_LABELS,
    TEAM_ABBR,
    best_market_snapshot,
    edition_games,
    filter_for_week,
    infer_games,
    load_all,
    top_edges_for_game,
    top_props_for_game,
)


@dataclass
class ReportArtifacts:
    pdf_path: Path
    json_path: Path


DAY_HEADLINES = {
    "tnf": "Single-game Fort Knox board built from live markets, best edges, and player-level value.",
    "sunday_morning": "Full-slate preview with flagship sides, totals, parlay anchors, and market-shape narrative.",
    "sunday_afternoon": "Late-window refresh built from remaining board value, updated market shape, and surviving premium edges.",
    "snf": "Primetime spotlight with the cleanest remaining side-total structure and same-game angles.",
    "monday": "Monday board with recap context, strongest surviving edges, and clean anchor legs.",
    "tuesday": "Weekly wrap with strongest graded ideas, portfolio notes, and next-cycle watchlist.",
}

CONF_BANDS = [
    (90, "Gold", "Fort Knox Tier (ultra-rare)"),
    (75, "Dark Green", "Premium Edge"),
    (60, "Green", "Playable Edge"),
    (45, "Yellow", "Light Lean"),
    (25, "Amber", "Very Light"),
    (0, "Red", "Chaos / No Play"),
]


def _title_case_market(text: object) -> str:
    if text is None or pd.isna(text):
        return "Market"
    s = str(text).strip()
    return s.replace("_", " ").title() if s else "Market"


def _fmt_prob(x: object) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    if np.isnan(v):
        return "-"
    return f"{100.0 * v:.1f}%"


def _fmt_ev(x: object) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    if np.isnan(v):
        return "-"
    return f"{v:+.3f}"


def _fmt_odds(x: object) -> str:
    try:
        v = int(float(x))
    except Exception:
        return "-"
    return f"{v:+d}"


def _abbr(matchup: str) -> str:
    parts = [p.strip() for p in str(matchup).split("@")]
    if len(parts) != 2:
        return matchup
    away = TEAM_ABBR.get(parts[0].upper(), parts[0].upper())
    home = TEAM_ABBR.get(parts[1].upper(), parts[1].upper())
    return f"{away} @ {home}"


def confidence_score(row: pd.Series) -> int:
    p = pd.to_numeric(pd.Series([row.get("p_win")]), errors="coerce").iloc[0]
    edge = pd.to_numeric(pd.Series([row.get("edge_pct")]), errors="coerce").iloc[0]
    ev = pd.to_numeric(pd.Series([row.get("ev")]), errors="coerce").iloc[0]
    parlay = pd.to_numeric(pd.Series([row.get("parlay_proba")]), errors="coerce").iloc[0]
    score = 50.0
    if pd.notna(p):
        score += (float(p) - 0.5) * 65.0
    if pd.notna(edge):
        score += float(edge) * 180.0
    if pd.notna(ev):
        score += float(ev) * 85.0
    if pd.notna(parlay):
        score += min(float(parlay), 0.85) * 12.0
    return int(max(0, min(99, round(score))))


def confidence_band(score: int) -> tuple[str, str]:
    for threshold, label, desc in CONF_BANDS:
        if score >= threshold:
            return label, desc
    return "Red", "Chaos / No Play"


def weather_note(game: pd.Series) -> str:
    wind = pd.to_numeric(pd.Series([game.get("weather_wind_mph")]), errors="coerce").iloc[0]
    temp = pd.to_numeric(pd.Series([game.get("weather_temperature")]), errors="coerce").iloc[0]
    stadium = str(game.get("stadium", "")).strip()
    detail = str(game.get("weather_detail", "")).strip()
    if stadium:
        indoor_hint = any(x in stadium.lower() for x in ["dome", "indoor", "roof", "stadium"])
        if indoor_hint and pd.isna(wind):
            return f"Venue: {stadium}. Environment reads neutral-to-fast unless market is already fully inflated."
    bits = []
    if pd.notna(wind):
        if wind >= 20:
            bits.append(f"wind {wind:.0f} mph is a real scoring penalty")
        elif wind >= 15:
            bits.append(f"wind {wind:.0f} mph is a mild-to-moderate drag")
        elif wind > 0:
            bits.append(f"wind {wind:.0f} mph is mostly neutral")
    if pd.notna(temp):
        if temp < 25:
            bits.append(f"temperature {temp:.0f}F suppresses efficiency")
        elif temp < 40:
            bits.append(f"temperature {temp:.0f}F is a small efficiency tax")
    if detail:
        bits.append(detail)
    return "Environment: " + ("; ".join(bits) if bits else "neutral board")


def narrative_for_edge(row: pd.Series, game: Optional[pd.Series] = None) -> str:
    market = str(row.get("market", "")).lower()
    edge = pd.to_numeric(pd.Series([row.get("edge_pct")]), errors="coerce").iloc[0]
    p = pd.to_numeric(pd.Series([row.get("p_win")]), errors="coerce").iloc[0]
    ev = pd.to_numeric(pd.Series([row.get("ev")]), errors="coerce").iloc[0]
    side = str(row.get("side", "")).strip()
    clauses: list[str] = []
    if pd.notna(edge):
        if edge >= 0.06:
            clauses.append("model sees a clear market miss")
        elif edge >= 0.03:
            clauses.append("model still holds a playable cushion over implied price")
        elif edge <= -0.01:
            clauses.append("number is thin and closer to fair than the headline suggests")
    if pd.notna(p):
        if p >= 0.62:
            clauses.append("win-rate profile is comfortably above break-even")
        elif p >= 0.56:
            clauses.append("hit-rate sits in the stable playable band")
    if pd.notna(ev):
        if ev >= 0.08:
            clauses.append("EV sits in premium territory")
        elif ev >= 0.03:
            clauses.append("EV is positive without needing a heroic outcome")
    if "total" in market or market == "over" or market == "under":
        clauses.append("total structure matters more than brand-name team strength here")
    elif "moneyline" in market or market == "h2h":
        clauses.append(f"{side or 'this side'} wins more often than the price implies")
    elif "spread" in market:
        clauses.append("margin shape supports the current number")
    if game is not None:
        wn = weather_note(game)
        if "penalty" in wn or "drag" in wn:
            clauses.append(wn.replace("Environment: ", ""))
    if not clauses:
        clauses.append("market and model align well enough to keep this on the board")
    return "; ".join(dict.fromkeys(clauses)).capitalize() + "."


def summarize_game(game: pd.Series, edges: pd.DataFrame, parlays: pd.DataFrame, markets: pd.DataFrame) -> dict:
    gid = str(game.get("game_id", ""))
    matchup = str(game.get("matchup", "")).strip()
    g_edges = top_edges_for_game(edges, gid, matchup, limit=8)
    g_parlays = top_edges_for_game(parlays, gid, matchup, limit=8) if not parlays.empty else pd.DataFrame()
    g_props = top_props_for_game(markets, g_edges, gid, matchup, limit=4)
    snap = best_market_snapshot(markets, gid, matchup)

    top_side = g_edges.iloc[0] if not g_edges.empty else None

    parlay_best = None
    if not g_parlays.empty:
        sort_cols = [c for c in ["parlay_score", "parlay_proba", "ev"] if c in g_parlays.columns]
        if sort_cols:
            g_parlays = g_parlays.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
        parlay_best = g_parlays.iloc[0]

    edge_lines = []
    for _, row in g_edges.head(3).iterrows():
        score = confidence_score(row)
        band, _ = confidence_band(score)
        desc = f"{_title_case_market(row.get('market'))}: {row.get('side', '')}"
        if pd.notna(row.get("line")):
            desc += f" {float(row['line']):+.1f}"
        if pd.notna(row.get("odds")):
            desc += f" at {_fmt_odds(row['odds'])}"
        desc += f" | conf {score} ({band}) | win {_fmt_prob(row.get('p_win'))} | EV {_fmt_ev(row.get('ev'))}"
        edge_lines.append(desc)

    prop_lines = []
    for _, row in g_props.head(3).iterrows():
        desc = f"{_title_case_market(row.get('market'))}: {row.get('side', '')}"
        if pd.notna(row.get("line")):
            desc += f" {float(row['line']):.1f}"
        if pd.notna(row.get("odds")):
            desc += f" at {_fmt_odds(row['odds'])}"
        prop_lines.append(desc)

    top_score = confidence_score(top_side) if top_side is not None else None
    top_band = confidence_band(top_score)[0] if top_score is not None else None
    line_side = snap.get("side", "")
    line_total = snap.get("total", "")

    return {
        "game_id": gid,
        "matchup": matchup,
        "matchup_short": _abbr(matchup),
        "kickoff": "" if pd.isna(game.get("game_dt_et")) else pd.to_datetime(game.get("game_dt_et")).strftime("%a %b %d, %I:%M %p ET"),
        "market_side": line_side,
        "market_total": line_total,
        "weather_note": weather_note(game),
        "headline": narrative_for_edge(top_side, game) if top_side is not None else "Board is driven more by market shape than by one runaway edge.",
        "top_edge": None if top_side is None else {
            "market": str(top_side.get("market", "")),
            "side": str(top_side.get("side", "")),
            "line": None if pd.isna(top_side.get("line")) else float(top_side.get("line")),
            "odds": None if pd.isna(top_side.get("odds")) else int(float(top_side.get("odds"))),
            "p_win": None if pd.isna(top_side.get("p_win")) else float(top_side.get("p_win")),
            "ev": None if pd.isna(top_side.get("ev")) else float(top_side.get("ev")),
            "book": str(top_side.get("book", "")),
            "confidence": top_score,
            "band": top_band,
            "why": narrative_for_edge(top_side, game),
        },
        "best_parlay": None if parlay_best is None else {
            "side": str(parlay_best.get("side", "")),
            "market": str(parlay_best.get("market", "")),
            "parlay_proba": None if pd.isna(parlay_best.get("parlay_proba")) else float(parlay_best.get("parlay_proba")),
            "ev": None if pd.isna(parlay_best.get("ev")) else float(parlay_best.get("ev")),
            "legs": None if pd.isna(parlay_best.get("legs")) else int(float(parlay_best.get("legs"))),
            "parlay_score": None if pd.isna(parlay_best.get("parlay_score")) else float(parlay_best.get("parlay_score")),
        },
        "edge_lines": edge_lines,
        "prop_lines": prop_lines,
    }


def _portfolio_from_parlays(parlays: pd.DataFrame) -> list[dict]:
    if parlays.empty:
        return []
    pool = parlays.copy()
    sort_cols = [c for c in ["parlay_score", "parlay_proba", "ev"] if c in pool.columns]
    if sort_cols:
        pool = pool.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    out = []
    for i, (_, row) in enumerate(pool.head(3).iterrows(), start=1):
        conf = confidence_score(row)
        band, _ = confidence_band(conf)
        out.append({
            "name": f"Parlay #{i}",
            "label": f"{str(row.get('side', '')).strip()} / {_title_case_market(row.get('market'))}",
            "legs": None if pd.isna(row.get("legs")) else int(float(row.get("legs"))),
            "prob": None if pd.isna(row.get("parlay_proba")) else float(row.get("parlay_proba")),
            "ev": None if pd.isna(row.get("ev")) else float(row.get("ev")),
            "confidence": conf,
            "band": band,
        })
    return out


def _performance_summary(edges: pd.DataFrame, parlays: pd.DataFrame) -> list[str]:
    lines = []
    if not edges.empty:
        top5 = edges.sort_values(["ev", "edge_pct", "p_win"], ascending=[False, False, False], na_position="last").head(5)
        mean_ev = pd.to_numeric(top5.get("ev"), errors="coerce").mean()
        mean_p = pd.to_numeric(top5.get("p_win"), errors="coerce").mean()
        lines.append(f"Top 5 edges carry average win probability {_fmt_prob(mean_p)} with average EV {_fmt_ev(mean_ev)}.")
        market_mix = top5.get("market", pd.Series(dtype=str)).astype("string").str.upper().value_counts().head(3)
        if not market_mix.empty:
            lines.append("Top board concentration: " + ", ".join(f"{k} x{v}" for k, v in market_mix.items()) + ".")
    if not parlays.empty:
        mean_prob = pd.to_numeric(parlays.get("parlay_proba"), errors="coerce").dropna().head(25).mean()
        mean_ev = pd.to_numeric(parlays.get("ev"), errors="coerce").dropna().head(25).mean()
        if pd.notna(mean_prob):
            lines.append(f"Parlay portfolio average hit profile is {_fmt_prob(mean_prob)} with EV {_fmt_ev(mean_ev)} across the strongest carryovers.")
    if not lines:
        lines.append("Performance appendix is waiting on stable week-level grading inputs.")
    return lines


def build_report_payload(season: Optional[int], week: int, edition: str) -> dict:
    data = load_all()
    edges = filter_for_week(data.edges, season, week)
    parlays = filter_for_week(data.parlays, season, week)
    markets = filter_for_week(data.markets, season, week)
    games = edition_games(infer_games(data, season, week), edition)

    sections = [summarize_game(row, edges, parlays, markets) for _, row in games.iterrows()]

    top_edges_df = edges.sort_values(["ev", "edge_pct", "p_win"], ascending=[False, False, False], na_position="last").head(10) if not edges.empty else pd.DataFrame()
    top_edges = []
    for _, r in top_edges_df.iterrows():
        score = confidence_score(r)
        band, _ = confidence_band(score)
        top_edges.append({
            "matchup": str(r.get("matchup", "")),
            "matchup_short": _abbr(str(r.get("matchup", ""))),
            "market": str(r.get("market", "")),
            "side": str(r.get("side", "")),
            "line": None if pd.isna(r.get("line")) else float(r.get("line")),
            "odds": None if pd.isna(r.get("odds")) else int(float(r.get("odds"))),
            "p_win": None if pd.isna(r.get("p_win")) else float(r.get("p_win")),
            "ev": None if pd.isna(r.get("ev")) else float(r.get("ev")),
            "confidence": score,
            "band": band,
            "why": narrative_for_edge(r),
        })

    portfolio = _portfolio_from_parlays(parlays)
    heatmap = [{
        "game": x["matchup_short"],
        "angle": f"{_title_case_market(x['market'])}: {x['side']}",
        "confidence": x["confidence"],
        "band": x["band"],
    } for x in top_edges[:8]]

    scoring_appendix = []
    for sec in sections[:3]:
        te = sec.get("top_edge")
        if not te:
            continue
        scoring_appendix.append({
            "matchup": sec["matchup_short"],
            "component_rows": [
                ["Model Win Rate", _fmt_prob(te.get("p_win")), "How often the core angle lands in model space"],
                ["EV", _fmt_ev(te.get("ev")), "Edge after price"],
                ["Market Shape", sec.get("market_side") or sec.get("market_total") or "-", "What the live board is offering"],
                ["Environment", sec.get("weather_note", "-"), "Venue / weather pressure"],
            ],
        })

    return {
        "edition": edition,
        "edition_label": EDITION_LABELS.get(edition, edition),
        "season": season,
        "week": week,
        "headline": DAY_HEADLINES.get(edition, "Fort Knox board generated from live edges, parlays, and market context."),
        "counts": {
            "games": int(len(games)),
            "edges": int(len(edges)),
            "parlays": int(len(parlays)),
            "markets": int(len(markets)),
        },
        "confidence_bands": [{"range": f"{thr}-100" if thr == 90 else None, "label": label, "description": desc} for thr, label, desc in CONF_BANDS],
        "top_edges": top_edges,
        "heatmap": heatmap,
        "portfolio": portfolio,
        "appendix_performance": _performance_summary(edges, parlays),
        "appendix_scoring": scoring_appendix,
        "sections": sections,
    }


def build_pdf(payload: dict, out_path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=20, leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#163A63")))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Heading2"], fontSize=13, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#4A4A4A")))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#163A63"), spaceAfter=7))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=8.5, leading=11))
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 12

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"Fort Knox Week {payload['week']} {payload['edition_label']}",
        author="Calculated Risk / Doc Odds",
    )

    story = []
    story.append(Paragraph("CALCULATED RISK™ · EDGE FACTOR™ · 3v1™", styles["CoverTitle"]))
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(f"FORT KNOX — WEEK {payload['week']} {payload['edition_label'].upper()}", styles["SubTitle"]))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph("Prepared by R.M. Coy, Ph.D. (\"Doc Odds\")", styles["BodyText"]))
    story.append(Paragraph("Educational & entertainment purposes only.", styles["BodyText"]))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("0. Slate Snapshot & Confidence Key", styles["SectionTitle"]))
    counts = payload["counts"]
    story.append(Paragraph(payload["headline"], styles["BodyText"]))
    story.append(Paragraph(f"Board size: {counts['games']} games | {counts['edges']} edges | {counts['parlays']} parlay rows | {counts['markets']} market rows.", styles["BodyText"]))
    band_table = [["Band", "Meaning"]] + [[label, desc] for _, label, desc in CONF_BANDS]
    bt = Table(band_table, colWidths=[1.2 * inch, 4.7 * inch], repeatRows=1)
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163A63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#C9D3DE")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DFE5ED")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(Spacer(1, 0.07 * inch))
    story.append(bt)
    story.append(Spacer(1, 0.14 * inch))

    if payload["top_edges"]:
        story.append(Paragraph("1. Top 5 Edges", styles["SectionTitle"]))
        for i, row in enumerate(payload["top_edges"][:5], start=1):
            title = f"{i}) {row['matchup_short']} — {_title_case_market(row['market'])}: {row['side']}"
            if row.get("line") is not None:
                title += f" {row['line']:+.1f}"
            story.append(Paragraph(title, styles["BodyText"]))
            story.append(Paragraph(f"Confidence: {row['confidence']} → {row['band']} | win {_fmt_prob(row['p_win'])} | EV {_fmt_ev(row['ev'])}", styles["Tiny"]))
            story.append(Paragraph(row["why"], styles["Tiny"]))
            story.append(Spacer(1, 0.05 * inch))
        story.append(Spacer(1, 0.08 * inch))

    if payload["sections"]:
        story.append(Paragraph("2. Game Reports", styles["SectionTitle"]))
        for idx, sec in enumerate(payload["sections"], start=1):
            story.append(Paragraph(sec["matchup"], styles["Heading3"]))
            bits = [b for b in [sec["kickoff"], sec["market_side"], sec["market_total"]] if b]
            if bits:
                story.append(Paragraph(" | ".join(bits), styles["Tiny"]))
            story.append(Paragraph(sec["weather_note"], styles["Tiny"]))
            story.append(Paragraph(sec["headline"], styles["BodyText"]))
            if sec["top_edge"]:
                te = sec["top_edge"]
                line_txt = f" {te['line']:+.1f}" if te["line"] is not None else ""
                odds_txt = f" at {te['odds']:+d}" if te["odds"] is not None else ""
                story.append(Paragraph(f"<b>Best angle:</b> {_title_case_market(te['market'])} — {te['side']}{line_txt}{odds_txt} | conf {te['confidence']} ({te['band']})", styles["BodyText"]))
                story.append(Paragraph(f"Why: {te['why']}", styles["Tiny"]))
            if sec["best_parlay"]:
                bp = sec["best_parlay"]
                story.append(Paragraph(f"<b>Parlay carryover:</b> {bp['side']} ({_title_case_market(bp['market'])}) | legs {bp.get('legs') or '-'} | hit {_fmt_prob(bp.get('parlay_proba'))} | EV {_fmt_ev(bp.get('ev'))}", styles["Tiny"]))
            if sec["edge_lines"]:
                story.append(Paragraph("<b>Angles</b><br/>" + "<br/>".join(f"• {x}" for x in sec["edge_lines"]), styles["Tiny"]))
            if sec["prop_lines"]:
                story.append(Paragraph("<b>Prop / market notes</b><br/>" + "<br/>".join(f"• {x}" for x in sec["prop_lines"]), styles["Tiny"]))
            story.append(Spacer(1, 0.09 * inch))
            if idx % 3 == 0 and idx != len(payload["sections"]):
                story.append(PageBreak())

    if payload["heatmap"]:
        story.append(Paragraph("3. Fort Knox Heatmap", styles["SectionTitle"]))
        heat = [["Game", "Model Angle", "Conf", "Color"]]
        for row in payload["heatmap"]:
            heat.append([row["game"], row["angle"], str(row["confidence"]), row["band"]])
        ht = Table(heat, colWidths=[1.1 * inch, 3.8 * inch, 0.55 * inch, 1.0 * inch], repeatRows=1)
        ht.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163A63")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#C9D3DE")),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DFE5ED")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.3),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ht)
        story.append(Spacer(1, 0.12 * inch))

    if payload["portfolio"]:
        story.append(Paragraph("4. Fort Knox Parlay Portfolio", styles["SectionTitle"]))
        for p in payload["portfolio"]:
            story.append(Paragraph(f"<b>{p['name']}</b> — {p['label']}", styles["BodyText"]))
            story.append(Paragraph(f"Confidence: {p['confidence']} → {p['band']} | legs {p.get('legs') or '-'} | hit {_fmt_prob(p.get('prob'))} | EV {_fmt_ev(p.get('ev'))}", styles["Tiny"]))
            story.append(Spacer(1, 0.04 * inch))
        story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Appendix A — Performance Model Note", styles["SectionTitle"]))
    for line in payload["appendix_performance"]:
        story.append(Paragraph(f"• {line}", styles["Tiny"]))
    story.append(Spacer(1, 0.1 * inch))

    if payload["appendix_scoring"]:
        story.append(Paragraph("Appendix B — Scoring / Market Decomposition", styles["SectionTitle"]))
        for block in payload["appendix_scoring"]:
            story.append(Paragraph(block["matchup"], styles["Heading3"]))
            tbl = Table([["Component", "Value", "Comment"]] + block["component_rows"], colWidths=[1.4 * inch, 1.1 * inch, 3.7 * inch], repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163A63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#C9D3DE")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DFE5ED")),
                ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.08 * inch))

    if not payload["sections"]:
        story.append(Paragraph("No matching games were found for this week/edition in the current files. Rebuild the canonical weekly CSVs and rerun.", styles["BodyText"]))

    doc.build(story)


def generate_report(season: Optional[int], week: int, edition: str, out_dir: Optional[Path] = None) -> ReportArtifacts:
    data = load_all()
    out_dir = out_dir or data.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload(season, week, edition)

    stem = f"week{week}_{edition}_3v1"
    pdf_path = out_dir / f"{stem}.pdf"
    json_path = out_dir / f"{stem}.json"

    build_pdf(payload, pdf_path)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ReportArtifacts(pdf_path=pdf_path, json_path=json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Fort Knox / Doc Odds report from edges, parlay scores, markets, and game files.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--edition", required=True, choices=list(EDITION_LABELS))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    artifacts = generate_report(args.season, args.week, args.edition, Path(args.out_dir) if args.out_dir else None)
    print(f"[OK] PDF: {artifacts.pdf_path}")
    print(f"[OK] JSON: {artifacts.json_path}")


if __name__ == "__main__":
    main()
