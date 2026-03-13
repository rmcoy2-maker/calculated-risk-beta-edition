from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from report_data import load_report_inputs
from report_metrics import heatmap_board, top_edges, top_parlay_legs
from report_narrative import confidence_band, edge_blurb, game_script, parlay_blurb
from report_pdf import render_report_pdf


THIS = Path(__file__).resolve()


def find_repo_root() -> Path:
    for p in [THIS.parent] + list(THIS.parents):
        if (p / "streamlit_app.py").exists():
            return p
    return Path.cwd()


ROOT = find_repo_root()
EXPORTS_DIR = ROOT / "exports"
CANONICAL_DIR = EXPORTS_DIR / "canonical"
REPORTS_DIR = EXPORTS_DIR / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Fort Knox / 3v1 report.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--edition",
        type=str,
        required=True,
        choices=[
            "tnf",
            "sunday_morning",
            "sunday_afternoon",
            "snf",
            "monday",
            "tuesday",
        ],
    )
    parser.add_argument(
        "--asof",
        type=str,
        default=None,
        help="Optional as-of timestamp (ISO-like string).",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Directory to write PDF/JSON outputs.",
    )
    return parser.parse_args()


def _safe_get(row: pd.Series | dict[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        value = row.get(key, default)
    else:
        value = row[key] if key in row else default

    if pd.isna(value):
        return default
    return value


def _pick_first(row: pd.Series, keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and pd.notna(row[key]):
            val = row[key]
            if val != "":
                return val
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def edition_title(edition: str, week: int) -> str:
    labels = {
        "tnf": f"FORT KNOX — WEEK {week} THURSDAY NIGHT EDITION",
        "sunday_morning": f"FORT KNOX — WEEK {week} SUNDAY MORNING EDITION",
        "sunday_afternoon": f"FORT KNOX — WEEK {week} SUNDAY AFTERNOON EDITION",
        "snf": f"FORT KNOX — WEEK {week} SUNDAY NIGHT EDITION",
        "monday": f"FORT KNOX — WEEK {week} MONDAY EDITION",
        "tuesday": f"FORT KNOX — WEEK {week} TUESDAY WRAP EDITION",
    }
    return labels.get(edition, f"FORT KNOX — WEEK {week} REPORT")


def build_top_edges_payload(edges_df: pd.DataFrame) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []

    if edges_df.empty:
        return payload

    ranked = top_edges(edges_df, n=5)

    for _, row in ranked.iterrows():
        confidence = _to_float(_pick_first(row, ["confidence", "confidence_score"], 50.0), 50.0)
        item = row.to_dict()

        label = _pick_first(
            row,
            [
                "label",
                "play",
                "market_label",
                "market",
                "bet_label",
                "pick",
                "selection",
            ],
            default="Edge",
        )

        matchup = _pick_first(
            row,
            ["matchup", "game", "game_label"],
            default="",
        )

        if not matchup:
            away = _pick_first(row, ["away_team", "team_away"], "")
            home = _pick_first(row, ["home_team", "team_home"], "")
            if away and home:
                matchup = f"{away} @ {home}"

        item["matchup"] = matchup
        item["label"] = label
        item["confidence"] = confidence
        item["confidence_band"] = confidence_band(confidence)
        item["blurb"] = edge_blurb(row)

        payload.append(item)

    return payload


def build_heatmap_payload(edges_df: pd.DataFrame) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []

    if edges_df.empty:
        return payload

    ranked = heatmap_board(edges_df, n=12)

    for _, row in ranked.iterrows():
        confidence = _to_float(_pick_first(row, ["confidence", "confidence_score"], 50.0), 50.0)
        item = row.to_dict()

        label = _pick_first(
            row,
            [
                "label",
                "play",
                "market_label",
                "market",
                "bet_label",
                "pick",
                "selection",
            ],
            default="Edge",
        )

        matchup = _pick_first(
            row,
            ["matchup", "game", "game_label"],
            default="",
        )

        if not matchup:
            away = _pick_first(row, ["away_team", "team_away"], "")
            home = _pick_first(row, ["home_team", "team_home"], "")
            if away and home:
                matchup = f"{away} @ {home}"

        item["matchup"] = matchup
        item["label"] = label
        item["confidence"] = confidence
        item["confidence_band"] = confidence_band(confidence)

        payload.append(item)

    return payload


def build_games_payload(games_df: pd.DataFrame) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []

    if games_df.empty:
        return payload

    for _, row in games_df.iterrows():
        away = _pick_first(row, ["away_team", "team_away"], "Away")
        home = _pick_first(row, ["home_team", "team_home"], "Home")

        payload.append(
            {
                "matchup": f"{away} @ {home}",
                "away_team": away,
                "home_team": home,
                "market_total": _pick_first(row, ["market_total", "total"], ""),
                "market_spread": _pick_first(row, ["market_spread", "spread"], ""),
                "model_away_score": _pick_first(row, ["model_away_score", "away_score_model"], ""),
                "model_home_score": _pick_first(row, ["model_home_score", "home_score_model"], ""),
                "script": game_script(row),
            }
        )

    return payload


def build_parlay_payload(parlays_df: pd.DataFrame) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []

    if parlays_df.empty:
        return payload

    ranked = top_parlay_legs(parlays_df, n=9)
    if ranked.empty:
        return payload

    rows = ranked.reset_index(drop=True)

    chunk_size = 3
    for i in range(0, len(rows), chunk_size):
        chunk = rows.iloc[i : i + chunk_size].copy()
        if chunk.empty:
            continue

        legs: list[str] = []
        conf_vals: list[float] = []

        for _, row in chunk.iterrows():
            label = _pick_first(
                row,
                [
                    "label",
                    "play",
                    "market_label",
                    "market",
                    "bet_label",
                    "pick",
                    "selection",
                    "leg_label",
                ],
                default="Parlay leg",
            )
            legs.append(str(label))
            conf_vals.append(_to_float(_pick_first(row, ["confidence", "parlay_score"], 55.0), 55.0))

        portfolio = {
            "title": f"Parlay #{len(payload) + 1}",
            "legs": legs,
            "confidence": round(sum(conf_vals) / len(conf_vals), 1) if conf_vals else 55.0,
            "blurb": parlay_blurb(chunk),
        }
        payload.append(portfolio)

    return payload


def build_appendix_payload(
    edges_df: pd.DataFrame,
    games_df: pd.DataFrame,
    parlays_df: pd.DataFrame,
) -> dict[str, Any]:
    appendix: dict[str, Any] = {
        "summary": {},
        "notes": [],
    }

    appendix["summary"] = {
        "edge_count": int(len(edges_df)),
        "game_count": int(len(games_df)),
        "parlay_count": int(len(parlays_df)),
    }

    notes = [
        "Top board is ranked by Fort Knox score, blending edge, confidence, disagreement, synergy, weather, and stability.",
        "Narratives are template-driven from the canonical week files.",
        "This report is for educational and entertainment purposes only.",
    ]

    appendix["notes"] = notes
    return appendix


def build_report(
    season: int,
    week: int,
    edition: str,
    asof: str | None,
) -> dict[str, Any]:
    inputs = load_report_inputs(
        season=season,
        week=week,
        canonical_dir=CANONICAL_DIR,
    )

    games_df = inputs.get("games", pd.DataFrame())
    edges_df = inputs.get("edges", pd.DataFrame())
    markets_df = inputs.get("markets", pd.DataFrame())
    parlays_df = inputs.get("parlay_scores", pd.DataFrame())
    scores_df = inputs.get("scores", pd.DataFrame())

    # Keep the report resilient even if some files are sparse.
    if not games_df.empty and not markets_df.empty:
        # Optional lightweight merge hook if your report_data loader does not already merge.
        merge_keys = [k for k in ["game_id", "matchup", "game"] if k in games_df.columns and k in markets_df.columns]
        if merge_keys:
            key = merge_keys[0]
            market_cols = [c for c in markets_df.columns if c not in games_df.columns or c == key]
            try:
                games_df = games_df.merge(markets_df[market_cols], on=key, how="left")
            except Exception:
                pass

    if not games_df.empty and not scores_df.empty:
        merge_keys = [k for k in ["game_id", "matchup", "game"] if k in games_df.columns and k in scores_df.columns]
        if merge_keys:
            key = merge_keys[0]
            score_cols = [c for c in scores_df.columns if c not in games_df.columns or c == key]
            try:
                games_df = games_df.merge(scores_df[score_cols], on=key, how="left")
            except Exception:
                pass

    report = {
        "meta": {
            "title": edition_title(edition, week),
            "season": season,
            "week": week,
            "edition": edition,
            "asof": asof,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "top_edges": build_top_edges_payload(edges_df),
        "heatmap": build_heatmap_payload(edges_df),
        "games": build_games_payload(games_df),
        "parlays": build_parlay_payload(parlays_df),
        "appendix": build_appendix_payload(edges_df, games_df, parlays_df),
    }

    return report


def write_report_outputs(
    report: dict[str, Any],
    reports_dir: Path,
    week: int,
    edition: str,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = reports_dir / f"week{week}_{edition}_3v1.pdf"
    json_path = reports_dir / f"week{week}_{edition}_3v1.json"

    # Real PDF only. Never write text placeholders to a .pdf path.
    render_report_pdf(report, pdf_path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return pdf_path, json_path


def main() -> int:
    args = parse_args()

    reports_dir = Path(args.reports_dir)

    try:
        report = build_report(
            season=args.season,
            week=args.week,
            edition=args.edition,
            asof=args.asof,
        )

        pdf_path, json_path = write_report_outputs(
            report=report,
            reports_dir=reports_dir,
            week=args.week,
            edition=args.edition,
        )

        print(f"Generated PDF: {pdf_path}")
        print(f"Generated JSON: {json_path}")
        return 0

    except Exception as e:
        print(f"ERROR generating report: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())