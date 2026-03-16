from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def american_to_implied_prob(odds: pd.Series) -> pd.Series:
    x = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    pos = x > 0
    neg = x < 0
    out.loc[pos] = 100.0 / (x.loc[pos] + 100.0)
    out.loc[neg] = (-x.loc[neg]) / ((-x.loc[neg]) + 100.0)
    return out


def american_to_payout_per_1(odds: pd.Series) -> pd.Series:
    x = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    pos = x > 0
    neg = x < 0
    out.loc[pos] = x.loc[pos] / 100.0
    out.loc[neg] = 100.0 / (-x.loc[neg])
    return out


def expected_value_per_1(p_win: pd.Series, odds: pd.Series) -> pd.Series:
    payout = american_to_payout_per_1(odds)
    p = pd.to_numeric(p_win, errors="coerce")
    return p * payout - (1.0 - p)


def maybe_col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def find_root(start: Path) -> Path:
    start = start.resolve()
    for up in [start.parent] + list(start.parents):
        if (up / "exports").exists():
            return up
    return start.parent


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

def resolve_input(root: Path, season: int, week: int, input_path: Optional[str]) -> Path:
    if input_path:
        p = Path(input_path)
        if not p.is_absolute():
            p = root / p
        return p
    return root / "exports" / "canonical" / f"game_projection_master_{season}_w{week}.csv"


# ------------------------------------------------------------
# Edge row builders
# ------------------------------------------------------------

def build_moneyline_rows(df: pd.DataFrame, ts: str) -> pd.DataFrame:
    home = pd.DataFrame({
        "ts": ts,
        "season": maybe_col(df, "season"),
        "week": maybe_col(df, "week"),
        "sport": "NFL",
        "league": "NFL",
        "market": "H2H",
        "side": maybe_col(df, "home_team") + " ML",
        "team_name": maybe_col(df, "home_team"),
        "line": np.nan,
        "odds": maybe_col(df, "market_home_ml"),
        "p_win": maybe_col(df, "sim_win_prob_home"),
        "edge": maybe_col(df, "home_ml_edge"),
        "clv": maybe_col(df, "home_ml_clv"),
        "close_line": maybe_col(df, "close_home_ml"),
        "home": maybe_col(df, "home_team"),
        "away": maybe_col(df, "away_team"),
        "game_id": maybe_col(df, "game_id"),
        "event_id": maybe_col(df, "event_id"),
        "game_date": maybe_col(df, "game_date"),
        "commence_time": maybe_col(df, "commence_time"),
        "best_overall_market": maybe_col(df, "best_overall_market"),
        "best_overall_label": maybe_col(df, "best_overall_label"),
        "best_overall_edge": maybe_col(df, "best_overall_edge"),
        "best_overall_confidence": maybe_col(df, "best_overall_confidence"),
        "fort_knox_score": maybe_col(df, "fort_knox_score"),
        "market_snapshot_timestamp": maybe_col(df, "market_snapshot_timestamp"),
        "close_snapshot_timestamp": maybe_col(df, "close_snapshot_timestamp"),
        "market_books": maybe_col(df, "market_books"),
        "close_books": maybe_col(df, "close_books"),
        "source_projection_file": "game_projection_master",
    })
    away = pd.DataFrame({
        "ts": ts,
        "season": maybe_col(df, "season"),
        "week": maybe_col(df, "week"),
        "sport": "NFL",
        "league": "NFL",
        "market": "H2H",
        "side": maybe_col(df, "away_team") + " ML",
        "team_name": maybe_col(df, "away_team"),
        "line": np.nan,
        "odds": maybe_col(df, "market_away_ml"),
        "p_win": maybe_col(df, "sim_win_prob_away"),
        "edge": maybe_col(df, "away_ml_edge"),
        "clv": maybe_col(df, "away_ml_clv"),
        "close_line": maybe_col(df, "close_away_ml"),
        "home": maybe_col(df, "home_team"),
        "away": maybe_col(df, "away_team"),
        "game_id": maybe_col(df, "game_id"),
        "event_id": maybe_col(df, "event_id"),
        "game_date": maybe_col(df, "game_date"),
        "commence_time": maybe_col(df, "commence_time"),
        "best_overall_market": maybe_col(df, "best_overall_market"),
        "best_overall_label": maybe_col(df, "best_overall_label"),
        "best_overall_edge": maybe_col(df, "best_overall_edge"),
        "best_overall_confidence": maybe_col(df, "best_overall_confidence"),
        "fort_knox_score": maybe_col(df, "fort_knox_score"),
        "market_snapshot_timestamp": maybe_col(df, "market_snapshot_timestamp"),
        "close_snapshot_timestamp": maybe_col(df, "close_snapshot_timestamp"),
        "market_books": maybe_col(df, "market_books"),
        "close_books": maybe_col(df, "close_books"),
        "source_projection_file": "game_projection_master",
    })
    return pd.concat([home, away], ignore_index=True)



def build_spread_rows(df: pd.DataFrame, ts: str) -> pd.DataFrame:
    home_line = pd.to_numeric(maybe_col(df, "market_spread_home"), errors="coerce")
    away_line = pd.to_numeric(maybe_col(df, "market_spread_away"), errors="coerce")
    home = pd.DataFrame({
        "ts": ts,
        "season": maybe_col(df, "season"),
        "week": maybe_col(df, "week"),
        "sport": "NFL",
        "league": "NFL",
        "market": "SPREADS",
        "side": maybe_col(df, "home_team"),
        "team_name": maybe_col(df, "home_team"),
        "line": home_line,
        "odds": -110.0,
        "p_win": maybe_col(df, "sim_cover_prob_home"),
        "edge": maybe_col(df, "home_spread_edge"),
        "clv": maybe_col(df, "home_spread_clv"),
        "close_line": maybe_col(df, "close_spread_home"),
        "home": maybe_col(df, "home_team"),
        "away": maybe_col(df, "away_team"),
        "game_id": maybe_col(df, "game_id"),
        "event_id": maybe_col(df, "event_id"),
        "game_date": maybe_col(df, "game_date"),
        "commence_time": maybe_col(df, "commence_time"),
        "best_overall_market": maybe_col(df, "best_overall_market"),
        "best_overall_label": maybe_col(df, "best_overall_label"),
        "best_overall_edge": maybe_col(df, "best_overall_edge"),
        "best_overall_confidence": maybe_col(df, "best_overall_confidence"),
        "fort_knox_score": maybe_col(df, "fort_knox_score"),
        "market_snapshot_timestamp": maybe_col(df, "market_snapshot_timestamp"),
        "close_snapshot_timestamp": maybe_col(df, "close_snapshot_timestamp"),
        "market_books": maybe_col(df, "market_books"),
        "close_books": maybe_col(df, "close_books"),
        "source_projection_file": "game_projection_master",
    })
    away = pd.DataFrame({
        "ts": ts,
        "season": maybe_col(df, "season"),
        "week": maybe_col(df, "week"),
        "sport": "NFL",
        "league": "NFL",
        "market": "SPREADS",
        "side": maybe_col(df, "away_team"),
        "team_name": maybe_col(df, "away_team"),
        "line": away_line,
        "odds": -110.0,
        "p_win": maybe_col(df, "sim_cover_prob_away"),
        "edge": maybe_col(df, "away_spread_edge"),
        "clv": maybe_col(df, "away_spread_clv"),
        "close_line": maybe_col(df, "close_spread_away"),
        "home": maybe_col(df, "home_team"),
        "away": maybe_col(df, "away_team"),
        "game_id": maybe_col(df, "game_id"),
        "event_id": maybe_col(df, "event_id"),
        "game_date": maybe_col(df, "game_date"),
        "commence_time": maybe_col(df, "commence_time"),
        "best_overall_market": maybe_col(df, "best_overall_market"),
        "best_overall_label": maybe_col(df, "best_overall_label"),
        "best_overall_edge": maybe_col(df, "best_overall_edge"),
        "best_overall_confidence": maybe_col(df, "best_overall_confidence"),
        "fort_knox_score": maybe_col(df, "fort_knox_score"),
        "market_snapshot_timestamp": maybe_col(df, "market_snapshot_timestamp"),
        "close_snapshot_timestamp": maybe_col(df, "close_snapshot_timestamp"),
        "market_books": maybe_col(df, "market_books"),
        "close_books": maybe_col(df, "close_books"),
        "source_projection_file": "game_projection_master",
    })
    return pd.concat([home, away], ignore_index=True)



def build_total_rows(df: pd.DataFrame, ts: str) -> pd.DataFrame:
    total_line = pd.to_numeric(maybe_col(df, "market_total"), errors="coerce")
    over = pd.DataFrame({
        "ts": ts,
        "season": maybe_col(df, "season"),
        "week": maybe_col(df, "week"),
        "sport": "NFL",
        "league": "NFL",
        "market": "TOTALS",
        "side": "Over",
        "team_name": "",
        "line": total_line,
        "odds": -110.0,
        "p_win": maybe_col(df, "sim_over_prob"),
        "edge": maybe_col(df, "over_edge"),
        "clv": maybe_col(df, "over_clv"),
        "close_line": maybe_col(df, "close_total"),
        "home": maybe_col(df, "home_team"),
        "away": maybe_col(df, "away_team"),
        "game_id": maybe_col(df, "game_id"),
        "event_id": maybe_col(df, "event_id"),
        "game_date": maybe_col(df, "game_date"),
        "commence_time": maybe_col(df, "commence_time"),
        "best_overall_market": maybe_col(df, "best_overall_market"),
        "best_overall_label": maybe_col(df, "best_overall_label"),
        "best_overall_edge": maybe_col(df, "best_overall_edge"),
        "best_overall_confidence": maybe_col(df, "best_overall_confidence"),
        "fort_knox_score": maybe_col(df, "fort_knox_score"),
        "market_snapshot_timestamp": maybe_col(df, "market_snapshot_timestamp"),
        "close_snapshot_timestamp": maybe_col(df, "close_snapshot_timestamp"),
        "market_books": maybe_col(df, "market_books"),
        "close_books": maybe_col(df, "close_books"),
        "source_projection_file": "game_projection_master",
    })
    under = pd.DataFrame({
        "ts": ts,
        "season": maybe_col(df, "season"),
        "week": maybe_col(df, "week"),
        "sport": "NFL",
        "league": "NFL",
        "market": "TOTALS",
        "side": "Under",
        "team_name": "",
        "line": total_line,
        "odds": -110.0,
        "p_win": maybe_col(df, "sim_under_prob"),
        "edge": maybe_col(df, "under_edge"),
        "clv": maybe_col(df, "under_clv"),
        "close_line": maybe_col(df, "close_total"),
        "home": maybe_col(df, "home_team"),
        "away": maybe_col(df, "away_team"),
        "game_id": maybe_col(df, "game_id"),
        "event_id": maybe_col(df, "event_id"),
        "game_date": maybe_col(df, "game_date"),
        "commence_time": maybe_col(df, "commence_time"),
        "best_overall_market": maybe_col(df, "best_overall_market"),
        "best_overall_label": maybe_col(df, "best_overall_label"),
        "best_overall_edge": maybe_col(df, "best_overall_edge"),
        "best_overall_confidence": maybe_col(df, "best_overall_confidence"),
        "fort_knox_score": maybe_col(df, "fort_knox_score"),
        "market_snapshot_timestamp": maybe_col(df, "market_snapshot_timestamp"),
        "close_snapshot_timestamp": maybe_col(df, "close_snapshot_timestamp"),
        "market_books": maybe_col(df, "market_books"),
        "close_books": maybe_col(df, "close_books"),
        "source_projection_file": "game_projection_master",
    })
    return pd.concat([over, under], ignore_index=True)



def build_standardized_edges(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.Timestamp.utcnow().isoformat()
    edges = pd.concat([
        build_moneyline_rows(df, ts),
        build_spread_rows(df, ts),
        build_total_rows(df, ts),
    ], ignore_index=True)

    edges["odds"] = pd.to_numeric(edges["odds"], errors="coerce")
    edges["line"] = pd.to_numeric(edges["line"], errors="coerce")
    edges["close_line"] = pd.to_numeric(edges["close_line"], errors="coerce")
    edges["p_win"] = pd.to_numeric(edges["p_win"], errors="coerce")
    edges["edge"] = pd.to_numeric(edges["edge"], errors="coerce")
    edges["clv"] = pd.to_numeric(edges["clv"], errors="coerce")
    edges["fort_knox_score"] = pd.to_numeric(edges["fort_knox_score"], errors="coerce")
    edges["best_overall_confidence"] = pd.to_numeric(edges["best_overall_confidence"], errors="coerce")

    edges["implied_prob"] = american_to_implied_prob(edges["odds"])
    edges["ev"] = expected_value_per_1(edges["p_win"], edges["odds"])
    edges["ev_pct"] = edges["ev"]

    edges["is_best_bet_for_game"] = (
        edges["best_overall_label"].astype(str).str.strip().str.lower()
        == edges["side"].astype(str).str.strip().str.lower()
        + np.where(
            edges["market"].eq("SPREADS"),
            " " + edges["line"].map(lambda x: f"{float(x):+0.1f}" if pd.notna(x) else ""),
            np.where(
                edges["market"].eq("TOTALS"),
                " " + edges["line"].map(lambda x: f"{float(x):0.1f}" if pd.notna(x) else ""),
                np.where(edges["market"].eq("H2H"), "", "")
            )
        )
    )

    market_rank = {"H2H": 1, "SPREADS": 2, "TOTALS": 3}
    edges["market_rank"] = edges["market"].map(market_rank)

    desired = [
        "ts", "season", "week", "sport", "league", "game_date", "commence_time",
        "game_id", "event_id", "home", "away", "market", "side", "team_name",
        "line", "odds", "implied_prob", "p_win", "edge", "ev", "ev_pct", "clv", "close_line",
        "fort_knox_score", "best_overall_confidence", "best_overall_market", "best_overall_label",
        "best_overall_edge", "is_best_bet_for_game", "market_snapshot_timestamp", "close_snapshot_timestamp",
        "market_books", "close_books", "source_projection_file", "market_rank",
    ]
    for c in desired:
        if c not in edges.columns:
            edges[c] = np.nan

    edges = edges[desired].sort_values(
        ["season", "week", "game_date", "home", "away", "market_rank", "edge"],
        ascending=[True, True, True, True, True, True, False],
    ).reset_index(drop=True)
    return edges


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

def save_outputs(edges: pd.DataFrame, root: Path, season: int, week: int) -> list[Path]:
    exports_dir = root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = exports_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []

    full_week = exports_dir / f"edges_standardized_{season}_w{week}.csv"
    edges.to_csv(full_week, index=False)
    out_paths.append(full_week)

    standard = exports_dir / "edges_standardized.csv"
    edges.to_csv(standard, index=False)
    out_paths.append(standard)

    master = exports_dir / "edges_master.csv"
    edges.to_csv(master, index=False)
    out_paths.append(master)

    generic = exports_dir / "edges.csv"
    edges.to_csv(generic, index=False)
    out_paths.append(generic)

    pick_explorer = exports_dir / "pick_explorer_edges.csv"
    edges.sort_values(["fort_knox_score", "edge", "ev"], ascending=False).to_csv(pick_explorer, index=False)
    out_paths.append(pick_explorer)

    parlay_builder = exports_dir / "parlay_builder_edges.csv"
    parlay = edges[(edges["ev"].fillna(-999) > 0) & (edges["p_win"].fillna(0) >= 0.50)].copy()
    parlay = parlay.sort_values(["p_win", "ev", "fort_knox_score"], ascending=False)
    parlay.to_csv(parlay_builder, index=False)
    out_paths.append(parlay_builder)

    edge_finder = exports_dir / "edge_finder_edges.csv"
    edge_scan = edges[(edges["edge"].fillna(-999) > 0) | (edges["ev"].fillna(-999) > 0)].copy()
    edge_scan = edge_scan.sort_values(["edge", "ev", "fort_knox_score"], ascending=False)
    edge_scan.to_csv(edge_finder, index=False)
    out_paths.append(edge_finder)

    canonical_week = canonical_dir / f"edges_standardized_{season}_w{week}.csv"
    edges.to_csv(canonical_week, index=False)
    out_paths.append(canonical_week)

    return out_paths


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build downstream edge outputs from game_projection_master.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else find_root(script_path)
    input_path = resolve_input(root, args.season, args.week, args.input)

    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False, encoding="utf-8-sig")
    if df.empty:
        raise SystemExit(f"Input file is empty: {input_path}")

    edges = build_standardized_edges(df)
    out_paths = save_outputs(edges, root, args.season, args.week)

    print(f"[OK] source: {input_path}")
    print(f"[OK] edge rows: {len(edges):,}")
    for p in out_paths:
        print(f"[OK] wrote {p}")
    print("[OK] sample:")
    preview_cols = [c for c in ["week", "away", "home", "market", "side", "line", "odds", "p_win", "edge", "ev", "fort_knox_score"] if c in edges.columns]
    print(edges[preview_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
