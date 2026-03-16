from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from generate_report_rebuild_full import resolve_paths, load_csv, enrich_edges, enrich_games, enrich_parlays


def main() -> None:
    parser = argparse.ArgumentParser(description="Stamp explicit labels and upgraded confidence fields into canonical week files.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--edition", type=str, default="monday")
    args = parser.parse_args()

    paths = resolve_paths(args.season, args.week, args.edition)
    games = load_csv(paths.games_path)
    edges = load_csv(paths.edges_path)
    markets = load_csv(paths.markets_path)
    scores = load_csv(paths.scores_path)
    parlays = load_csv(paths.parlay_path)

    if games.empty or edges.empty:
        raise SystemExit("Need at least the canonical games and edges CSVs to enrich the week files.")

    edges2 = enrich_edges(edges, games)
    games2 = enrich_games(games, markets, scores)
    parlays2 = enrich_parlays(parlays, edges2, games2)

    games_back = games.copy()
    for col in [
        "game_id", "game_label", "home_team", "away_team", "market_spread_raw", "market_total",
        "power_diff", "recent_form_diff", "offense_diff", "defense_diff", "qb_diff", "pace_diff",
        "home_field_pts", "raw_proj_margin", "shrunk_proj_margin", "raw_proj_total", "shrunk_proj_total",
        "proj_home_points", "proj_away_points", "median_home_points", "median_away_points",
    ]:
        if col in games2.columns:
            games_back[col] = games2[col]
    games_back.to_csv(paths.games_path, index=False)
    print(f"[OK] updated games: {paths.games_path}")

    edges_back = edges.copy()
    for col in ["game_id", "game_label", "market_norm", "play_label", "board_label", "confidence", "score", "p_win", "implied_prob", "edge_prob"]:
        if col in edges2.columns:
            edges_back[col] = edges2[col]
    edges_back.to_csv(paths.edges_path, index=False)
    print(f"[OK] updated edges: {paths.edges_path}")

    if not parlays.empty:
        parlays_back = parlays.copy()
        if "confidence" in parlays2.columns:
            parlays_back["confidence"] = parlays2["confidence"]
        for i in range(3):
            col = f"leg_{i+1}_label"
            vals = [legs[i] if len(legs) > i else "" for legs in parlays2.get("leg_labels", pd.Series([[]] * len(parlays2)))]
            parlays_back[col] = vals
        if "correlation_note" in parlays2.columns:
            parlays_back["correlation_note"] = parlays2["correlation_note"]
        parlays_back.to_csv(paths.parlay_path, index=False)
        print(f"[OK] updated parlays: {paths.parlay_path}")


if __name__ == "__main__":
    main()
