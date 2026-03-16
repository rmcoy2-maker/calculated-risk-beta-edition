from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from report_data import TEAM_ABBR, _read_csv, _read_excel, project_root


def _std_team(s: object) -> str:
    raw = str(s or "").strip().upper()
    return TEAM_ABBR.get(raw, raw)


def _norm_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["season"] = pd.to_numeric(out.get("season", out.get("Season")), errors="coerce")
    out["week"] = pd.to_numeric(out.get("week", out.get("Week")), errors="coerce")
    date_src = out.get("game_date")
    if date_src is None:
        date_src = out.get("date")
    if date_src is None:
        date_src = out.get("schedule_date")
    if date_src is None:
        date_src = out.get("Date")
    if date_src is None:
        date_src = pd.Series([pd.NA] * len(out), index=out.index)
    out["game_date"] = pd.to_datetime(date_src, errors="coerce").dt.strftime("%Y-%m-%d")
    away_src = out.get("away_team")
    if away_src is None:
        away_src = out.get("away")
    if away_src is None:
        away_src = out.get("AwayTeam")
    if away_src is None:
        away_src = pd.Series([pd.NA] * len(out), index=out.index)
    home_src = out.get("home_team")
    if home_src is None:
        home_src = out.get("home")
    if home_src is None:
        home_src = out.get("HomeTeam")
    if home_src is None:
        home_src = pd.Series([pd.NA] * len(out), index=out.index)
    out["away_team"] = away_src.map(_std_team)
    out["home_team"] = home_src.map(_std_team)
    if "game_id" not in out.columns:
        out["game_id"] = pd.NA
    missing_id = out["game_id"].isna() | (out["game_id"].astype(str).str.strip() == "")
    out.loc[missing_id, "game_id"] = (
        out.loc[missing_id, "game_date"].fillna("")
        + "_" + out.loc[missing_id, "away_team"].fillna("")
        + "_AT_" + out.loc[missing_id, "home_team"].fillna("")
    )
    cols = ["season", "week", "game_date", "game_id", "away_team", "home_team"]
    extra = [c for c in ["away_score", "home_score", "spread_close", "total_close", "ml_home", "ml_away", "stadium", "weather_temperature", "weather_wind_mph", "weather_detail"] if c in out.columns]
    return out[cols + extra].drop_duplicates(subset=["game_id"]).reset_index(drop=True)


def _norm_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    rename = {
        "ref": "book", "price": "odds", "_market_norm": "market", "selection": "side",
        "probability": "p_win", "win_prob": "p_win",
    }
    out = out.rename(columns=rename)
    keep = [c for c in ["season", "week", "game_id", "market", "side", "line", "odds", "p_win", "ev", "book", "parlay_proba", "legs", "parlay_score"] if c in out.columns]
    return out[keep].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical Calculated Risk weekly CSVs.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    root = project_root()
    out_dir = Path(args.out_dir) if args.out_dir else root / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    games = _read_csv(root / "games_master_template.csv")
    if games.empty:
        games = _read_excel(root / "games_master_with_ev_and_tiers.xlsx")
    edges = _read_csv(root / "parlay_scores.csv")
    markets = _read_csv(root / "lines_snapshots.csv")
    scores = _read_csv(root / "scores_normalized_std_maxaligned.csv")

    games = _norm_games(games)
    edges = _norm_edges(edges)
    markets = _norm_edges(markets)
    scores = _norm_games(scores)

    games = games[(games["season"] == args.season) & (games["week"] == args.week)] if not games.empty and "week" in games.columns else games.head(0)
    edges = edges[(pd.to_numeric(edges.get("season"), errors="coerce") == args.season) & (pd.to_numeric(edges.get("week"), errors="coerce") == args.week)] if not edges.empty and "week" in edges.columns else edges.head(0)
    markets = markets[(pd.to_numeric(markets.get("season"), errors="coerce") == args.season) & (pd.to_numeric(markets.get("week"), errors="coerce") == args.week)] if not markets.empty and "week" in markets.columns else markets.head(0)
    scores = scores[(scores["season"] == args.season) & (scores["week"] == args.week)] if not scores.empty and "week" in scores.columns else scores.head(0)

    stems = {
        "games": out_dir / f"cr_games_{args.season}_w{args.week}.csv",
        "edges": out_dir / f"cr_edges_{args.season}_w{args.week}.csv",
        "parlay_scores": out_dir / f"cr_parlay_scores_{args.season}_w{args.week}.csv",
        "markets": out_dir / f"cr_markets_{args.season}_w{args.week}.csv",
        "scores": out_dir / f"cr_scores_{args.season}_w{args.week}.csv",
    }

    games.to_csv(stems["games"], index=False)
    edges.to_csv(stems["edges"], index=False)
    edges.to_csv(stems["parlay_scores"], index=False)
    markets.to_csv(stems["markets"], index=False)
    scores.to_csv(stems["scores"], index=False)

    latest_names = {
        "cr_games_latest.csv": stems["games"],
        "cr_edges_latest.csv": stems["edges"],
        "cr_parlay_scores_latest.csv": stems["parlay_scores"],
        "cr_markets_latest.csv": stems["markets"],
        "cr_scores_latest.csv": stems["scores"],
    }
    for latest, src in latest_names.items():
        if src.exists():
            (out_dir / latest).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    for k, v in stems.items():
        print(f"[OK] {k}: {v}")


if __name__ == "__main__":
    main()
