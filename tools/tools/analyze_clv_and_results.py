import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:/Projects/calculated-risk-beta-edition")
EXPORTS_DIR = PROJECT_ROOT / "exports"
ANALYSIS_DIR = EXPORTS_DIR / "analysis"
SCORES_PATH = PROJECT_ROOT / "raw/2017-2025_scores.csv"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def as_series(value, index):
    if isinstance(value, pd.Series):
        return to_num(value)
    return pd.Series(value, index=index, dtype="float64")


def find_first_existing(df, cols, default=None):
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index)


def canonical_team(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    aliases = {
        "arizona cardinals": "cardinals",
        "atlanta falcons": "falcons",
        "baltimore ravens": "ravens",
        "buffalo bills": "bills",
        "carolina panthers": "panthers",
        "chicago bears": "bears",
        "cincinnati bengals": "bengals",
        "cleveland browns": "browns",
        "dallas cowboys": "cowboys",
        "denver broncos": "broncos",
        "detroit lions": "lions",
        "green bay packers": "packers",
        "houston texans": "texans",
        "indianapolis colts": "colts",
        "jacksonville jaguars": "jaguars",
        "kansas city chiefs": "chiefs",
        "las vegas raiders": "raiders",
        "los angeles chargers": "chargers",
        "los angeles rams": "rams",
        "miami dolphins": "dolphins",
        "minnesota vikings": "vikings",
        "new england patriots": "patriots",
        "new orleans saints": "saints",
        "new york giants": "giants",
        "new york jets": "jets",
        "philadelphia eagles": "eagles",
        "pittsburgh steelers": "steelers",
        "san francisco 49ers": "49ers",
        "seattle seahawks": "seahawks",
        "tampa bay buccaneers": "buccaneers",
        "tennessee titans": "titans",
        "washington commanders": "commanders",
        "washington football team": "commanders",
        "washington": "commanders",
    }
    return aliases.get(s, s)


def american_to_decimal(odds):
    if isinstance(odds, pd.Series):
        odds = to_num(odds)
        out = pd.Series(np.nan, index=odds.index)
        pos = odds > 0
        neg = odds < 0
        out.loc[pos] = 1 + (odds.loc[pos] / 100.0)
        out.loc[neg] = 1 + (100.0 / odds.loc[neg].abs())
        return out

    odds = float(to_num(odds)) if pd.notna(odds) else np.nan
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 1 + (odds / 100.0)
    if odds < 0:
        return 1 + (100.0 / abs(odds))
    return np.nan


def profit_per_unit(odds):
    dec = american_to_decimal(odds)
    return dec - 1.0


def expected_value_units(p_win, odds):
    if isinstance(p_win, pd.Series):
        p_win = to_num(p_win)
    else:
        p_win = float(to_num(p_win)) if pd.notna(p_win) else np.nan
    profit = profit_per_unit(odds)
    return p_win * profit - (1 - p_win)


def break_even_prob(odds):
    if isinstance(odds, pd.Series):
        odds = to_num(odds)
        out = pd.Series(np.nan, index=odds.index)
        neg = odds < 0
        pos = odds > 0
        out.loc[neg] = odds.loc[neg].abs() / (odds.loc[neg].abs() + 100.0)
        out.loc[pos] = 100.0 / (odds.loc[pos] + 100.0)
        return out

    odds = float(to_num(odds)) if pd.notna(odds) else np.nan
    if pd.isna(odds):
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return np.nan


def bucketize(series, bins, labels):
    s = to_num(series)
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)


def load_edges(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    rename_map = {
        "bet_type": "market",
        "bet_side": "side",
        "market_type": "market",
        "selection": "side",
        "win_prob": "p_win",
        "probability": "p_win",
        "model_prob": "p_win",
        "expected_value": "ev",
        "fort_knox": "fort_knox_score",
        "fk_score": "fort_knox_score",
        "home_team": "HomeTeam",
        "away_team": "AwayTeam",
        "season": "Season",
        "week": "Week",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["market", "side"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in edges file: {col}")

    for col in ["Season", "Week", "line", "close_line", "odds", "close_odds", "p_win", "edge", "ev", "clv", "fort_knox_score"]:
        if col in df.columns:
            df[col] = to_num(df[col])

    if "event_id" not in df.columns:
        df["event_id"] = find_first_existing(df, ["game_id", "EventID", "eventid"], default="").astype(str)
    else:
        df["event_id"] = df["event_id"].astype(str)

    df["HomeTeam_key"] = df["HomeTeam"].map(canonical_team) if "HomeTeam" in df.columns else ""
    df["AwayTeam_key"] = df["AwayTeam"].map(canonical_team) if "AwayTeam" in df.columns else ""
    df["market"] = df["market"].astype(str).str.strip().str.lower()
    df["side"] = df["side"].astype(str).str.strip().str.lower()
    return df


def load_scores(scores_path: Path) -> pd.DataFrame:
    s = pd.read_csv(scores_path, low_memory=False)
    s = s.rename(columns={
        "season": "Season",
        "week": "Week",
        "home_team": "HomeTeam",
        "away_team": "AwayTeam",
        "home_score": "home_score",
        "away_score": "away_score",
    })

    if "HomeTeam" not in s.columns:
        s["HomeTeam"] = find_first_existing(s, ["home_team", "Home", "HomeTeam"], default="")
    if "AwayTeam" not in s.columns:
        s["AwayTeam"] = find_first_existing(s, ["away_team", "Away", "AwayTeam"], default="")
    if "home_score" not in s.columns:
        s["home_score"] = to_num(find_first_existing(s, ["home_score", "HomeScore", "home_pts"], default=np.nan))
    if "away_score" not in s.columns:
        s["away_score"] = to_num(find_first_existing(s, ["away_score", "AwayScore", "away_pts"], default=np.nan))

    s["Season"] = to_num(find_first_existing(s, ["Season", "season"], default=np.nan)).astype("Int64")
    s["Week"] = to_num(find_first_existing(s, ["Week", "week"], default=np.nan)).astype("Int64")
    s["HomeTeam_key"] = s["HomeTeam"].map(canonical_team)
    s["AwayTeam_key"] = s["AwayTeam"].map(canonical_team)
    s = s.dropna(subset=["home_score", "away_score"]).copy()
    s["home_margin"] = s["home_score"] - s["away_score"]
    s["away_margin"] = -s["home_margin"]
    s["game_total"] = s["home_score"] + s["away_score"]
    return s[["Season", "Week", "HomeTeam", "AwayTeam", "HomeTeam_key", "AwayTeam_key", "home_score", "away_score", "home_margin", "away_margin", "game_total"]]


def merge_scores(edges: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    return edges.merge(
        scores,
        on=["Season", "Week", "HomeTeam_key", "AwayTeam_key"],
        how="left",
        suffixes=("", "_score"),
    )


def grade_bets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["grade"] = np.nan
    out["win_flag"] = np.nan
    out["push_flag"] = np.nan
    out["loss_flag"] = np.nan
    out["roi_units"] = np.nan
    out["beat_close_flag"] = np.nan

    market = out["market"].fillna("").astype(str).str.lower()
    side = out["side"].fillna("").astype(str).str.lower()
    line = as_series(out["line"] if "line" in out.columns else np.nan, out.index)
    close_line = as_series(out["close_line"] if "close_line" in out.columns else np.nan, out.index)
    odds = as_series(out["odds"] if "odds" in out.columns else np.nan, out.index)
    close_odds = as_series(out["close_odds"] if "close_odds" in out.columns else np.nan, out.index)

    home_margin = as_series(out["home_margin"] if "home_margin" in out.columns else np.nan, out.index)
    away_margin = as_series(out["away_margin"] if "away_margin" in out.columns else np.nan, out.index)
    game_total = as_series(out["game_total"] if "game_total" in out.columns else np.nan, out.index)

    profit = profit_per_unit(odds)

    is_ml = market.isin(["moneyline", "ml", "h2h"])
    is_spread = market.isin(["spread", "spreads", "ats"])
    is_total = market.isin(["total", "totals"])

    home_side = side.str.contains("home")
    away_side = side.str.contains("away")
    over_side = side.str.contains("over")
    under_side = side.str.contains("under")

    home_ml_win = is_ml & home_side & (home_margin > 0)
    away_ml_win = is_ml & away_side & (away_margin > 0)

    home_spread_res = home_margin + line
    away_spread_res = away_margin + line
    over_res = game_total - line
    under_res = line - game_total

    home_spread_win = is_spread & home_side & (home_spread_res > 0)
    home_spread_push = is_spread & home_side & (home_spread_res == 0)
    away_spread_win = is_spread & away_side & (away_spread_res > 0)
    away_spread_push = is_spread & away_side & (away_spread_res == 0)

    over_win = is_total & over_side & (over_res > 0)
    over_push = is_total & over_side & (over_res == 0)
    under_win = is_total & under_side & (under_res > 0)
    under_push = is_total & under_side & (under_res == 0)

    win_mask = home_ml_win | away_ml_win | home_spread_win | away_spread_win | over_win | under_win
    push_mask = home_spread_push | away_spread_push | over_push | under_push
    loss_mask = (~win_mask) & (~push_mask) & (is_ml | is_spread | is_total)

    out.loc[win_mask, "grade"] = "W"
    out.loc[push_mask, "grade"] = "P"
    out.loc[loss_mask, "grade"] = "L"

    out.loc[win_mask, "win_flag"] = 1.0
    out.loc[push_mask, "push_flag"] = 1.0
    out.loc[loss_mask, "loss_flag"] = 1.0

    out.loc[win_mask, "roi_units"] = profit.loc[win_mask]
    out.loc[push_mask, "roi_units"] = 0.0
    out.loc[loss_mask, "roi_units"] = -1.0

    be_current = break_even_prob(odds)
    be_close = break_even_prob(close_odds)

    ml_mask = is_ml & odds.notna() & close_odds.notna()
    ml_better_price = ((odds > 0) & (close_odds > 0) & (odds > close_odds)) | ((odds < 0) & (close_odds < 0) & (odds > close_odds))
    ml_better_price = ml_better_price | ((odds > 0) & (close_odds < 0))
    out.loc[ml_mask, "beat_close_flag"] = ml_better_price.loc[ml_mask].astype(float)

    directional_clv = pd.Series(np.nan, index=out.index)
    directional_clv.loc[home_side | over_side] = close_line.loc[home_side | over_side] - line.loc[home_side | over_side]
    directional_clv.loc[away_side | under_side] = line.loc[away_side | under_side] - close_line.loc[away_side | under_side]

    line_mask = (is_spread | is_total) & line.notna() & close_line.notna()
    out.loc[line_mask, "beat_close_flag"] = (directional_clv.loc[line_mask] > 0).astype(float)

    price_fallback_mask = (is_spread | is_total) & out["beat_close_flag"].isna() & odds.notna() & close_odds.notna()
    out.loc[price_fallback_mask, "beat_close_flag"] = (be_current.loc[price_fallback_mask] < be_close.loc[price_fallback_mask]).astype(float)

    if "ev" not in out.columns or out["ev"].isna().all():
        if "p_win" in out.columns:
            out["ev"] = expected_value_units(out["p_win"], odds)

    if "clv" not in out.columns or out["clv"].isna().all():
        out["clv"] = np.nan
        out.loc[ml_mask, "clv"] = be_close.loc[ml_mask] - be_current.loc[ml_mask]
        out.loc[line_mask, "clv"] = directional_clv.loc[line_mask]
        fb_mask = (is_spread | is_total) & out["clv"].isna() & odds.notna() & close_odds.notna()
        out.loc[fb_mask, "clv"] = be_close.loc[fb_mask] - be_current.loc[fb_mask]

    return out


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fort_knox_bucket"] = bucketize(
        out["fort_knox_score"] if "fort_knox_score" in out.columns else pd.Series(np.nan, index=out.index),
        bins=[-np.inf, 50, 60, 70, 80, 90, np.inf],
        labels=["<=50", "51-60", "61-70", "71-80", "81-90", "91+"],
    )
    out["prob_bucket"] = bucketize(
        out["p_win"] if "p_win" in out.columns else pd.Series(np.nan, index=out.index),
        bins=[-np.inf, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, np.inf],
        labels=["<=.50", ".51-.55", ".56-.60", ".61-.65", ".66-.70", ".71-.75", ".76+"],
    )
    out["edge_bucket"] = bucketize(
        out["edge"] if "edge" in out.columns else pd.Series(np.nan, index=out.index),
        bins=[-np.inf, 0, 0.02, 0.04, 0.06, 0.08, 0.10, np.inf],
        labels=["<=0", "0-.02", ".02-.04", ".04-.06", ".06-.08", ".08-.10", ".10+"],
    )
    return out


def summarize(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by_cols, dropna=False)
    summary = g.agg(
        n=("market", "size"),
        avg_clv=("clv", "mean"),
        median_clv=("clv", "median"),
        beat_close_pct=("beat_close_flag", "mean"),
        avg_edge=("edge", "mean"),
        avg_ev=("ev", "mean"),
        roi_units=("roi_units", "sum"),
        roi_per_bet=("roi_units", "mean"),
        win_pct=("win_flag", "mean"),
        push_pct=("push_flag", "mean"),
        avg_fk=("fort_knox_score", "mean"),
        avg_p_win=("p_win", "mean"),
    ).reset_index()
    return summary.sort_values(by=["n"], ascending=False)


def console_report(df: pd.DataFrame) -> str:
    total_n = len(df)
    avg_clv = df["clv"].mean() if "clv" in df.columns else np.nan
    beat_close = df["beat_close_flag"].mean() if "beat_close_flag" in df.columns else np.nan
    roi = df["roi_units"].mean() if "roi_units" in df.columns else np.nan
    win_pct = df["win_flag"].mean() if "win_flag" in df.columns else np.nan
    by_market = summarize(df, ["market"])

    lines = [
        "=== CLV + RESULTS AUDIT ===",
        f"Bets graded: {total_n:,}",
        f"Average CLV: {avg_clv:.4f}" if pd.notna(avg_clv) else "Average CLV: n/a",
        f"Beat close %: {beat_close:.2%}" if pd.notna(beat_close) else "Beat close %: n/a",
        f"ROI / bet: {roi:.4f}u" if pd.notna(roi) else "ROI / bet: n/a",
        f"Win %: {win_pct:.2%}" if pd.notna(win_pct) else "Win %: n/a",
        "",
        "By market:",
    ]
    for _, r in by_market.iterrows():
        lines.append(
            f"- {r['market']}: n={int(r['n'])}, avg_clv={r['avg_clv']:.4f}, beat_close={r['beat_close_pct']:.2%}, roi/bet={r['roi_per_bet']:.4f}u, win%={r['win_pct']:.2%}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze CLV, grading, ROI, and model bucket performance.")
    parser.add_argument("--input", default=str(EXPORTS_DIR / "edges_master.csv"))
    parser.add_argument("--scores", default=str(SCORES_PATH))
    parser.add_argument("--outdir", default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    edges = load_edges(Path(args.input))
    scores = load_scores(Path(args.scores))
    merged = merge_scores(edges, scores)
    graded = grade_bets(merged)
    graded = graded.dropna(subset=["grade"]).copy()
    graded = add_buckets(graded)

    graded.to_csv(outdir / "graded_bets_master.csv", index=False)
    summarize(graded, ["market"]).to_csv(outdir / "clv_summary_by_market.csv", index=False)
    summarize(graded, ["Season", "Week"]).to_csv(outdir / "clv_summary_by_week.csv", index=False)
    summarize(graded, ["fort_knox_bucket"]).to_csv(outdir / "clv_summary_by_fort_knox_bucket.csv", index=False)
    summarize(graded, ["prob_bucket"]).to_csv(outdir / "clv_summary_by_prob_bucket.csv", index=False)
    summarize(graded, ["edge_bucket"]).to_csv(outdir / "clv_summary_by_edge_bucket.csv", index=False)
    summarize(graded, ["market", "fort_knox_bucket"]).to_csv(outdir / "roi_summary_by_market_and_fort_knox.csv", index=False)

    report = console_report(graded)
    print(report)
    (outdir / "audit_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
