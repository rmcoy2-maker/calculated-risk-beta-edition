import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:/Projects/calculated-risk-beta-edition")
EXPORTS_DIR = PROJECT_ROOT / "exports"
ANALYSIS_DIR = EXPORTS_DIR / "analysis"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
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
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in ["edge", "p_win", "ev", "clv", "fort_knox_score", "line", "close_line", "odds", "close_odds"]:
        if col in df.columns:
            df[col] = to_num(df[col])
    df["market"] = df["market"].astype(str).str.lower().str.strip()
    df["side"] = df["side"].astype(str).str.lower().str.strip()
    return df


def load_summary_optional(path: Path, key_cols: list[str]) -> pd.DataFrame:
    metric_cols = ["avg_clv", "beat_close_pct", "roi_per_bet", "win_pct"]
    if not path.exists():
        return pd.DataFrame(columns=key_cols + metric_cols)
    df = pd.read_csv(path)
    for col in metric_cols:
        if col in df.columns:
            df[col] = to_num(df[col])
    return df


def add_prob_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prob_bucket"] = pd.cut(
        to_num(out.get("p_win", np.nan)),
        bins=[-np.inf, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, np.inf],
        labels=["<=.50", ".51-.55", ".56-.60", ".61-.65", ".66-.70", ".71-.75", ".76+"],
        include_lowest=True,
    )
    return out


def add_fort_knox_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fort_knox_bucket"] = pd.cut(
        to_num(out.get("fort_knox_score", np.nan)),
        bins=[-np.inf, 50, 60, 70, 80, 90, np.inf],
        labels=["<=50", "51-60", "61-70", "71-80", "81-90", "91+"],
        include_lowest=True,
    )
    return out


def add_edge_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["edge_bucket"] = pd.cut(
        to_num(out.get("edge", np.nan)),
        bins=[-np.inf, 0, 0.02, 0.04, 0.06, 0.08, 0.10, np.inf],
        labels=["<=0", "0-.02", ".02-.04", ".04-.06", ".06-.08", ".08-.10", ".10+"],
        include_lowest=True,
    )
    return out


def apply_rule_logic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rule_block_reason"] = ""

    min_edge = out["market"].map({"moneyline": 0.015, "ml": 0.015, "h2h": 0.015, "spread": 0.025, "spreads": 0.025, "ats": 0.025, "total": 0.025, "totals": 0.025}).fillna(0.02)
    min_prob = out["market"].map({"moneyline": 0.52, "ml": 0.52, "h2h": 0.52, "spread": 0.535, "spreads": 0.535, "ats": 0.535, "total": 0.535, "totals": 0.535}).fillna(0.53)
    min_fk = out["market"].map({"moneyline": 60, "ml": 60, "h2h": 60, "spread": 65, "spreads": 65, "ats": 65, "total": 65, "totals": 65}).fillna(62)

    out["passes_edge"] = to_num(out.get("edge", np.nan)) >= min_edge
    out["passes_prob"] = to_num(out.get("p_win", np.nan)) >= min_prob
    out["passes_fk"] = to_num(out.get("fort_knox_score", np.nan)) >= min_fk
    out["passes_ev"] = to_num(out.get("ev", np.nan)) > 0
    out["passes_clv"] = to_num(out.get("historical_avg_clv", np.nan)).fillna(0) >= 0
    out["passes_roi"] = to_num(out.get("historical_roi_per_bet", np.nan)).fillna(0) >= 0
    out["passes_beat_close"] = to_num(out.get("historical_beat_close_pct", np.nan)).fillna(0.5) >= 0.50

    for mask, reason in [
        (~out["passes_edge"], "edge"),
        (~out["passes_prob"], "prob"),
        (~out["passes_fk"], "fort_knox"),
        (~out["passes_ev"], "ev"),
        (~out["passes_clv"], "hist_clv"),
        (~out["passes_roi"], "hist_roi"),
        (~out["passes_beat_close"], "beat_close"),
    ]:
        out.loc[mask & out["rule_block_reason"].eq(""), "rule_block_reason"] = reason

    strong_hist = (
        (to_num(out.get("historical_roi_per_bet", np.nan)).fillna(-999) >= 0.03)
        & (to_num(out.get("historical_beat_close_pct", np.nan)).fillna(0) >= 0.54)
        & (to_num(out.get("historical_avg_clv", np.nan)).fillna(-999) > 0)
    )
    medium_hist = (
        (to_num(out.get("historical_roi_per_bet", np.nan)).fillna(-999) >= 0.00)
        & (to_num(out.get("historical_beat_close_pct", np.nan)).fillna(0) >= 0.50)
    )

    elite_now = (
        (to_num(out.get("edge", np.nan)) >= 0.06)
        & (to_num(out.get("p_win", np.nan)) >= 0.58)
        & (to_num(out.get("fort_knox_score", np.nan)) >= 80)
    )
    strong_now = (
        (to_num(out.get("edge", np.nan)) >= 0.04)
        & (to_num(out.get("p_win", np.nan)) >= 0.55)
        & (to_num(out.get("fort_knox_score", np.nan)) >= 70)
    )
    playable_now = (
        (to_num(out.get("edge", np.nan)) >= 0.02)
        & (to_num(out.get("p_win", np.nan)) >= 0.53)
        & (to_num(out.get("fort_knox_score", np.nan)) >= 60)
    )

    out["play_status"] = np.where(out["rule_block_reason"].eq(""), "PLAY", "PASS")
    out["tier"] = "PASS"
    out.loc[out["play_status"].eq("PLAY") & playable_now & medium_hist, "tier"] = "C"
    out.loc[out["play_status"].eq("PLAY") & strong_now & medium_hist, "tier"] = "B"
    out.loc[out["play_status"].eq("PLAY") & elite_now & strong_hist, "tier"] = "A"
    out.loc[out["play_status"].eq("PLAY") & out["tier"].eq("PASS"), "tier"] = "WATCH"

    out["stake_units"] = 0.0
    out.loc[out["tier"].eq("C"), "stake_units"] = 0.50
    out.loc[out["tier"].eq("B"), "stake_units"] = 1.00
    out.loc[out["tier"].eq("A"), "stake_units"] = 1.50
    return out


def attach_historical_context(edges: pd.DataFrame, prob_summary: pd.DataFrame, fk_summary: pd.DataFrame, market_summary: pd.DataFrame) -> pd.DataFrame:
    out = add_prob_bucket(edges)
    out = add_fort_knox_bucket(out)
    out = add_edge_bucket(out)

    prob_ctx = prob_summary[[c for c in ["prob_bucket", "avg_clv", "beat_close_pct", "roi_per_bet", "win_pct"] if c in prob_summary.columns]].rename(
        columns={
            "avg_clv": "historical_prob_avg_clv",
            "beat_close_pct": "historical_prob_beat_close_pct",
            "roi_per_bet": "historical_prob_roi_per_bet",
            "win_pct": "historical_prob_win_pct",
        }
    )
    fk_ctx = fk_summary[[c for c in ["fort_knox_bucket", "avg_clv", "beat_close_pct", "roi_per_bet", "win_pct"] if c in fk_summary.columns]].rename(
        columns={
            "avg_clv": "historical_fk_avg_clv",
            "beat_close_pct": "historical_fk_beat_close_pct",
            "roi_per_bet": "historical_fk_roi_per_bet",
            "win_pct": "historical_fk_win_pct",
        }
    )
    market_ctx = market_summary[[c for c in ["market", "avg_clv", "beat_close_pct", "roi_per_bet", "win_pct"] if c in market_summary.columns]].rename(
        columns={
            "avg_clv": "historical_market_avg_clv",
            "beat_close_pct": "historical_market_beat_close_pct",
            "roi_per_bet": "historical_market_roi_per_bet",
            "win_pct": "historical_market_win_pct",
        }
    )

    if not prob_ctx.empty:
        out = out.merge(prob_ctx, on="prob_bucket", how="left")
    if not fk_ctx.empty:
        out = out.merge(fk_ctx, on="fort_knox_bucket", how="left")
    if not market_ctx.empty:
        out = out.merge(market_ctx, on="market", how="left")

    hist_clv_cols = [c for c in ["historical_prob_avg_clv", "historical_fk_avg_clv", "historical_market_avg_clv"] if c in out.columns]
    hist_bc_cols = [c for c in ["historical_prob_beat_close_pct", "historical_fk_beat_close_pct", "historical_market_beat_close_pct"] if c in out.columns]
    hist_roi_cols = [c for c in ["historical_prob_roi_per_bet", "historical_fk_roi_per_bet", "historical_market_roi_per_bet"] if c in out.columns]
    hist_win_cols = [c for c in ["historical_prob_win_pct", "historical_fk_win_pct", "historical_market_win_pct"] if c in out.columns]

    out["historical_avg_clv"] = out[hist_clv_cols].mean(axis=1) if hist_clv_cols else np.nan
    out["historical_beat_close_pct"] = out[hist_bc_cols].mean(axis=1) if hist_bc_cols else np.nan
    out["historical_roi_per_bet"] = out[hist_roi_cols].mean(axis=1) if hist_roi_cols else np.nan
    out["historical_win_pct"] = out[hist_win_cols].mean(axis=1) if hist_win_cols else np.nan
    return out


def export_outputs(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "selection_rules_master.csv", index=False)
    df[df["play_status"].eq("PLAY")].to_csv(outdir / "selection_rules_playable.csv", index=False)
    df[df["tier"].eq("A")].to_csv(outdir / "selection_rules_tier_a.csv", index=False)
    df[df["tier"].eq("B")].to_csv(outdir / "selection_rules_tier_b.csv", index=False)
    df[df["tier"].eq("C")].to_csv(outdir / "selection_rules_tier_c.csv", index=False)
    df[df["tier"].eq("WATCH")].to_csv(outdir / "selection_rules_watchlist.csv", index=False)


def print_summary(df: pd.DataFrame) -> None:
    print("=== SELECTION RULES SUMMARY ===")
    print(f"Rows evaluated: {len(df):,}")
    playable = df[df["play_status"].eq("PLAY")]
    print(f"Playable: {len(playable):,}")
    for tier in ["A", "B", "C", "WATCH", "PASS"]:
        print(f"{tier}: {(df['tier'] == tier).sum():,}")


def main():
    parser = argparse.ArgumentParser(description="Build rules-aware betting card outputs from current edges and historical summaries.")
    parser.add_argument("--edges", default=str(EXPORTS_DIR / "edges_master.csv"))
    parser.add_argument("--analysis-dir", default=str(ANALYSIS_DIR))
    parser.add_argument("--outdir", default=str(ANALYSIS_DIR / "selection_rules"))
    args = parser.parse_args()

    edges = load_edges(Path(args.edges))
    analysis_dir = Path(args.analysis_dir)
    prob_summary = load_summary_optional(analysis_dir / "clv_summary_by_prob_bucket.csv", ["prob_bucket"])
    fk_summary = load_summary_optional(analysis_dir / "clv_summary_by_fort_knox_bucket.csv", ["fort_knox_bucket"])
    market_summary = load_summary_optional(analysis_dir / "clv_summary_by_market.csv", ["market"])

    enriched = attach_historical_context(edges, prob_summary, fk_summary, market_summary)
    ruled = apply_rule_logic(enriched)
    export_outputs(ruled, Path(args.outdir))
    print_summary(ruled)

    if prob_summary.empty or fk_summary.empty or market_summary.empty:
        print("Note: one or more historical summary files were missing, so rules fell back to whatever historical context was available.")


if __name__ == "__main__":
    main()
