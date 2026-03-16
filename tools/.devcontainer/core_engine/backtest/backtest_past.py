import pandas as pd
from core_engine.utils.paths import DB_DIR, ensure_dirs
from core_engine.utils.bet_logger import log_bet
from serving_ui.serving.score_week import score

def run_backtest(seasons, weeks=None, edge_threshnew=0.00):
    ensure_dirs()
    feats = pd.read_csv(DB_DIR / "features.csv")
    total=0
    for season in seasons:
        wks = sorted(feats.loc[feats["season"]==season, "week"].unique())
        if weeks: wks = [w for w in wks if w in weeks]
        for wk in wks:
            edges = score(season, wk, edge_threshnew=edge_threshnew)
            for _, r in edges.iterrows():
                log_bet({
                    "game_id": r.get("game_id"),
                    "market": r.get("market","spread"),
                    "ref": f"{season}W{wk}",
                    "side": r.get("side","HOME"),
                    "line": float(r.get("line",0)),
                    "odds": int(r.get("odds",-110)),
                    "p_win": float(r.get("p_win",0.5)),
                    "ev": float(r.get("ev",0.0)),
                }, tag="simulated")
                total+=1
    print(f"[backtest] wrote {total} simulated bets")

if __name__ == "__main__":
    run_backtest([2024])





