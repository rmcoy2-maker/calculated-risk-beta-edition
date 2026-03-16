from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from core_engine.utils.paths import DB_DIR, ensure_dirs

@dataclass
class Bet:
    ts: str; game_id: str; market: str; ref: str; side: str
    line: float; odds: int; p_win: float; ev: float; tag: str

def _ensure_csv(path: Path):
    if not path.exists():
        cols = ["ts","game_id","market","ref","side","line","odds","p_win","ev","tag"]
        pd.DataFrame(columns=cols).to_csv(path, index=False)

def log_bet(row: dict, *, tag: str = "live", out_name: str | None = None):
    ensure_dirs()
    out_name = out_name or ("bets_log.csv" if tag=="live" else "simulated_bets_log.csv")
    out = DB_DIR / out_name
    _ensure_csv(out)
    ts = datetime.now(tz=timezone.utc).isoformat()
    b = Bet(ts=ts,
            game_id=str(row.get("game_id","")), market=str(row.get("market","")),
            ref=str(row.get("ref","")), side=str(row.get("side","")),
            line=float(row.get("line",0)), odds=int(row.get("odds",-110)),
            p_win=float(row.get("p_win",0.5)), ev=float(row.get("ev",0.0)), tag=tag)
    pd.DataFrame([asdict(b)]).to_csv(out, mode="a", header=False, index=False)
    return out





