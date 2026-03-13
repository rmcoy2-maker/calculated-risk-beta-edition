from __future__ import annotations
from pathlib import Path
import pandas as pd
import datetime as dt
from core_engine.utils.paths import EXPORTS_DIR

BANKROLL_CSV = EXPORTS_DIR / "bankroll.csv"

def _ensure_exports():
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

def init_if_missing(start_amount: float = 1500.0) -> float:
    _ensure_exports()
    if not BANKROLL_CSV.exists():
        pd.DataFrame([{"date": dt.date.today().isoformat(), "bankroll": float(start_amount)}]).to_csv(BANKROLL_CSV, index=False)
        return float(start_amount)
    return current()

def current() -> float:
    _ensure_exports()
    if not BANKROLL_CSV.exists():
        return init_if_missing()
    df = pd.read_csv(BANKROLL_CSV)
    if df.empty or "bankroll" not in df.columns:
        return init_if_missing()
    try:
        return float(df.iloc[-1]["bankroll"])
    except Exception:
        return init_if_missing()

def append_entry(new_value: float) -> None:
    _ensure_exports()
    base = pd.read_csv(BANKROLL_CSV) if BANKROLL_CSV.exists() else pd.DataFrame(columns=["date","bankroll"])
    out = pd.concat([base, pd.DataFrame([{"date": dt.date.today().isoformat(), "bankroll": float(new_value)}])], ignore_index=True)
    out.to_csv(BANKROLL_CSV, index=False)

def adjust(delta: float) -> float:
    cur = current()
    new_val = cur + float(delta)
    append_entry(new_val)
    return new_val

def set_to(amount: float) -> float:
    append_entry(float(amount))
    return float(amount)




