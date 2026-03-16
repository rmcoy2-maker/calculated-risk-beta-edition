import numpy as np
import pandas as pd

def american_to_decimal(odds):
    if odds is None or pd.isna(odds):
        return np.nan
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 1.0 + o / 100.0
    if o < 0:
        return 1.0 + 100.0 / abs(o)
    return np.nan

def apply_clv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "clv" not in out.columns:
        out["clv"] = np.nan
    if "beat_closing" not in out.columns:
        out["beat_closing"] = np.nan
    return out