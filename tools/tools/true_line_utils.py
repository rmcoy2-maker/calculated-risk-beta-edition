import numpy as np
import pandas as pd


def american_to_implied_prob(odds):
    if pd.isna(odds):
        return np.nan
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return abs(o) / (abs(o) + 100.0)
    return np.nan


def prob_to_american(p):
    if pd.isna(p):
        return np.nan
    p = float(p)

    if p <= 0 or p >= 1:
        return np.nan

    if p > 0.5:
        return -100 * p / (1 - p)
    else:
        return 100 * (1 - p) / p


def select_true_prob(row, cal_col="cal_prob", raw_col="Model Prob"):
    cal = row.get(cal_col, np.nan)
    raw = row.get(raw_col, np.nan)

    if pd.notna(cal):
        return cal
    return raw


def assign_value_tier(edge):
    if pd.isna(edge):
        return "None"

    if edge >= 0.05:
        return "Elite"
    if edge >= 0.03:
        return "Strong"
    if edge >= 0.015:
        return "Playable"
    if edge >= 0.005:
        return "Thin"
    return "Negative"


def assign_governor(edge):
    if pd.isna(edge):
        return "None"

    if edge >= 0.03:
        return "Tight"
    if edge >= 0.02:
        return "Normal"
    if edge >= 0.01:
        return "Loose"
    return "Block"


def compute_true_line(df, prob_col="Model Prob", odds_col="American"):

    out = df.copy()

    out["market_prob"] = out[odds_col].apply(american_to_implied_prob)

    out["true_prob"] = out[prob_col]

    out["true_line"] = out["true_prob"].apply(prob_to_american)

    out["edge_prob"] = out["true_prob"] - out["market_prob"]

    out["value_tier"] = out["edge_prob"].apply(assign_value_tier)

    out["governor"] = out["edge_prob"].apply(assign_governor)

    return out