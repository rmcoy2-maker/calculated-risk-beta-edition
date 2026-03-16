import pandas as pd
import numpy as np

INPUT_PATH = "analysis_out/true_line/true_line_board.csv"
OUTPUT_PATH = "analysis_out/true_line/true_line_board_governed.csv"

print("Loading board...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

# ----------------------------
# Convert numeric fields
# ----------------------------

numeric_cols = [
    "true_prob_shrunk_40",
    "market_prob",
    "close_prob",
    "market_decimal",
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ----------------------------
# Probability compression
# ----------------------------

def compress_prob(row):
    p = row["true_prob_shrunk_40"]
    m = row["market_prob"]

    if pd.isna(p):
        return p

    # Only compress favorites
    if m > 0.5:
        return 0.5 + 0.65 * (p - 0.5)

    return p

df["true_prob_governed"] = df.apply(compress_prob, axis=1)

# ----------------------------
# True line calculation
# ----------------------------

def prob_to_moneyline(p):

    if pd.isna(p):
        return np.nan

    if p >= 0.5:
        return -100 * (p / (1 - p))
    else:
        return 100 * ((1 - p) / p)

df["true_line_governed"] = df["true_prob_governed"].apply(prob_to_moneyline)

# ----------------------------
# Edge calculations
# ----------------------------

df["edge_prob_governed"] = df["true_prob_governed"] - df["market_prob"]

# Expected value
def calc_ev(row):

    p = row["true_prob_governed"]
    odds = row["market_decimal"]

    if pd.isna(p) or pd.isna(odds):
        return np.nan

    return (p * (odds - 1)) - (1 - p)

df["ev_per_1_governed"] = df.apply(calc_ev, axis=1)

# ----------------------------
# Edge tiers
# ----------------------------

def tier(edge):

    if pd.isna(edge):
        return "None"

    if edge < 0:
        return "Negative"

    if edge < 0.01:
        return "Tiny"

    if edge < 0.02:
        return "Small"

    if edge < 0.04:
        return "Medium"

    return "Large"

df["value_tier_governed"] = df["edge_prob_governed"].apply(tier)

# ----------------------------
# Quick diagnostics
# ----------------------------

summary = (
    df.groupby("Season")
      .agg(
        bets=("profit","size"),
        wins=("actual_win","sum"),
        hit_rate=("actual_win","mean"),
        avg_edge=("edge_prob_governed","mean")
      )
)

print("\nGovernor Summary\n")
print(summary)

# ----------------------------
# Save
# ----------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:")
print(OUTPUT_PATH)