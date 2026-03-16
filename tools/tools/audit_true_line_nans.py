import pandas as pd

df = pd.read_csv("analysis_out/true_line/true_line_board.csv", low_memory=False)

cols = [
    "game_id", "Season", "Week", "selected_team", "profit", "actual_win",
    "market_odds", "market_prob", "close_prob", "model_prob", "cal_prob",
    "true_prob_shrunk_40", "edge_prob_shrunk_40", "model_edge_vs_close",
    "hybrid_score", "market_confirmation", "odds_bucket"
]

mask = (
    df["edge_prob_shrunk_40"].isna() |
    df["model_edge_vs_close"].isna() |
    df["market_prob"].isna()
)

bad = df.loc[mask, [c for c in cols if c in df.columns]].copy()

print("\nNaN audit counts:\n")
print(bad.isna().sum())

print("\nRows with missing critical fields:", len(bad))
print("\nSample rows:\n")
print(bad.head(50).to_string(index=False))

bad.to_csv("analysis_out/true_line_nan_audit.csv", index=False)
print("\nSaved: analysis_out/true_line_nan_audit.csv")