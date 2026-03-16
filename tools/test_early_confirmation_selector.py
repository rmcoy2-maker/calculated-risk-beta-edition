import pandas as pd

PATH = r"C:\Projects\calculated-risk-beta-edition\exports\fort_knox_market_joined_moneyline.csv"

df = pd.read_csv(PATH)

df["model_edge_vs_open"] = df["model_prob"] - df["open_prob"]
df["model_edge_vs_mid"] = df["model_prob"] - df["mid_prob"]
df["model_edge_vs_close"] = df["model_prob"] - df["close_prob"]
df["hit"] = (df["profit"] > 0).astype(int)

# Early-confirmed selector:
# model likes it at open
# market moves toward it by mid
# still has edge at close
sel = df[
    (df["model_prob"] >= 0.70) &
    (df["model_edge_vs_open"] >= 0.03) &
    (df["open_to_mid_prob_move"] > 0) &
    (df["model_edge_vs_close"] > 0)
].copy()

print("bets:", len(sel))
if len(sel):
    print("hit rate:", sel["hit"].mean())
    print("ROI:", sel["profit"].sum() / len(sel))
    print("avg edge vs open:", sel["model_edge_vs_open"].mean())
    print("avg edge vs close:", sel["model_edge_vs_close"].mean())

print("\nBy season:")
print(sel.groupby("Season")["hit"].agg(["count", "mean"]))
print("\nSeason ROI:")
print(sel.groupby("Season")["profit"].mean())