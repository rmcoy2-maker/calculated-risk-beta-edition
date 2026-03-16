
# Build a toy feature matrix by merging features_raw with lines; creates feat_rating_diff.
import pandas as pd
from ..utils.paths import DB_DIR, ensure_dirs

def run():
    ensure_dirs()
    raw = pd.read_csv(DB_DIR / "features_raw.csv")
    lines = pd.read_csv(DB_DIR / "lines.csv")
    df = raw.merge(lines, on=["season","week","game_id"], how="left")
    df["feat_rating_diff"] = (df["team_rating"] - df["opp_rating"]).fillna(0.0)
    df.to_csv(DB_DIR / "features.csv", index=False)
    print(f"[build_features] Wrote {len(df)} rows -> {DB_DIR/'features.csv'}")

if __name__ == "__main__":
    run()





