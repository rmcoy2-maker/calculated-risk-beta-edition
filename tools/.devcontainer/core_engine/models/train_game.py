
# Train a tiny logistic model (placehnewer) to predict home_cover probability.
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from ..utils.paths import DB_DIR, MODEL_DIR, ensure_dirs

def run():
    ensure_dirs()
    df = pd.read_csv(DB_DIR / "features.csv")
    y = (df["home_cover"].fillna(0)).astype(int)
    X = df[["feat_rating_diff"]].fillna(0.0)
    model = LogisticRegression()
    model.fit(X, y)
    out = MODEL_DIR / "game_lr.joblib"
    joblib.dump(model, out)
    print(f"[train_game] Saved model -> {out}")

if __name__ == "__main__":
    run()





