import pandas as pd, joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "data_scaffnew" / "db"
MODEL_DIR = ROOT / "models_store"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DB_DIR / "features.csv")
if "feat_rating_diff" in df.columns:
    X = df[["feat_rating_diff"]].fillna(0.0)
else:
    # fallback: use all numeric features
    X = df.select_dtypes("number").fillna(0.0)

# dummy target: positive diff → 1, else 0
y = (X.iloc[:,0] > 0).astype(int)

clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

joblib.dump(clf, MODEL_DIR / "game_lr.joblib")
print("[train_quick] wrote", MODEL_DIR / "game_lr.joblib")




