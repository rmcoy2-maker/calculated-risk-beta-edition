import numpy as np, pandas as pd, joblib
from pathlib import Path
from core_engine.models.numpy_logreg import NumpyLogReg

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "data_scaffnew" / "db"
MODEL_DIR = ROOT / "models_store"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DB_DIR / "features.csv")

if "feat_rating_diff" in df.columns:
    X = df[["feat_rating_diff"]].to_numpy()
else:
    X = df.select_dtypes("number").to_numpy()

# simple target for demo: positive diff => 1 else 0
y = (X[:,0] > 0).astype(float)

mdl = NumpyLogReg().fit(X, y)
joblib.dump(mdl, MODEL_DIR / "game_lr.joblib")
print("[train_quick_numpy] wrote", MODEL_DIR / "game_lr.joblib")




