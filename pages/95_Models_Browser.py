from __future__ import annotations

from pathlib import Path
import json
import tempfile

import pandas as pd
import streamlit as st


st.set_page_config(page_title="95 Models Browser", page_icon="📈", layout="wide")

# -----------------------------
# Login guard
# -----------------------------
if not st.session_state.get("authenticated", False):
    st.switch_page("Home.py")
    st.stop()

st.sidebar.success(f"Logged in as {st.session_state.get('user', '')}")

# -----------------------------
# Paths
# -----------------------------
REPO = Path(__file__).resolve().parents[1]
MODELS_DIRS = [
    REPO / "models",
    REPO / "exports" / "models",
    REPO / "artifacts" / "models",
]

# writable temp area for cloud
TMP_DIR = Path(tempfile.gettempdir()) / "calculated_risk"
TMP_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_MARKER = TMP_DIR / "ACTIVE_MODEL.txt"


def current_model_version(default: str = "none") -> str:
    if ACTIVE_MARKER.exists():
        try:
            return ACTIVE_MARKER.read_text(encoding="utf-8").strip() or default
        except Exception:
            return default
    return default


def load_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_models() -> list[dict]:
    rows: list[dict] = []

    for base in MODELS_DIRS:
        if not base.exists():
            continue

        # Case 1: model folders
        for p in base.iterdir():
            if not p.is_dir():
                continue

            files = list(p.glob("*.joblib")) + list(p.glob("*.pkl"))
            if not files:
                continue

            meta = {}
            for meta_name in ("model.json", "metadata.json", "meta.json"):
                mp = p / meta_name
                if mp.exists():
                    meta = load_json_safe(mp)
                    break

            rows.append(
                {
                    "name": meta.get("name", p.name),
                    "version": meta.get("version", p.name),
                    "updated": pd.to_datetime(p.stat().st_mtime, unit="s"),
                    "notes": meta.get("notes", ""),
                    "path": str(p),
                    "kind": "folder",
                }
            )

        # Case 2: flat model files directly under models/
        flat_models = list(base.glob("*.joblib")) + list(base.glob("*.pkl"))
        if flat_models:
            meta_candidates = {}
            for meta_name in ("model.json", "metadata.json", "meta.json"):
                mp = base / meta_name
                if mp.exists():
                    meta_candidates = load_json_safe(mp)
                    break

            for mf in flat_models:
                rows.append(
                    {
                        "name": meta_candidates.get("name", mf.stem),
                        "version": meta_candidates.get("version", mf.stem),
                        "updated": pd.to_datetime(mf.stat().st_mtime, unit="s"),
                        "notes": meta_candidates.get("notes", ""),
                        "path": str(mf),
                        "kind": "file",
                    }
                )

    # dedupe on path
    seen = set()
    out = []
    for r in rows:
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        out.append(r)

    return sorted(out, key=lambda d: (str(d["name"]).lower(), d["updated"]), reverse=False)


def set_active_model(version: str) -> None:
    ACTIVE_MARKER.write_text(version, encoding="utf-8")


st.title("🧠 Models Browser")

cur = current_model_version()
st.caption(f"Current model version: **{cur}**")

models = discover_models()

if not models:
    st.warning(
        "No models found. Add model files under `models/`, `exports/models/`, or `artifacts/models/`."
    )
    st.stop()

df = pd.DataFrame(models)[["name", "version", "updated", "notes", "path", "kind"]]
st.dataframe(df, hide_index=True, use_container_width=True)

st.subheader("Activate")

choices = [f"{m['name']} — {m['version']}" for m in models]
default_idx = next((i for i, m in enumerate(models) if m["version"] == cur), 0)

col1, col2 = st.columns([3, 1])
with col1:
    picked = st.selectbox("Select a model to activate", choices, index=default_idx)
with col2:
    if st.button("Set Active", type="primary", use_container_width=True):
        sel_idx = choices.index(picked)
        ver = models[sel_idx]["version"]
        set_active_model(ver)
        st.success(f"Set active model to: {ver}")

with st.expander("🔎 Inspect selected model files", expanded=False):
    sel_idx = choices.index(picked)
    p = Path(models[sel_idx]["path"])

    if p.is_file():
        files = [p]
        root = p.parent
    else:
        files = [f for f in p.rglob("*") if f.is_file()]
        root = p

    if not files:
        st.info("No files found for this model.")
    else:
        table = pd.DataFrame(
            {
                "relpath": [str(f.relative_to(root)) for f in files],
                "size_kb": [round(f.stat().st_size / 1024, 2) for f in files],
            }
        ).sort_values("relpath")
        st.dataframe(table, hide_index=True, use_container_width=True)