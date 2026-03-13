from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    """
    Add the repo root and serving_ui_recovered root to sys.path.
    Returns the detected repo root.
    """
    here = Path(__file__).resolve()

    candidates = [
        here.parents[2],  # .../edge-finder
        here.parents[1],  # .../serving_ui_recovered
        Path.cwd(),
    ]

    repo_root = None
    for p in candidates:
        if p and p.exists():
            if (p / "exports").exists() or p.name.lower() == "edge-finder":
                repo_root = p
                break

    if repo_root is None:
        repo_root = Path.cwd()

    path_additions = [
        repo_root,
        repo_root / "serving_ui_recovered",
        repo_root / "serving_ui_recovered" / "app",
        repo_root / "tools",
    ]

    for p in path_additions:
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)

    return repo_root