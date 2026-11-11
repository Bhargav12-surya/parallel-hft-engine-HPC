"""Remove only generated build/package directories inside this checkout."""
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
for name in ("build", "dist", ".pytest_cache", ".ruff_cache", "hftlab.egg-info"):
    target = (project_root / name).resolve()
    if target.parent != project_root:
        raise RuntimeError(f"Refusing to clean outside workspace: {target}")
    if target.is_dir():
        shutil.rmtree(target)
