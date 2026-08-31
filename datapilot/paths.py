"""Canonical filesystem locations for DataPilot runtime artifacts.

These paths are *runtime* locations, not source. Everything under
``data/`` is git-ignored (see ``.gitignore``) and is created on demand by
the components that write to it. Nothing here is created at import time.
"""

from __future__ import annotations

from pathlib import Path

# Repo root = two levels up from this file (datapilot/paths.py -> datapilot -> root).
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_VERSIONS_DIR = DATA_DIR / "versions"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
