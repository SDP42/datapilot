"""Guard: `data_engine` itself never imports a later-phase stack.

The only modeling dependency `data_engine` ever needs is ``scikit-learn``
(Phase 7.4). Deep learning (Phase 8), explainability (Phase 10),
experiment tracking (Phase 9), and the AI/backend stack (Phase 11/13)
all belong to other packages and must never appear in `data_engine`'s
own imports — the deterministic data engine stays independently usable
without any of them installed, exactly as every later phase's own
"every Phase 0-7 capability works without X installed" test already
verifies from the opposite direction (e.g. `tests/dl_engine/test_package.py`).

This module does **not** assert the *project's* full declared-dependency
set is minimal forever — Phase 13 legitimately added ``fastapi`` /
``sqlalchemy`` / ``uvicorn`` / ``python-multipart`` / ``pydantic-settings``
as real, unconditional dependencies of the ``backend`` package (see
`docs/decisions.md`); ``torch`` / ``mlflow`` / ``shap`` / ``anthropic`` /
``duckdb`` / ``psycopg2-binary`` remain correctly scoped to their own
optional extras, never promoted to the base dependency list.
"""

from __future__ import annotations

import ast
import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ENGINE = REPO_ROOT / "data_engine"

# Never acceptable inside data_engine's own imports, regardless of
# whether the project declares it as a dependency elsewhere.
_BANNED_IMPORT_ROOTS = frozenset(
    {
        "mlflow",
        "optuna",
        "xgboost",
        "lightgbm",
        "catboost",
        "torch",
        "tensorflow",
        "keras",
        "statsmodels",
        "prophet",
        "sktime",
        "pmdarima",
        "tbats",
        "darts",
        "shap",
        "fastapi",
        "flask",
        "django",
        "sqlalchemy",
        "psycopg2",
        "duckdb",
        "anthropic",
        "openai",
        "langchain",
    }
)

# fastapi / sqlalchemy are the two entries in _BANNED_IMPORT_ROOTS that
# Phase 13 legitimately promoted to the project's base `dependencies` —
# still banned from data_engine's own imports (above), but no longer
# expected to be absent from the project's declared dependency set.
_STILL_DEFERRED_FROM_BASE_DEPENDENCIES = _BANNED_IMPORT_ROOTS - {"fastapi", "sqlalchemy"}


def _iter_imported_roots(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_no_banned_imports_anywhere_in_data_engine():
    offenders: dict[str, list[str]] = {}
    for py_file in sorted(DATA_ENGINE.rglob("*.py")):
        hits = _BANNED_IMPORT_ROOTS.intersection(_iter_imported_roots(py_file))
        if hits:
            offenders[str(py_file.relative_to(REPO_ROOT))] = sorted(hits)
    assert not offenders, f"deferred-phase stack imported: {offenders}"


def test_declared_runtime_dependencies_are_the_expected_set():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        dep.split(">")[0].split("=")[0].split("<")[0].split("[")[0].strip().lower()
        for dep in pyproject["project"]["dependencies"]
    }
    assert declared == {
        "pandas",
        "numpy",
        "scipy",
        "pydantic",
        "pyyaml",
        "matplotlib",
        "plotly",
        "scikit-learn",
        "joblib",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "sqlalchemy",
        "pydantic-settings",
        "pyjwt",
        "openpyxl",
    }
    assert _STILL_DEFERRED_FROM_BASE_DEPENDENCIES.isdisjoint(declared)


def test_modeling_only_learning_dependency_is_sklearn():
    modeling_dir = DATA_ENGINE / "modeling"
    roots: set[str] = set()
    for py_file in modeling_dir.rglob("*.py"):
        roots.update(_iter_imported_roots(py_file))
    learning_libs = {
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "torch",
        "tensorflow",
        "keras",
        "statsmodels",
        "prophet",
    }
    assert roots & learning_libs == {"sklearn"}
