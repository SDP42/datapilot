"""Guard: the Phase 0-7 stabilization pass introduced no deferred-phase stack.

The only modeling dependency currently justified is ``scikit-learn``.
Deep learning, hyperparameter-search, boosting, explainability, tracking,
backend, and database stacks belong to later phases and must not appear
in the data engine's imports or the project's declared dependencies.
"""

from __future__ import annotations

import ast
import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ENGINE = REPO_ROOT / "data_engine"

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
    }
    assert _BANNED_IMPORT_ROOTS.isdisjoint(declared)


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
