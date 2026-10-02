"""Phase 9.1 / 9.2 / 9.3 / 9.4 — package-level sanity: imports, exports,
and the torch-/mlflow-free-import boundaries for `experimentation`.
"""

from __future__ import annotations

import subprocess
import sys

import experimentation


def test_experimentation_imports_correctly():
    assert experimentation is not None
    assert experimentation.__doc__ is not None


def test_public_exports_are_intentional():
    expected = {
        "EXPERIMENTATION_ENGINE_VERSION",
        "DuplicateExperimentError",
        "EnvironmentSnapshot",
        "ExperimentComparisonEntry",
        "ExperimentComparisonResult",
        "ExperimentNotFoundError",
        "ExperimentRecord",
        "ExperimentSource",
        "ExperimentStatus",
        "ExperimentStore",
        "ExperimentStoreError",
        "MLflowAvailability",
        "MLflowLogResult",
        "capture_environment",
        "compare_experiments",
        "is_mlflow_available",
        "log_experiment_to_mlflow",
        "mlflow_availability",
        "record_experiment",
    }
    assert set(experimentation.__all__) == expected
    for name in experimentation.__all__:
        assert hasattr(experimentation, name)
    # Phase 9 (0 through 9.4) is complete: no database-backed store, no
    # automatic recording from an existing Phase-7/8 entry point, no
    # model-registry integration, no experiment recommendation
    for forbidden in ("database", "recommend", "registry", "automatic"):
        assert not any(forbidden in name.lower() for name in experimentation.__all__)


def test_importing_experimentation_never_imports_torch_at_package_load_time():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experimentation; import sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_importing_experimentation_does_not_affect_dl_engine_torch_boundary():
    # experimentation imports dl_engine (it nests DLModelingResult /
    # DLSelectionResult) — confirm that chain still never imports torch.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experimentation; import dl_engine; import sys; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_importing_experimentation_never_imports_mlflow_at_package_load_time():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import experimentation; import sys; assert 'mlflow' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
