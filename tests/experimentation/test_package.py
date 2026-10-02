"""Phase 9.1 — package-level sanity: imports, exports, and the
torch-free-import boundary for `experimentation`.
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
        "EnvironmentSnapshot",
        "ExperimentRecord",
        "ExperimentSource",
        "ExperimentStatus",
        "capture_environment",
        "record_experiment",
    }
    assert set(experimentation.__all__) == expected
    for name in experimentation.__all__:
        assert hasattr(experimentation, name)
    # Phase 9.1 is a foundation: no store, no query/comparison layer, no
    # MLflow integration, and nothing automatically recorded from an
    # existing Phase-7/8 entry point
    for forbidden in ("store", "query", "compare", "comparison", "mlflow"):
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
