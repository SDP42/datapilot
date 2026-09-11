"""Phase 8.1 / 8.2 / 8.3 / 8.4 — package-level sanity: imports, exports,
no circular imports, and the PyTorch-optional boundary at the process
level.
"""

from __future__ import annotations

import subprocess
import sys

import dl_engine
from data_engine import modeling as data_engine_modeling


def test_dl_engine_imports_correctly():
    assert dl_engine is not None
    assert dl_engine.__doc__ is not None


def test_public_exports_are_intentional():
    expected = {
        "DL_ENGINE_VERSION",
        "DLDevice",
        "DLEvaluationResult",
        "DLLoss",
        "DLOptimizer",
        "DLTrainingConfig",
        "DLTrainingResult",
        "DLTrainingStatus",
        "DeviceResolution",
        "MLPActivation",
        "MLPArchitectureConfig",
        "TensorBatch",
        "TorchAvailability",
        "build_mlp",
        "evaluate_model",
        "is_torch_available",
        "resolve_device",
        "seed_everything",
        "to_tensors",
        "torch_availability",
        "train_model",
    }
    assert set(dl_engine.__all__) == expected
    # every declared export is actually resolvable as an attribute
    for name in dl_engine.__all__:
        assert hasattr(dl_engine, name)
    # still no reuse-breaking parallel evaluation contract — DLTrainingResult /
    # DLEvaluationResult are additive (see dl_engine.contracts), not a second
    # TrainingOutcome / EvaluationResults
    assert not any("outcome" in name.lower() for name in dl_engine.__all__)
    assert not any("trainingrun" in name.lower() for name in dl_engine.__all__)
    # Phase 8.5+ (model selection, pipeline integration) is not implemented —
    # no ranking / selection / experiment-tracking exports
    for forbidden in ("select", "rank", "experiment", "mlflow"):
        assert not any(forbidden in name.lower() for name in dl_engine.__all__)
    # no architecture beyond the MLP is implemented — MLP is the only one exported
    for arch_hint in ("cnn", "lstm", "transformer", "attention"):
        assert not any(arch_hint in name.lower() for name in dl_engine.__all__)


def test_no_circular_import_dl_engine_first():
    result = subprocess.run(
        [sys.executable, "-c", "import dl_engine; import data_engine.modeling"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_no_circular_import_modeling_first():
    result = subprocess.run(
        [sys.executable, "-c", "import data_engine.modeling; import dl_engine"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dl_engine_reuses_existing_model_family_not_a_parallel_one():
    # dl_engine.contracts.DLTrainingConfig.family is the existing Phase-7
    # ModelFamily enum, not a new/parallel vocabulary.
    assert (
        dl_engine.DLTrainingConfig.model_fields["family"].annotation
        is data_engine_modeling.ModelFamily
    )


def test_classical_modeling_remains_importable_independently():
    # Phase 0-7 must not have started depending on dl_engine at all.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import data_engine.modeling; import sys; assert 'dl_engine' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_importing_dl_engine_never_imports_torch_at_package_load_time():
    result = subprocess.run(
        [sys.executable, "-c", "import dl_engine; import sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_importing_modeling_package_never_requires_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import data_engine.modeling; import sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
