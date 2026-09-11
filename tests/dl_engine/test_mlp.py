"""Phase 8.3 — MLP model construction (`dl_engine.mlp.build_mlp`).

Every test here needs a real `torch.nn.Module`, so each starts with
`pytest.importorskip("torch")` and skips cleanly when PyTorch is not
installed.
"""

from __future__ import annotations

import pytest

from data_engine.problem_understanding import TaskType
from dl_engine.architectures import MLPActivation, MLPArchitectureConfig
from dl_engine.mlp import build_mlp
from dl_engine.runtime import seed_everything


def _arch(**overrides: object) -> MLPArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_features": 4,
        "output_dim": 1,
        "hidden_layer_sizes": [8],
    }
    defaults.update(overrides)
    return MLPArchitectureConfig.model_validate(defaults)


def test_build_mlp_raises_runtime_error_when_torch_missing():
    def _raise_import_error(name: str):
        raise ImportError(f"No module named '{name}'")

    with pytest.raises(RuntimeError, match="PyTorch"):
        build_mlp(_arch(), _import=_raise_import_error)


def test_build_mlp_returns_an_nn_module():
    torch = pytest.importorskip("torch")
    model = build_mlp(_arch())
    assert isinstance(model, torch.nn.Module)


def test_regression_output_shape():
    torch = pytest.importorskip("torch")
    model = build_mlp(_arch(task_type=TaskType.REGRESSION, output_dim=1, input_features=5))
    x = torch.randn(10, 5)
    out = model(x)
    assert out.shape == (10, 1)


def test_binary_classification_output_shape_two_logits():
    torch = pytest.importorskip("torch")
    model = build_mlp(
        _arch(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=2, input_features=6)
    )
    x = torch.randn(7, 6)
    out = model(x)
    assert out.shape == (7, 2)


def test_multiclass_classification_output_shape():
    torch = pytest.importorskip("torch")
    model = build_mlp(
        _arch(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=4, input_features=3)
    )
    x = torch.randn(9, 3)
    out = model(x)
    assert out.shape == (9, 4)


def test_output_carries_no_softmax_raw_logits_can_be_negative():
    torch = pytest.importorskip("torch")
    model = build_mlp(
        _arch(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=3, input_features=3)
    )
    x = torch.randn(20, 3) * 10  # large-magnitude input to push logits outside [0, 1]
    out = model(x)
    # a raw logit layer is not constrained to [0, 1] / doesn't sum to 1 per row
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_uses_configured_hidden_layer_sizes():
    torch = pytest.importorskip("torch")
    model = build_mlp(_arch(hidden_layer_sizes=[16, 8, 4], input_features=10, output_dim=1))
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    # input->16, 16->8, 8->4, 4->output(1) == 4 Linear layers
    assert len(linear_layers) == 4
    assert linear_layers[0].in_features == 10
    assert linear_layers[0].out_features == 16
    assert linear_layers[1].out_features == 8
    assert linear_layers[2].out_features == 4
    assert linear_layers[3].out_features == 1


@pytest.mark.parametrize(
    ("activation", "cls_name"),
    [(MLPActivation.RELU, "ReLU"), (MLPActivation.TANH, "Tanh"), (MLPActivation.GELU, "GELU")],
)
def test_uses_configured_activation(activation, cls_name):
    pytest.importorskip("torch")
    model = build_mlp(_arch(activation=activation))
    activation_modules = [type(m).__name__ for m in model.modules()]
    assert cls_name in activation_modules


def test_dropout_zero_adds_no_dropout_layer():
    torch = pytest.importorskip("torch")
    model = build_mlp(_arch(dropout=0.0))
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_dropout_positive_adds_dropout_layer_with_configured_probability():
    torch = pytest.importorskip("torch")
    model = build_mlp(_arch(dropout=0.3))
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == pytest.approx(0.3)


def test_no_activation_after_final_output_layer():
    torch = pytest.importorskip("torch")
    model = build_mlp(_arch(hidden_layer_sizes=[8], output_dim=1))
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert linear_layers[-1].out_features == 1
    # the network's very last leaf module is the output Linear, not an activation
    leaves = [m for m in model.modules() if len(list(m.children())) == 0]
    assert isinstance(leaves[-1], torch.nn.Linear)


def test_no_training_or_evaluation_methods_added_to_model():
    pytest.importorskip("torch")
    model = build_mlp(_arch())
    # only forward is defined by this module — no fit/evaluate/predict/score leak in
    for forbidden in ("fit", "evaluate", "score", "accuracy"):
        assert not hasattr(model, forbidden)


# --- deterministic construction --------------------------------------------


def test_deterministic_construction_when_seeded():
    torch = pytest.importorskip("torch")

    seed_everything(42)
    model_a = build_mlp(_arch(hidden_layer_sizes=[16, 8]))

    seed_everything(42)
    model_b = build_mlp(_arch(hidden_layer_sizes=[16, 8]))

    params_a = list(model_a.parameters())
    params_b = list(model_b.parameters())
    assert len(params_a) == len(params_b)
    for pa, pb in zip(params_a, params_b, strict=True):
        assert torch.equal(pa, pb)


def test_different_seeds_produce_different_initial_parameters():
    pytest.importorskip("torch")
    seed_everything(1)
    model_a = build_mlp(_arch(hidden_layer_sizes=[16]))

    seed_everything(2)
    model_b = build_mlp(_arch(hidden_layer_sizes=[16]))

    import torch

    first_a = next(model_a.parameters())
    first_b = next(model_b.parameters())
    assert not torch.equal(first_a, first_b)
