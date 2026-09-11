"""Phase 8.3 — the first Phase-8 neural architecture: a small feed-forward MLP.

:func:`build_mlp` is the **only** builder in this module — a fixed stack
of ``nn.Linear`` + activation (+ optional ``nn.Dropout``) blocks, sized
by :class:`~dl_engine.architectures.MLPArchitectureConfig`. It contains
**no training logic** (no optimizer, no loss, no epoch loop — that stays
in :mod:`dl_engine.training_loop`) and **no evaluation logic** (no
accuracy / F1 / RMSE — that remains out of scope for Phase 8 entirely so
far). ``forward`` returns raw logits; **no softmax / sigmoid is applied**
— :mod:`dl_engine.training_loop`'s ``CrossEntropyLoss`` (classification)
/ ``MSELoss`` / ``L1Loss`` (regression) expect raw scores, matching the
target conventions :func:`dl_engine.tensors.to_tensors` already produces.

Deterministic construction: PyTorch draws a layer's initial weights from
its own global RNG at construction time, so calling
``dl_engine.runtime.seed_everything(seed)`` immediately before
:func:`build_mlp` makes two independently-built models with the same
:class:`~dl_engine.architectures.MLPArchitectureConfig` produce
bit-identical initial parameters (see the Phase-8.3 determinism tests).
``build_mlp`` itself introduces no additional randomness.

This is a **separate implementation from the Phase-7 scikit-learn MLP
baseline** (``data_engine.modeling.training``'s ``MLPRegressor`` /
``MLPClassifier``, under ``ModelFamily.NEURAL``) — Phase 7's behavior is
untouched by this module. `dl_engine`'s PyTorch MLP is a distinct,
opt-in execution path; the two are not merged and do not call one
another.

Like every other ``dl_engine`` module, PyTorch is imported lazily —
only inside :func:`build_mlp`, only when called — so importing this
module never requires PyTorch to be installed.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .architectures import MLPActivation, MLPArchitectureConfig
from .availability import torch_availability

if TYPE_CHECKING:
    import torch

_ACTIVATION_FACTORY: dict[MLPActivation, str] = {
    MLPActivation.RELU: "ReLU",
    MLPActivation.TANH: "Tanh",
    MLPActivation.GELU: "GELU",
}


def build_mlp(
    config: MLPArchitectureConfig, *, _import: Callable[[str], ModuleType] = import_module
) -> torch.nn.Module:
    """Build an untrained ``torch.nn.Module`` MLP from ``config``.

    The network is ``nn.Linear(input_features, hidden[0])`` → activation
    → (``nn.Dropout(config.dropout)`` if ``config.dropout > 0``) → ... →
    ``nn.Linear(hidden[-1], output_dim)``, with one ``Linear`` +
    activation (+ optional dropout) block per entry in
    ``config.hidden_layer_sizes``, in order, and a final ``Linear`` to
    ``config.output_dim`` carrying **no** activation (raw logits /
    regression output).

    Raises ``RuntimeError`` if PyTorch is not installed — construction
    genuinely cannot proceed without it, unlike the pure-Pydantic
    validation already performed when ``config`` itself was constructed.
    """
    availability = torch_availability(_import=_import)
    if not availability.available:
        raise RuntimeError(availability.reason)

    torch_module = _import("torch")
    nn = torch_module.nn
    activation_cls = getattr(nn, _ACTIVATION_FACTORY[config.activation])

    class MLP(nn.Module):  # type: ignore[misc,name-defined]
        """A small feed-forward MLP sized by ``MLPArchitectureConfig``."""

        def __init__(self) -> None:
            super().__init__()
            blocks = []
            in_features = config.input_features
            for hidden_size in config.hidden_layer_sizes:
                blocks.append(nn.Linear(in_features, hidden_size))
                blocks.append(activation_cls())
                if config.dropout > 0.0:
                    blocks.append(nn.Dropout(config.dropout))
                in_features = hidden_size
            blocks.append(nn.Linear(in_features, config.output_dim))
            self.network = nn.Sequential(*blocks)

        def forward(self, x):
            return self.network(x)

    return MLP()


__all__ = ["build_mlp"]
