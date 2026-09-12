"""Phase 8.7 — a small 1D CNN architecture foundation.

:func:`build_cnn` is the **only** builder in this module — a fixed stack
of ``nn.Conv1d`` + activation (+ optional ``nn.Dropout``) blocks, sized
by :class:`~dl_engine.architectures.CNNArchitectureConfig`, ending in a
single ``nn.Linear`` output layer. Like :func:`dl_engine.mlp.build_mlp`,
it contains **no training logic** and **no evaluation logic** — those
stay in :mod:`dl_engine.training_loop` / :mod:`dl_engine.evaluation`,
and this architecture is **not yet wired into either** (nor into
:func:`dl_engine.execution.run_mlp_modeling` or
:func:`dl_engine.selection.select_dl_models`) — that integration is
explicitly deferred to a later increment. ``forward`` returns raw
logits; no softmax / sigmoid is applied.

Expected input tensor shape: ``(batch, input_channels, sequence_length)``
— exactly ``config.input_channels`` / ``config.sequence_length``. A
mismatched shape is **rejected** with a clear ``ValueError``; this
module never reshapes a caller's tensor to make it fit.

Deterministic construction: exactly like ``build_mlp``, PyTorch draws a
layer's initial weights from its own global RNG at construction time, so
calling ``dl_engine.runtime.seed_everything(seed)`` immediately before
:func:`build_cnn` makes two independently-built models with the same
config produce bit-identical initial parameters. ``build_cnn`` itself
introduces no additional randomness.

Like every other ``dl_engine`` module, PyTorch is imported lazily — only
inside :func:`build_cnn`, only when called — so importing this module
never requires PyTorch to be installed.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .architectures import CNNArchitectureConfig, MLPActivation
from .availability import torch_availability

if TYPE_CHECKING:
    import torch

_ACTIVATION_FACTORY: dict[MLPActivation, str] = {
    MLPActivation.RELU: "ReLU",
    MLPActivation.TANH: "Tanh",
    MLPActivation.GELU: "GELU",
}


def build_cnn(
    config: CNNArchitectureConfig, *, _import: Callable[[str], ModuleType] = import_module
) -> torch.nn.Module:
    """Build an untrained ``torch.nn.Module`` 1D CNN from ``config``.

    One ``Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)``
    + activation (+ optional ``Dropout``) block per entry in
    ``config.conv_channels``, in order (``kernel_size`` is validated odd
    at the contract boundary, so this padding preserves ``sequence_
    length`` exactly through every conv layer). When ``config.pooling``
    is ``True`` the stack ends in ``AdaptiveMaxPool1d(1)`` before
    flattening (flatten size ``conv_channels[-1]``); otherwise the conv
    output is flattened directly (flatten size
    ``conv_channels[-1] * sequence_length``). A final ``Linear`` maps
    that flatten size to ``config.output_dim`` with **no** activation.

    ``forward`` requires input shape exactly
    ``(batch, config.input_channels, config.sequence_length)`` — any
    other shape raises ``ValueError`` rather than being reshaped.

    Raises ``RuntimeError`` if PyTorch is not installed.
    """
    availability = torch_availability(_import=_import)
    if not availability.available:
        raise RuntimeError(availability.reason)

    torch_module = _import("torch")
    nn = torch_module.nn
    activation_cls = getattr(nn, _ACTIVATION_FACTORY[config.activation])

    class CNN(nn.Module):  # type: ignore[misc,name-defined]
        """A small 1D CNN sized by ``CNNArchitectureConfig``."""

        def __init__(self) -> None:
            super().__init__()
            blocks = []
            in_channels = config.input_channels
            for out_channels in config.conv_channels:
                blocks.append(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=config.kernel_size,
                        padding=config.kernel_size // 2,
                    )
                )
                blocks.append(activation_cls())
                if config.dropout > 0.0:
                    blocks.append(nn.Dropout(config.dropout))
                in_channels = out_channels
            self.conv = nn.Sequential(*blocks)

            if config.pooling:
                self.pool = nn.AdaptiveMaxPool1d(1)
                flatten_size = config.conv_channels[-1]
            else:
                self.pool = None
                flatten_size = config.conv_channels[-1] * config.sequence_length
            self.output = nn.Linear(flatten_size, config.output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if (
                x.dim() != 3
                or x.shape[1] != config.input_channels
                or x.shape[2] != config.sequence_length
            ):
                raise ValueError(
                    f"expected input shape (batch, {config.input_channels}, "
                    f"{config.sequence_length}); got {tuple(x.shape)}"
                )
            features = self.conv(x)
            if self.pool is not None:
                features = self.pool(features)
            flattened = features.reshape(features.shape[0], -1)
            return self.output(flattened)

    return CNN()


__all__ = ["build_cnn"]
