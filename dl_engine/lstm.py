"""Phase 8.7 — a standard sequence-processing LSTM architecture foundation.

:func:`build_lstm` is the **only** builder in this module — an
``nn.LSTM`` stack sized by
:class:`~dl_engine.architectures.LSTMArchitectureConfig`, followed by a
single ``nn.Linear`` output layer applied to the final timestep's hidden
state. Like :func:`dl_engine.mlp.build_mlp`, it contains **no training
logic** and **no evaluation logic**, and is **not yet wired into**
:func:`dl_engine.training_loop.train_model`,
:func:`dl_engine.execution.run_mlp_modeling`, or
:func:`dl_engine.selection.select_dl_models` — that integration is
explicitly deferred to a later increment. ``forward`` returns raw
logits; no softmax / sigmoid is applied.

Expected input tensor shape: ``(batch, seq_len, config.input_size)``
(``batch_first=True``) — ``seq_len`` may vary at call time, but the
per-timestep feature count must exactly equal ``config.input_size``. A
mismatched feature count is **rejected** with a clear ``ValueError``;
this module never reshapes a caller's tensor to make it fit.

Deterministic construction: exactly like ``build_mlp``, PyTorch draws a
layer's initial weights from its own global RNG at construction time, so
calling ``dl_engine.runtime.seed_everything(seed)`` immediately before
:func:`build_lstm` makes two independently-built models with the same
config produce bit-identical initial parameters.

Like every other ``dl_engine`` module, PyTorch is imported lazily — only
inside :func:`build_lstm`, only when called.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .architectures import LSTMArchitectureConfig
from .availability import torch_availability

if TYPE_CHECKING:
    import torch


def build_lstm(
    config: LSTMArchitectureConfig, *, _import: Callable[[str], ModuleType] = import_module
) -> torch.nn.Module:
    """Build an untrained ``torch.nn.Module`` LSTM from ``config``.

    ``nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
    bidirectional=config.bidirectional, dropout=config.dropout if
    num_layers > 1 else 0.0)`` followed by ``nn.Linear(hidden_size * (2
    if bidirectional else 1), output_dim)`` applied to the final
    timestep's hidden state (both directions concatenated when
    bidirectional) — no activation on the output layer.

    ``forward`` requires a 3D input whose last dimension equals
    ``config.input_size`` — any other shape raises ``ValueError`` rather
    than being reshaped.

    Raises ``RuntimeError`` if PyTorch is not installed.
    """
    availability = torch_availability(_import=_import)
    if not availability.available:
        raise RuntimeError(availability.reason)

    torch_module = _import("torch")
    nn = torch_module.nn

    class LSTM(nn.Module):  # type: ignore[misc,name-defined]
        """A standard sequence-processing LSTM sized by ``LSTMArchitectureConfig``."""

        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            directions = 2 if config.bidirectional else 1
            self.output = nn.Linear(config.hidden_size * directions, config.output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 3 or x.shape[2] != config.input_size:
                raise ValueError(
                    f"expected input shape (batch, seq_len, {config.input_size}); got "
                    f"{tuple(x.shape)}"
                )
            _, (hidden, _) = self.lstm(x)
            # hidden: (num_layers * directions, batch, hidden_size) — take the last layer
            if config.bidirectional:
                last_forward = hidden[-2]
                last_backward = hidden[-1]
                final = torch_module.cat([last_forward, last_backward], dim=1)
            else:
                final = hidden[-1]
            return self.output(final)

    return LSTM()


__all__ = ["build_lstm"]
