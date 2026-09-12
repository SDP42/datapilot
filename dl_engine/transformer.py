"""Phase 8.7 — a compact numeric-sequence Transformer encoder foundation.

:func:`build_transformer` is the **only** builder in this module — an
input ``nn.Linear`` projection into an ``nn.TransformerEncoder`` stack,
sized by
:class:`~dl_engine.architectures.TransformerArchitectureConfig`, followed
by mean-pooling over the sequence dimension and a single ``nn.Linear``
output layer. Like :func:`dl_engine.mlp.build_mlp`, it contains **no
training logic** and **no evaluation logic**, and is **not yet wired
into** :func:`dl_engine.training_loop.train_model`,
:func:`dl_engine.execution.run_mlp_modeling`, or
:func:`dl_engine.selection.select_dl_models` — that integration is
explicitly deferred to a later increment. ``forward`` returns raw
logits; no softmax / sigmoid is applied.

Expected input tensor shape: ``(batch, seq_len, config.input_size)``
(``batch_first=True``) — ``seq_len`` may vary at call time, but the
per-position feature count must exactly equal ``config.input_size``. A
mismatched feature count is **rejected** with a clear ``ValueError``;
this module never reshapes a caller's tensor to make it fit. No
positional encoding is added — this is a minimal foundation, not a
sophisticated production architecture (a later increment may add one
alongside actual training integration).

Deterministic construction: exactly like ``build_mlp``, PyTorch draws a
layer's initial weights from its own global RNG at construction time, so
calling ``dl_engine.runtime.seed_everything(seed)`` immediately before
:func:`build_transformer` makes two independently-built models with the
same config produce bit-identical initial parameters.

Like every other ``dl_engine`` module, PyTorch is imported lazily — only
inside :func:`build_transformer`, only when called.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .architectures import TransformerArchitectureConfig
from .availability import torch_availability

if TYPE_CHECKING:
    import torch


def build_transformer(
    config: TransformerArchitectureConfig, *, _import: Callable[[str], ModuleType] = import_module
) -> torch.nn.Module:
    """Build an untrained ``torch.nn.Module`` Transformer encoder from ``config``.

    ``Linear(input_size, d_model)`` → ``TransformerEncoder`` (``num_
    encoder_layers`` layers of ``TransformerEncoderLayer(d_model, num_
    heads, dim_feedforward, dropout, batch_first=True)``) → mean-pool
    over the sequence dimension → ``Linear(d_model, output_dim)`` with no
    activation on the output.

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

    class TransformerEncoderModel(nn.Module):  # type: ignore[misc,name-defined]
        """A compact numeric-sequence Transformer encoder sized by
        ``TransformerArchitectureConfig``."""

        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(config.input_size, config.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.num_encoder_layers
            )
            self.output = nn.Linear(config.d_model, config.output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 3 or x.shape[2] != config.input_size:
                raise ValueError(
                    f"expected input shape (batch, seq_len, {config.input_size}); got "
                    f"{tuple(x.shape)}"
                )
            projected = self.input_projection(x)
            encoded = self.encoder(projected)
            pooled = encoded.mean(dim=1)
            return self.output(pooled)

    return TransformerEncoderModel()


__all__ = ["build_transformer"]
