"""Phase 8.3 / 8.7 — the neural architecture configuration contracts.

Pydantic v2, JSON round-trip safe, JSON-primitive only — no tensor,
fitted `torch.nn.Module`, timestamp, or UUID. Importing this module never
imports PyTorch (see :mod:`dl_engine.availability`); it does not need to
— an architecture *config* is just numbers and enum values, same as
:class:`~dl_engine.contracts.DLTrainingConfig`.

**Phase 8.3** established :class:`MLPArchitectureConfig` — a small
feed-forward network, not a generic neural-network DSL. It is a
separate, additive contract from
:class:`~dl_engine.contracts.DLTrainingConfig`: the training config says
*how* to train (optimizer, learning rate, epochs, batch size, device,
seed); an architecture config says *what* to build (layer sizes,
activation, dropout, input / output dimensionality). ``architecture_
name`` is a fixed ``Literal`` discriminator per architecture — not a
competing free-text field with ``DLTrainingConfig.architecture_name``
(that one is descriptive metadata on a training run; this one identifies
which builder produced the model).

**Phase 8.7** adds three more architecture *foundations* — contracts
only, no training/evaluation integration yet:
:class:`CNNArchitectureConfig` (:func:`dl_engine.cnn.build_cnn`),
:class:`LSTMArchitectureConfig` (:func:`dl_engine.lstm.build_lstm`), and
:class:`TransformerArchitectureConfig`
(:func:`dl_engine.transformer.build_transformer`). All four architecture
configs share the same ``task_type`` / ``output_dim`` convention:
``task_type`` reuses :class:`~data_engine.problem_understanding.TaskType`
and ``output_dim`` is validated against it, matching the tensor / loss
conventions already fixed in :mod:`dl_engine.tensors` /
:mod:`dl_engine.training_loop`: ``regression`` → ``output_dim == 1``
(paired with ``MSELoss`` / ``L1Loss`` against the ``(n, 1)`` target
:func:`dl_engine.tensors.to_tensors` already produces); ``binary_
classification`` → ``output_dim == 2`` (two class logits, paired with
``CrossEntropyLoss`` against the ``(n,)`` int64 class-index target
``to_tensors`` already produces for both binary and multiclass —
reviewed and found already internally consistent; no correction to the
Phase 8.1/8.2 contracts was needed); ``multiclass_classification`` →
``output_dim >= 2`` (one logit per class, same ``CrossEntropyLoss``
convention). ``_validate_output_dim_for_task`` / ``_check_task_type_
supported`` factor out that shared, identical validation for the three
new contracts; :class:`MLPArchitectureConfig` itself is untouched
(unchanged code, unchanged public semantics) — the shared helpers are
new, used only by the new classes.

None of the three Phase-8.7 architectures are wired into
:func:`dl_engine.training_loop.train_model`,
:func:`dl_engine.execution.run_mlp_modeling`, or
:func:`dl_engine.selection.select_dl_models` — that integration is
explicitly deferred to a later increment. Constructing one of these
configs, or calling its builder, trains and selects nothing.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from data_engine.problem_understanding import TaskType

_SUPPORTED_TASK_TYPES = (
    TaskType.REGRESSION,
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
)


class MLPActivation(str, Enum):
    """Supported hidden-layer activations for the Phase-8.3 MLP."""

    RELU = "relu"
    TANH = "tanh"
    GELU = "gelu"


class MLPArchitectureConfig(BaseModel):
    """Deterministic configuration for the first Phase-8 architecture: an MLP.

    Constructing this contract builds nothing — see
    :func:`dl_engine.mlp.build_mlp`. Every field is a JSON primitive /
    enum; invalid configurations (non-positive dimensions, an empty or
    non-positive hidden-layer list, an out-of-range dropout, an
    unsupported task type, or an ``output_dim`` inconsistent with
    ``task_type``) raise ``pydantic.ValidationError`` at construction —
    never silently accepted.
    """

    architecture_name: Literal["mlp"] = Field(
        default="mlp",
        description="Fixed discriminator identifying dl_engine.mlp.build_mlp as the builder; "
        "not free text (contrast DLTrainingConfig.architecture_name).",
    )
    task_type: TaskType = Field(
        description="regression, binary_classification, or multiclass_classification — the "
        "only task types the Phase-8.3 MLP supports."
    )
    input_features: int = Field(
        gt=0, description="Number of input feature columns; must match TensorBatch.n_features."
    )
    output_dim: int = Field(
        gt=0,
        description="1 for regression; 2 for binary_classification (two-logit CrossEntropyLoss "
        "convention); num_classes (>=2) for multiclass_classification.",
    )
    hidden_layer_sizes: list[int] = Field(
        min_length=1,
        description="Sizes of the hidden layers, in order; at least one hidden layer, each "
        "size a positive integer.",
    )
    activation: MLPActivation = Field(default=MLPActivation.RELU)
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability applied after each hidden activation; 0.0 (default) "
        "adds no Dropout layer at all.",
    )

    @field_validator("hidden_layer_sizes")
    @classmethod
    def _hidden_layer_sizes_positive(cls, value: list[int]) -> list[int]:
        if any(size <= 0 for size in value):
            raise ValueError(
                f"every hidden_layer_sizes entry must be a positive integer; got {value}"
            )
        return value

    @field_validator("task_type")
    @classmethod
    def _task_type_supported(cls, value: TaskType) -> TaskType:
        if value not in _SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"unsupported task_type for the Phase-8.3 MLP: '{value.value}'; supported: "
                f"{', '.join(t.value for t in _SUPPORTED_TASK_TYPES)}"
            )
        return value

    @model_validator(mode="after")
    def _output_dim_matches_task_type(self) -> MLPArchitectureConfig:
        if self.task_type is TaskType.REGRESSION and self.output_dim != 1:
            raise ValueError(f"regression requires output_dim == 1; got {self.output_dim}")
        if self.task_type is TaskType.BINARY_CLASSIFICATION and self.output_dim != 2:
            raise ValueError(
                "binary_classification requires output_dim == 2 (two-logit CrossEntropyLoss "
                f"convention); got {self.output_dim}"
            )
        if self.task_type is TaskType.MULTICLASS_CLASSIFICATION and self.output_dim < 2:
            raise ValueError(
                f"multiclass_classification requires output_dim >= 2; got {self.output_dim}"
            )
        return self


# --- Phase 8.7 — shared validation for the advanced architecture configs ----
#
# Identical to MLPArchitectureConfig's own two validators above; factored out
# here (not by editing MLPArchitectureConfig) so CNN/LSTM/Transformer share
# one implementation without touching the existing, unchanged Phase-8.3 class.


def _check_task_type_supported(value: TaskType, architecture_label: str) -> TaskType:
    if value not in _SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"unsupported task_type for the Phase-8.7 {architecture_label}: '{value.value}'; "
            f"supported: {', '.join(t.value for t in _SUPPORTED_TASK_TYPES)}"
        )
    return value


def _validate_output_dim_for_task(task_type: TaskType, output_dim: int) -> None:
    if task_type is TaskType.REGRESSION and output_dim != 1:
        raise ValueError(f"regression requires output_dim == 1; got {output_dim}")
    if task_type is TaskType.BINARY_CLASSIFICATION and output_dim != 2:
        raise ValueError(
            f"binary_classification requires output_dim == 2 (two-logit CrossEntropyLoss "
            f"convention); got {output_dim}"
        )
    if task_type is TaskType.MULTICLASS_CLASSIFICATION and output_dim < 2:
        raise ValueError(f"multiclass_classification requires output_dim >= 2; got {output_dim}")


class CNNArchitectureConfig(BaseModel):
    """Deterministic configuration for a small 1D CNN foundation.

    Expected input tensor shape: ``(batch, input_channels, sequence_length)``
    — a caller reshaping a plain ``(n_rows, n_features)`` matrix into this
    3D shape is responsible for that reshape; :func:`dl_engine.cnn.build_cnn`
    never reshapes data itself.

    Every ``conv_channels`` layer uses ``kernel_size`` (validated **odd**,
    so ``padding = kernel_size // 2`` preserves ``sequence_length`` exactly
    across every layer — no implicit shape drift to track) and
    ``activation``. When ``pooling`` is ``True`` the conv stack ends in
    ``AdaptiveMaxPool1d(1)`` (collapsing the sequence dimension before the
    output layer, flatten size ``conv_channels[-1]``); when ``False`` the
    conv output is flattened directly (flatten size
    ``conv_channels[-1] * sequence_length``). Both are exactly deterministic
    given the odd-kernel same-padding guarantee — no sophisticated
    production architecture, a clean, shape-safe foundation only.
    """

    architecture_name: Literal["cnn"] = Field(
        default="cnn",
        description="Fixed discriminator identifying dl_engine.cnn.build_cnn as the builder.",
    )
    task_type: TaskType = Field(
        description="regression, binary_classification, or multiclass_classification."
    )
    input_channels: int = Field(
        gt=0, description="Number of input channels, e.g. 1 for a plain feature vector."
    )
    sequence_length: int = Field(
        gt=0, description="Number of positions per channel; the input's third tensor dimension."
    )
    conv_channels: list[int] = Field(
        min_length=1,
        description="Output channel count of each 1D conv layer, in order; each a positive "
        "integer.",
    )
    kernel_size: int = Field(
        gt=0,
        description="Convolution kernel size, shared by every conv layer; must be odd (exact "
        "same-padding) and not exceed sequence_length.",
    )
    activation: MLPActivation = Field(default=MLPActivation.RELU)
    pooling: bool = Field(
        default=False,
        description="True: AdaptiveMaxPool1d(1) collapses the sequence dimension before the "
        "output layer. False (default): the conv output is flattened directly.",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability applied after each conv activation; 0.0 (default) "
        "adds no Dropout layer at all.",
    )
    output_dim: int = Field(
        gt=0,
        description="1 for regression; 2 for binary_classification; num_classes (>=2) for "
        "multiclass_classification.",
    )

    @field_validator("conv_channels")
    @classmethod
    def _conv_channels_positive(cls, value: list[int]) -> list[int]:
        if any(size <= 0 for size in value):
            raise ValueError(f"every conv_channels entry must be a positive integer; got {value}")
        return value

    @field_validator("task_type")
    @classmethod
    def _task_type_supported(cls, value: TaskType) -> TaskType:
        return _check_task_type_supported(value, "CNN")

    @model_validator(mode="after")
    def _kernel_size_odd_and_fits(self) -> CNNArchitectureConfig:
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for exact same-padding; got {self.kernel_size}"
            )
        if self.kernel_size > self.sequence_length:
            raise ValueError(
                f"kernel_size ({self.kernel_size}) must not exceed sequence_length "
                f"({self.sequence_length})"
            )
        return self

    @model_validator(mode="after")
    def _output_dim_matches_task_type(self) -> CNNArchitectureConfig:
        _validate_output_dim_for_task(self.task_type, self.output_dim)
        return self


class LSTMArchitectureConfig(BaseModel):
    """Deterministic configuration for a standard sequence-processing LSTM foundation.

    Expected input tensor shape: ``(batch, seq_len, input_size)``
    (``batch_first=True``) — the caller's ``seq_len`` may vary at call
    time; only ``input_size`` (the per-timestep feature count) is fixed by
    this contract. :func:`dl_engine.lstm.build_lstm` never reshapes data.

    The final timestep's hidden state (both directions concatenated when
    ``bidirectional``) feeds a single output ``Linear`` layer — a clean
    sequence-classification/regression foundation, not a sophisticated
    seq2seq architecture.
    """

    architecture_name: Literal["lstm"] = Field(
        default="lstm",
        description="Fixed discriminator identifying dl_engine.lstm.build_lstm as the builder.",
    )
    task_type: TaskType = Field(
        description="regression, binary_classification, or multiclass_classification."
    )
    input_size: int = Field(gt=0, description="Per-timestep feature count.")
    hidden_size: int = Field(gt=0, description="LSTM hidden-state dimension.")
    num_layers: int = Field(gt=0, description="Number of stacked LSTM layers.")
    bidirectional: bool = Field(
        default=False, description="Whether the LSTM processes the sequence in both directions."
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout between stacked LSTM layers; only valid when num_layers > 1 "
        "(matching torch.nn.LSTM's own constraint).",
    )
    output_dim: int = Field(
        gt=0,
        description="1 for regression; 2 for binary_classification; num_classes (>=2) for "
        "multiclass_classification.",
    )

    @field_validator("task_type")
    @classmethod
    def _task_type_supported(cls, value: TaskType) -> TaskType:
        return _check_task_type_supported(value, "LSTM")

    @model_validator(mode="after")
    def _dropout_requires_multiple_layers(self) -> LSTMArchitectureConfig:
        if self.num_layers == 1 and self.dropout > 0.0:
            raise ValueError(
                "dropout only applies between stacked LSTM layers; num_layers == 1 requires "
                f"dropout == 0.0, got {self.dropout}"
            )
        return self

    @model_validator(mode="after")
    def _output_dim_matches_task_type(self) -> LSTMArchitectureConfig:
        _validate_output_dim_for_task(self.task_type, self.output_dim)
        return self


class TransformerArchitectureConfig(BaseModel):
    """Deterministic configuration for a compact numeric-sequence Transformer encoder foundation.

    Expected input tensor shape: ``(batch, seq_len, input_size)``
    (``batch_first=True``). A ``Linear(input_size, d_model)`` input
    projection feeds a standard ``TransformerEncoder`` stack (no positional
    encoding — kept out deliberately to stay a minimal foundation); the
    per-position outputs are mean-pooled over the sequence dimension before
    a final output ``Linear(d_model, output_dim)``.
    """

    architecture_name: Literal["transformer"] = Field(
        default="transformer",
        description="Fixed discriminator identifying dl_engine.transformer.build_transformer "
        "as the builder.",
    )
    task_type: TaskType = Field(
        description="regression, binary_classification, or multiclass_classification."
    )
    input_size: int = Field(gt=0, description="Per-position numeric feature count.")
    d_model: int = Field(gt=0, description="Transformer model (embedding) dimension.")
    num_heads: int = Field(
        gt=0, description="Number of attention heads; must evenly divide d_model."
    )
    num_encoder_layers: int = Field(gt=0, description="Number of stacked encoder layers.")
    dim_feedforward: int = Field(gt=0, description="Hidden dimension of each encoder's FFN block.")
    dropout: float = Field(
        default=0.0, ge=0.0, lt=1.0, description="Dropout used throughout the encoder stack."
    )
    output_dim: int = Field(
        gt=0,
        description="1 for regression; 2 for binary_classification; num_classes (>=2) for "
        "multiclass_classification.",
    )

    @field_validator("task_type")
    @classmethod
    def _task_type_supported(cls, value: TaskType) -> TaskType:
        return _check_task_type_supported(value, "Transformer")

    @model_validator(mode="after")
    def _num_heads_divides_d_model(self) -> TransformerArchitectureConfig:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must evenly divide d_model ({self.d_model})"
            )
        return self

    @model_validator(mode="after")
    def _output_dim_matches_task_type(self) -> TransformerArchitectureConfig:
        _validate_output_dim_for_task(self.task_type, self.output_dim)
        return self


__all__ = [
    "CNNArchitectureConfig",
    "LSTMArchitectureConfig",
    "MLPActivation",
    "MLPArchitectureConfig",
    "TransformerArchitectureConfig",
]
