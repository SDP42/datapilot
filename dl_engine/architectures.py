"""Phase 8.3 — the MLP architecture configuration contract.

Pydantic v2, JSON round-trip safe, JSON-primitive only — no tensor,
fitted `torch.nn.Module`, timestamp, or UUID. Importing this module never
imports PyTorch (see :mod:`dl_engine.availability`); it does not need to
— an architecture *config* is just numbers and enum values, same as
:class:`~dl_engine.contracts.DLTrainingConfig`.

:class:`MLPArchitectureConfig` is deliberately the **only** architecture
contract in ``dl_engine`` — a small feed-forward network, not a generic
neural-network DSL. It is a separate, additive contract from
:class:`~dl_engine.contracts.DLTrainingConfig`: the training config says
*how* to train (optimizer, learning rate, epochs, batch size, device,
seed); this config says *what* to build (layer sizes, activation,
dropout, input / output dimensionality). ``architecture_name`` is a fixed
``Literal["mlp"]`` discriminator — not a competing free-text field with
``DLTrainingConfig.architecture_name`` (that one is descriptive metadata
on a training run; this one identifies which builder in
:mod:`dl_engine.mlp` produced the model) — and is kept a ``Literal`` so
adding a second architecture later is an additive enum value, not a
generic-DSL escape hatch.

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
convention).
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


__all__ = ["MLPActivation", "MLPArchitectureConfig"]
