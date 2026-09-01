# Model Development / Modeling (Phase 7)

`data_engine/modeling/` — a deterministic, **analysis-only** layer that
turns a dataset plus an **explicit** objective (and, in later increments,
the upstream Phase-5 `ProblemSpec` and Phase-6 `FeatureEngineeringSpec`)
into a structured `ModelingSpec`: whether the data is ready for modeling,
how to split it, which model families to consider, the training outcome,
the evaluation results, and the selected model.

> **Status.** Phase 7 is **in progress**. Only **Phase 7.1 — the
> `ModelingSpec` contract + `understand_modeling` foundation** — is
> implemented. It **infers nothing**: it validates an explicit request
> and returns a spec whose overall status and every nested section are
> `not_yet_inferred`. Model readiness, data-split planning, candidate
> model families, training, evaluation, and model selection are later
> Phase-7 increments (7.2+).

## Entrypoint

```python
from data_engine.modeling import ModelingRequest, understand_modeling

spec = understand_modeling(ModelingRequest(dataset_id="sales", objective="predict churn"))
```

`understand_modeling(request: ModelingRequest) -> ModelingSpec`

- A non-`ModelingRequest` argument → `TypeError` (a `dict`, `None`, or a
  **DataFrame** is **not** accepted).
- A blank / whitespace-only `dataset_id` → `ValueError`.
- Everything else is deterministic: **no DataFrame is inspected**, no
  model is trained, no split is performed, no metric is computed, no file
  is written, no timestamp / UUID / randomness / generated id is used, no
  dataset / version / lineage record is touched, no external or LLM call
  is made. Two calls with an equal request produce **byte-identical**
  serialised output.

## Important boundary — no DataFrame parameter

`understand_modeling` has **no `df` parameter** and never reads a
DataFrame. A later Phase-7 component may consume DataPilot's existing
contracts / registered data, but Phase 7.1 does not.

## `ModelingRequest`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dataset_id` | `str` | — (required) | Dataset identifier (the shared `dataset_id` convention). |
| `dataset_version_id` | `str \| None` | `None` | Registered `DatasetVersion` id, when the caller has one. |
| `objective` | `str \| None` | `None` | The user's plain-language goal, **verbatim**. Never inferred from column names or data. A blank string is **preserved exactly**, not replaced with `None`. |

`objective_provided` on the spec is `True` **only** when the objective is
non-blank after `.strip()`.

## `ModelingSpec`

| Field | Type | Default | Populated by 7.1? |
| --- | --- | --- | --- |
| `model_engine_version` | `str` | `"1"` | yes (constant) |
| `dataset_id` | `str` | — | yes (echoed) |
| `dataset_version_id` | `str \| None` | `None` | yes (echoed) |
| `objective` | `str \| None` | `None` | yes (echoed verbatim) |
| `objective_provided` | `bool` | — | yes (`True` iff non-blank after strip) |
| `status` | `ModelingStatus` | `not_yet_inferred` | yes — always `not_yet_inferred` in 7.1, with a `reason` |
| `reason` | `str \| None` | `None` | yes (states this is contract/foundation only) |
| `readiness` | `ModelReadiness` | all-`not_yet_inferred` | **no** — later increment |
| `split` | `DataSplitPlan` | all-`not_yet_inferred` | **no** — later increment |
| `candidates` | `ModelCandidates` | all-`not_yet_inferred` | **no** — later increment |
| `training` | `TrainingOutcome` | all-`not_yet_inferred` | **no** — later increment |
| `evaluation` | `EvaluationResults` | all-`not_yet_inferred` | **no** — later increment |
| `selection` | `ModelSelection` | all-`not_yet_inferred` | **no** — later increment |
| `notes` | `list[str]` | `[]` | yes (empty in 7.1) |

`model_engine_version` intentionally reuses the `model_` prefix for
consistency with the other engine-version fields; the model opts out of
Pydantic's protected `model_` namespace so it is a plain data field.

### Nested sections

- `ModelReadiness` — `status`, `reason`, `notes`. Whether the data /
  pipeline is structurally ready for modeling. **7.1 determines nothing.**
- `DataSplitPlan` — `status`, `reason`, `notes`. Future train /
  validation / test split decisions. **7.1 chooses no strategy and
  performs no split.**
- `ModelCandidates` — `status`, `reason`, `candidates: list[str]`,
  `notes`. Future candidate model families. **7.1 recommends and trains
  nothing;** `candidates` is empty.
- `TrainingOutcome` — `status`, `reason`, `notes`. Future model-training
  execution. **7.1 trains nothing.**
- `EvaluationResults` — `status`, `reason`, `notes`. Future evaluation /
  metric results. **7.1 calculates no metric.**
- `ModelSelection` — `status`, `reason`, `notes`. Future model comparison
  and selection. **7.1 selects no model.**

In Phase 7.1 every section is `status = not_yet_inferred` with the
payload `None` / `[]` — **never** a fabricated model name, score,
prediction, metric, split ratio, CV result, hyperparameter, feature
importance, fitted estimator, timestamp, UUID, or run id.

## Statuses — three explicit states

| `ModelingStatus` | Meaning |
| --- | --- |
| `not_yet_inferred` | No Phase-7 increment has attempted this yet (the 7.1 state). |
| `completed` | A later increment produced a value. |
| `unavailable` | A later increment attempted it and could not (see `reason`). |

## `ModelFamily` (stable enum, **nothing trained**)

`linear`, `tree_based`, `distance_based`, `probabilistic`, `ensemble`,
`neural`. Declarative only — Phase 7.1 trains, recommends, and names
**none** of these.

## No timestamp

Like `ProblemSpec` / `FeatureEngineeringSpec`, `ModelingSpec` has **no
`generated_at`** field — the determinism requirement is byte-identical
repeated output, so no wall-clock value is recorded.

## Boundaries (Phase 7.1)

- Separate from ingestion / profiling / quality / cleaning / validation /
  lineage / EDA / problem understanding / feature engineering. Phase 7.1
  depends on **nothing** beyond the stdlib and Pydantic and does not
  import or inspect a DataFrame.
- **Contract / foundation only.** No model is trained, fitted, tuned,
  selected, benchmarked, or compared; no estimator is created; no
  prediction is generated; no metric, CV result, or feature importance /
  SHAP is computed; no train/test split or cross-validation is performed;
  no target correlation or model-based feature selection or leakage
  detection is done; the DataFrame is never inspected or modified; no
  XGBoost / LightGBM / deep learning / Optuna / MLflow / experiment
  tracking / LLM / agent / external API is added. Those belong to Phase
  7.2+ or later phases.

## What Phase 7.2+ will eventually provide (not yet implemented)

- **7.2 — model readiness:** a deterministic structural check that the
  Phase-5 / Phase-6 outputs are coherent and sufficient to begin
  modeling.
- **7.3 — data-split planning:** a deterministic split-strategy
  recommendation (e.g. random vs grouped vs time-ordered holdout) — a
  recommendation, not an executed split.
- **7.4 — candidate model families:** a deterministic, rule-based mapping
  from the inferred task type to a shortlist of `ModelFamily` values.
- **7.5+ — training / evaluation / selection:** the execution stages that
  actually fit models, compute metrics, and choose one — a deliberately
  later, separately-gated increment.

None of these capabilities exist today.
