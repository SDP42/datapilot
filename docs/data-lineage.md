# Validation & Data Lineage (Phase 3, in progress)

`data_engine/validation/` — makes dataset **versions** and their
**provenance** first-class, auditable objects. Additive: it does not
change ingestion, profiling, quality, the cleaning planner, or the
cleaning executor.

```
DatasetReference (raw, immutable)
        │  ingest → register_raw()
        ▼
DatasetVersion  <dataset_id>:raw           kind=raw,  parent=None
        │
        │  execute_cleaning(...) → CleaningExecutionReport (+ processed file)
        │  register_from_execution(report, parent_version_id=<raw>)
        ▼
DatasetVersion  <dataset_id>:exec-<execution_id>   kind=processed, parent=<raw>
```

## `DatasetVersion`

A typed, JSON-serialisable record of one concrete dataset version.

| Field | Meaning |
| --- | --- |
| `dataset_version_id` | Stable identity: `<dataset_id>:raw` or `<dataset_id>:exec-<execution_id>` (deterministic — reuses the executor's deterministic `execution_id`). |
| `dataset_id` | The dataset *family* id (the raw dataset's id). |
| `version_number` | Monotonic index within the family, assigned by the store (raw = 0). Registration order, not identity. |
| `parent_version_id` | The version this one derives from (`None` for raw). |
| `kind` | `raw` \| `processed` (`DatasetVersionKind`). |
| `status` | `registered` \| `superseded` \| `invalid` (`DatasetVersionStatus`) — changed only explicitly. |
| `raw_dataset_id`, `raw_sha256` | Source / raw identity. |
| `created_at`, `created_by` | Timestamp + producer string. |
| `path`, `source_format`, `size_bytes`, `sha256` | File reference for this version. |
| `row_count`, `column_count` | Shape. |
| `schema_snapshot` | `SchemaSnapshot` — ordered column names + `(name, dtype)` per column. |
| `quality` | `QualitySnapshot` — score, total findings, `has_critical`, findings-by-type (from the execution report's after-cleaning quality, when available). |
| `execution_id`, `plan_fingerprint`, `lineage_step_count`, `applied_operation_ids` | Lineage / execution reference. |

**Reuses, does not duplicate:** `DatasetReference`,
`ProcessedDatasetReference`, `CleaningExecutionReport`, and the existing
`DatasetLineage` / `LineageStep` models. Factories:
`DatasetVersion.from_raw(reference, df)` and
`DatasetVersion.from_execution_report(report, parent_version_id=..., schema_source=...)`.

Model-level validation rejects: an id not prefixed with `dataset_id`; a
raw version with a parent; a processed version without a parent or
`execution_id`; a `column_count` that disagrees with the schema snapshot;
negative counts.

## `DatasetVersionStore`

Deterministic, filesystem-based, **no database**. Mirrors
`ProcessedDataStore`.

```
data/versions/<dataset_id>/
    raw.json
    exec-<execution_id>.json
```

- Metadata is persisted **separately** from the processed CSV (its own
  directory tree). Each record is written **read-only**.
- `register(version)` / `register_raw(...)` / `register_from_execution(...)`
  — assigns `version_number`, verifies the referenced file's sha256 (when
  the file is reachable), validates the parent, then writes the record.
- `get(dataset_version_id)` — retrieve one version (family id is parsed
  from the id prefix). `VersionNotFoundError` if absent.
- `list_versions(dataset_id)` — all versions for a family, ordered by
  `version_number`.
- `children(dataset_version_id)` — versions whose parent is this one.
- `mark_status(id, status)` — explicit status change; never silent.

**Never silently overwrites.** Registering the same identity again:
- identical (`dataset_version_id`, `sha256`, `parent_version_id`) →
  `DuplicateVersionError`
- anything different → `ConflictingVersionError`

**Parent validation** (on register): the parent must already be
registered, must be in the same dataset family, and must agree on the raw
dataset identity — otherwise `ConflictingVersionError`. A processed
version cannot claim an unrelated parent.

**Integrity:** if `version.path` exists and its sha256 does not match
`version.sha256` → `VersionIntegrityError`.

## `validate_lineage(report, *, raw_reference=, processed_path=, parent_version=, child_version=)`

Returns a `LineageValidationResult(valid, checks_run, errors)`. It never
mutates and never repairs — an inconsistency is an error string.
`result.raise_for_status()` raises `LineageValidationError` when invalid.

Checks:

1. **Raw identity** — `report.dataset_id == report.lineage.raw_dataset_id`; and, if given, `raw_reference.dataset_id` matches.
2. **Raw sha256** — `lineage.raw_sha256` matches `raw_reference.sha256`; and, if the raw file is on disk, its recomputed hash matches both.
3. **Plan fingerprint** — `report.plan_fingerprint == report.lineage.plan_fingerprint`.
4/5. **Lineage steps** — one step per operation record; each step's `operation_type`, `status`, and `source_finding_id` agree with its `OperationExecution`; a transforming step must carry a `source_finding_id`.
6. **Processed sha256** — `lineage.processed_sha256` matches `output_dataset_reference.sha256`; and, if the processed file is on disk, its recomputed hash matches both.
7. **Success accounting** — no lineage step may claim `success` while its operation record is not `success`; `report.operations_succeeded` equals the number of `success` records.
8. **Parent/child** — `child_version` links back to the report (`dataset_version_id`, `execution_id`, `dataset_id`, `plan_fingerprint`); `child_version.parent_version_id == parent_version.dataset_version_id`; parent and child share the dataset family and raw identity; a version is not its own parent.

`assert_roundtrip_consistent(report, **kwargs)` serialises the report to
JSON, reloads it, re-runs `validate_lineage` on both, and confirms the
result is unchanged — lineage survives a JSON round-trip without losing
information.

## `LineageGraph` — read-only DAG navigation

`LineageGraph` is an in-memory navigation layer over **one dataset
family's** registered versions. The filesystem `DatasetVersionStore`
stays the source of truth — the graph holds no state of its own.

```python
graph = LineageGraph.from_store(store, dataset_id)  # or LineageGraph(list_of_versions)

graph.parent(version_id)  # DatasetVersion | None
graph.children(version_id)  # deterministic order: (version_number, id)
graph.ancestors(version_id)  # [parent, grandparent, ..., root]
graph.descendants(version_id)  # transitive children, BFS, deterministic
graph.root(version_id)  # the raw version
graph.path_to(version_id)  # [root, ..., version_id]
graph.same_family(a_id, b_id)  # bool
graph.is_ancestor(a_id, b_id)  # bool
```

For `raw → exec-A → exec-B` and `raw → exec-C`:
`root(exec-B) == raw`, `path_to(exec-B) == [raw, exec-A, exec-B]`,
`children(raw) == [exec-A, exec-C]`.

**Deterministic:** the same registered versions always produce the same
graph and the same traversal order, regardless of input order.

**No silent repair** — construction raises `LineageGraphError` on:
versions spanning more than one `dataset_id`; a `parent_version_id` that
is not in the graph; a parent in another family; a self-parent; more than
one root (or a root that is not `kind == raw`); a cycle.

**Cycle protection:** `ancestors` / `descendants` also carry a visited
guard and raise `LineageGraphError` rather than looping forever, even if
the graph was built with the internal `_validate=False` escape hatch.

## Optional auto-registration — `execute_and_register_cleaning`

The default cleaning flow is **unchanged**: `execute_cleaning(...)` still
returns a `CleaningExecutionReport` and registers nothing. No field was
added to `CleaningExecutionReport`.

Opt in with the additive wrapper:

```python
from data_engine.validation import execute_and_register_cleaning

result = execute_and_register_cleaning(
    reference,
    plan,
    version_store=store,
    approved_operation_ids=[...],
    context=ExecutionContext(allow_full_data_fit=True),
)  # -> RegisteredCleaningResult
```

It runs `execute_cleaning` verbatim (all kwargs forwarded), then:

1. registers the **raw** version (idempotent — a prior identical
   registration is reused), attaching the raw quality snapshot from the
   report's `before_quality_summary`;
2. registers the **processed** version with `parent_version_id` = the raw
   version id, using the existing deterministic identity, the correct
   execution id, plan fingerprint, and processed file hash;
3. runs `validate_lineage(report, raw_reference=..., parent_version=...,
   child_version=...)`.

`RegisteredCleaningResult` (Pydantic, JSON round-trips):
`execution_report`, `raw_version`, `processed_version`,
`lineage_validation`.

**Failure is surfaced, never hidden:** a conflicting registration, a hash
mismatch, or a failed lineage validation raises `AutoRegistrationError`
(or the underlying `VersionStoreError`). The raw dataset and the plan are
never modified.

## Cross-version diffing — `diff_versions`

```python
from data_engine.validation import diff_versions, diff_registered_versions

diff = diff_versions(from_version, to_version, graph=graph)  # -> VersionDiff
# or, resolving ids through a graph:
diff = diff_registered_versions(graph, from_id, to_id)
```

`diff_versions` **rejects versions from different `dataset_id` families**
(`VersionDiffError`). Within a family, `raw → processed` and
`processed → processed` are valid.

`VersionDiff` (Pydantic, JSON round-trips, deterministically ordered):

| Section | Contents |
| --- | --- |
| top level | `dataset_id`, `from_version_id`, `to_version_id`, `lineage_relationship` (`same_version` / `ancestor_to_descendant` / `descendant_to_ancestor` / `siblings` / `unknown_same_family`), `ancestor_version_id`, `descendant_version_id` |
| `metadata` | `FieldChange` list for `parent_version_id`, `kind`, `status`, `row_count`, `column_count`, `size_bytes`, `sha256`, `version_number` |
| `schema_diff` | `added_columns`, `removed_columns`, `dtype_changes`, `column_order_changed`, `common_columns` |
| `quality` | `available`; score / total-findings / has-critical / missing-cells before+after; per-finding-type `FieldChange` list; deterministic `improvements` / `regressions` strings |
| `content` | **only when both data files are readable**; otherwise `available=false` with an `unavailable_reason` and `identical_content=null` (never assumed identical). When available: row count, duplicate-row count, missing-cell count before+after, and `identical_content` |

The lineage relationship is resolved from the `graph` when supplied
(full ancestor check); without a graph only a direct parent link can be
proven and deeper relationships report `unknown_same_family`. The diff
never infers a transformation that is not in the lineage.

## What is deliberately NOT here yet

- No "latest version" policy beyond the graph's single-root rule, no
  version deletion / garbage collection.
- No database, no schema migration, no remote/object storage.
- No automatic schema-difference *correction*, no semantic data
  reconciliation.
- No auto-registration by default — it stays opt-in via the wrapper.
- No AI, no ML, no train/test splitting.
