# Data Engine Contract — Ingestion ↔ Profiling

This document defines the interface between the first two stages of the
data engine. It is intentionally small and will grow one stage at a time
(quality, cleaning, validation, …).

## Conceptual flow

```
Raw File (.csv on disk)
   │
   ▼  data_engine.ingestion.ingest_dataset(path)
DatasetReference          ← immutable pointer to a preserved raw copy
   │
   ▼  data_engine.profiling.profile_dataset(reference)
DatasetProfile            ← structured, machine-readable description
```

A caller never passes a file path to the profiler and never passes a
DataFrame to ingestion. The only thing that crosses the boundary is the
`DatasetReference`.

## The handoff object: `DatasetReference`

Defined in `datapilot/contracts.py` (shared core, so neither engine
imports the other). It is **frozen** — it records a fact that already
happened.

| Field | Meaning |
| --- | --- |
| `dataset_id` | Unique id assigned at ingestion (`ds-<uuid4hex>`) |
| `original_filename` | Name of the file the caller supplied |
| `source_format` | `DatasetFormat` enum — only `csv` today |
| `raw_path` | Absolute path to the read-only preserved copy |
| `size_bytes` | Size of the raw copy |
| `sha256` | Digest of the raw bytes, for integrity checks |
| `created_at` | UTC ingestion timestamp |

It describes **the file**, never its contents. Row counts, column types,
and statistics are the profiler's job — this keeps responsibilities from
leaking across the boundary.

## The result object: `DatasetProfile`

Defined in `data_engine/profiling/models.py`. Every field is
JSON-serialisable so the future data-quality engine, API, frontend, and
AI engine can consume it with `profile.model_dump(mode="json")` — no
parsing of human-written text.

Top level: `dataset_id`, `profiler_version`, `generated_at`, `n_rows`,
`n_columns`, `column_names`, `duplicate_row_count`, and the derived lists
`numeric_columns` / `categorical_columns` / `datetime_columns`.

Per column (`ColumnProfile`): `name`, `pandas_dtype`, `inferred_type`
(`ColumnType` enum), `missing_count`, `missing_percentage`,
`unique_count`, and exactly one of `numeric_stats`, `categorical_stats`,
`datetime_stats` depending on the inferred type.

## Guarantees

- **Raw data is immutable.** The preserved copy is `chmod 0o444`; the
  caller's original file is never touched.
- **Ingestion does not transform.** It parses the CSV only to reject an
  invalid file; the parsed frame is discarded.
- **Profiling is read-only.** `profile_dataframe` is a pure function of a
  DataFrame — no imputation, de-duplication, or type coercion. A column
  stored as text that *looks* numeric or date-like is still reported with
  its real `pandas_dtype`; the mismatch is a Phase 2 data-quality concern.
- **Deterministic.** Same input file → same profile (modulo
  `generated_at`). No LLM is involved.

## Testable entrypoints

| Call | Signature | Use |
| --- | --- | --- |
| `ingest_dataset` | `path -> DatasetReference` | production flow |
| `ingest_csv` | `path -> DatasetReference` | CSV-specific |
| `profile_dataset` | `DatasetReference -> DatasetProfile` | contract flow |
| `profile_dataframe` | `DataFrame -> DatasetProfile` | unit-test / in-memory |
| `load_dataframe` | `DatasetReference -> DataFrame` | read the raw copy |
