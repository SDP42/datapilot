# DataPilot Documentation

- [architecture.md](architecture.md) — system goal, high-level architecture, data flow, AI vs. deterministic split, future architecture
- [modules.md](modules.md) — per-package responsibilities
- [data-engine-contract.md](data-engine-contract.md) — the Ingestion ↔ Profiling interface (Phase 1)
- [data-quality.md](data-quality.md) — the Data Quality Analysis Engine (Phase 2, analysis only)
- [cleaning.md](cleaning.md) — the Cleaning **planning** layer (Phase 2, proposals only)
- [cleaning-execution.md](cleaning-execution.md) — the Cleaning **execution** layer (Phase 2, runs approved operations on a derived copy)
- [data-lineage.md](data-lineage.md) — Validation & Data Lineage (Phase 3): `DatasetVersion`, the version store, and lineage validation
- [eda.md](eda.md) — Exploratory Data Analysis (Phase 4, in progress): deterministic, analysis-only univariate + basic bivariate EDA
- [roadmap.md](roadmap.md) — phased development plan (0–17)
- [architecture-principles.md](architecture-principles.md) — binding engineering rules
- [decisions.md](decisions.md) — architectural decision log
