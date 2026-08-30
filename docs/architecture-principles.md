# Architecture Principles

These rules are binding for every phase. A change that violates one of
them needs an entry in `docs/decisions.md` explaining why.

1. **Raw data is immutable.** The ingested dataset is never modified in
   place. Cleaning and preprocessing produce new, separately stored
   versions.
2. **Every transformation is traceable.** Each processed dataset records
   which operations produced it, with their parameters, in the lineage
   store.
3. **Cleaning decisions are explainable.** Every cleaning action carries a
   human-readable rationale and the evidence (from profiling/quality) that
   motivated it.
4. **Outliers are not auto-deleted.** Potential outliers are flagged and
   surfaced; removal only happens via an explicit, approved plan.
5. **Data leakage is actively considered.** Leakage checks run before
   modelling; suspected leakage blocks a pipeline until resolved.
6. **AI proposals become deterministic operations.** An LLM recommendation
   is executed only after translation into a typed, parameterised call to
   a deterministic tool, followed by validation.
7. **Every ML experiment is reproducible.** Seeds, data versions, code
   version, and environment are captured; a rerun yields the same result.
8. **Evaluation uses task-appropriate metrics.** Never accuracy alone;
   metric choice is justified by the problem spec (class balance, cost of
   errors, etc.).
9. **No framework without a reason.** New dependencies require a decision
   log entry stating the need and the alternatives rejected.
10. **The system stays modular.** Each engine is independently testable
    and communicates only through structured result objects.
11. **AI agents use tools, not internal state.** Future agents act solely
    through the tool layer; they never read or write engine internals or
    dataframes directly.
12. **Raw and processed data are stored separately.** `data/raw/` is
    preserved; processed and interim versions live in their own locations.
