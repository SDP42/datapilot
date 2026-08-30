"""Individual, modular data-quality checks.

Each module exposes one ``check(ctx: CheckContext) -> list[QualityFinding]``
function. Checks are pure and read-only.
"""
