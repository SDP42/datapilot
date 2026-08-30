"""Deterministic planning rules — one module per QualityFinding type.

Each exposes ``plan(finding, ctx) -> list[CleaningOperation]``. Pure and
read-only: rules propose, they never transform data.
"""
