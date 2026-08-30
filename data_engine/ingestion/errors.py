"""Typed errors raised by the ingestion module.

Callers (and later the API layer) can catch ``IngestionError`` to handle
every ingestion failure uniformly, or a specific subclass for targeted
handling.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion failures."""


class SourceFileNotFoundError(IngestionError):
    """The supplied path does not exist or is not a regular file."""


class SourceFileNotReadableError(IngestionError):
    """The supplied file exists but cannot be read (permissions)."""


class UnsupportedFormatError(IngestionError):
    """The file extension is not a format ingestion currently supports."""


class InvalidCSVError(IngestionError):
    """The file has a CSV extension but could not be parsed as CSV."""
