"""The raw data store — where immutable original datasets are preserved.

Layout on disk::

    <root>/
      <dataset_id>/
        <original_filename>     # byte-for-byte copy, chmod 0o444 (read-only)
        reference.json          # the DatasetReference, for provenance

The store never overwrites an existing dataset directory and never
modifies a file it has written. "Processed" data produced by later
stages lives elsewhere (``data/processed/``) and is out of scope here.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from datapilot import paths

_READ_ONLY = 0o444
_CHUNK = 1 << 20  # 1 MiB


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RawDataStore:
    """Manages the on-disk collection of preserved raw datasets."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def default(cls) -> RawDataStore:
        """Store rooted at the repo's ``data/raw/`` directory."""
        return cls(paths.DATA_RAW_DIR)

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.root / dataset_id

    def store(self, source_path: Path, *, dataset_id: str, original_filename: str) -> Path:
        """Copy ``source_path`` into the store as an immutable raw copy.

        Returns the path to the stored copy. Raises ``FileExistsError`` if
        a directory for ``dataset_id`` already exists (ids are unique, so
        this indicates a bug or a reused id).
        """
        dest_dir = self.dataset_dir(dataset_id)
        dest_dir.mkdir(parents=True, exist_ok=False)

        dest = dest_dir / original_filename
        # copy2 preserves the original file's mtime/metadata before we lock it.
        shutil.copy2(source_path, dest)
        dest.chmod(_READ_ONLY)
        return dest

    def write_reference_sidecar(self, dataset_id: str, reference_json: str) -> Path:
        """Persist the serialised DatasetReference next to the raw copy."""
        path = self.dataset_dir(dataset_id) / "reference.json"
        path.write_text(reference_json, encoding="utf-8")
        return path
