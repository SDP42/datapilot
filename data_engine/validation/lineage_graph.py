"""A lightweight, deterministic lineage DAG over registered DatasetVersions.

The filesystem :class:`DatasetVersionStore` stays the source of truth.
This graph is an in-memory, read-only navigation layer built from a set
of ``DatasetVersion`` records for a **single dataset family**.

No silent repair: a missing parent, a cross-family parent, a self-parent,
a cycle, or more than one root all raise :class:`LineageGraphError`.
"""

from __future__ import annotations

from collections import deque

from .version_models import DatasetVersion, DatasetVersionKind
from .version_store import DatasetVersionStore


class LineageGraphError(Exception):
    """The registered versions do not form a valid single-family lineage."""


def _sort_key(version: DatasetVersion) -> tuple[int, str]:
    return (version.version_number, version.dataset_version_id)


class LineageGraph:
    """Read-only navigation over one dataset family's versions.

    Traversal order is deterministic: children are ordered by
    ``(version_number, dataset_version_id)``.
    """

    def __init__(self, versions: list[DatasetVersion], *, _validate: bool = True) -> None:
        self._nodes: dict[str, DatasetVersion] = {}
        self._children: dict[str, list[str]] = {}
        for version in versions:
            if version.dataset_version_id in self._nodes:
                raise LineageGraphError(
                    f"duplicate version id in graph: {version.dataset_version_id!r}"
                )
            self._nodes[version.dataset_version_id] = version
            self._children.setdefault(version.dataset_version_id, [])

        for version in self._nodes.values():
            parent_id = version.parent_version_id
            if parent_id is None:
                continue
            self._children.setdefault(parent_id, []).append(version.dataset_version_id)

        for parent_id in self._children:
            self._children[parent_id].sort(key=lambda vid: _sort_key(self._nodes[vid]))

        if _validate:
            self._validate_structure()

    # ---- construction helpers ----------------------------------------

    @classmethod
    def from_store(cls, store: DatasetVersionStore, dataset_id: str) -> LineageGraph:
        versions = store.list_versions(dataset_id)
        if not versions:
            raise LineageGraphError(f"no registered versions for dataset {dataset_id!r}")
        return cls(versions)

    def _validate_structure(self) -> None:
        family_ids = {v.dataset_id for v in self._nodes.values()}
        if len(family_ids) != 1:
            raise LineageGraphError(
                f"a lineage graph must be a single dataset family; got {sorted(family_ids)}"
            )

        roots: list[str] = []
        for version in self._nodes.values():
            parent_id = version.parent_version_id
            if parent_id is None:
                roots.append(version.dataset_version_id)
                continue
            if parent_id == version.dataset_version_id:
                raise LineageGraphError(f"version {version.dataset_version_id!r} is its own parent")
            parent = self._nodes.get(parent_id)
            if parent is None:
                raise LineageGraphError(
                    f"parent {parent_id!r} of {version.dataset_version_id!r} is not in the lineage"
                )
            if parent.dataset_id != version.dataset_id:
                raise LineageGraphError(
                    f"parent {parent_id!r} belongs to a different dataset family"
                )

        if len(roots) != 1:
            raise LineageGraphError(
                f"a lineage family must have exactly one root; found {len(roots)}: {sorted(roots)}"
            )
        root = self._nodes[roots[0]]
        if root.kind is not DatasetVersionKind.RAW:
            raise LineageGraphError(f"the root version {root.dataset_version_id!r} is not 'raw'")

        self._detect_cycles()

    def _detect_cycles(self) -> None:
        WHITE, GREY, BLACK = 0, 1, 2
        colour = dict.fromkeys(self._nodes, WHITE)
        for start in sorted(self._nodes):
            if colour[start] != WHITE:
                continue
            stack = [start]
            while stack:
                node = stack[-1]
                if colour[node] == WHITE:
                    colour[node] = GREY
                progressed = False
                for child in self._children.get(node, []):
                    if colour[child] == GREY:
                        raise LineageGraphError(f"lineage cycle detected involving {child!r}")
                    if colour[child] == WHITE:
                        stack.append(child)
                        progressed = True
                        break
                if not progressed:
                    colour[node] = BLACK
                    stack.pop()

    # ---- accessors ---------------------------------------------------

    def has(self, version_id: str) -> bool:
        return version_id in self._nodes

    def get(self, version_id: str) -> DatasetVersion:
        try:
            return self._nodes[version_id]
        except KeyError as exc:
            raise LineageGraphError(f"unknown version {version_id!r}") from exc

    def nodes(self) -> list[DatasetVersion]:
        return sorted(self._nodes.values(), key=_sort_key)

    def parent(self, version_id: str) -> DatasetVersion | None:
        node = self.get(version_id)
        if node.parent_version_id is None:
            return None
        return self.get(node.parent_version_id)

    def children(self, version_id: str) -> list[DatasetVersion]:
        self.get(version_id)
        return [self._nodes[c] for c in self._children.get(version_id, [])]

    def ancestors(self, version_id: str) -> list[DatasetVersion]:
        """Parent, grandparent, ... up to (and including) the root."""
        node = self.get(version_id)
        result: list[DatasetVersion] = []
        seen = {node.dataset_version_id}
        current = node.parent_version_id
        while current is not None:
            if current in seen:
                raise LineageGraphError(
                    f"lineage cycle detected while walking ancestors of {version_id!r}"
                )
            parent = self.get(current)
            result.append(parent)
            seen.add(current)
            current = parent.parent_version_id
        return result

    def descendants(self, version_id: str) -> list[DatasetVersion]:
        """All transitive children, breadth-first, deterministic order."""
        self.get(version_id)
        result: list[DatasetVersion] = []
        seen = {version_id}
        queue: deque[str] = deque(self._children.get(version_id, []))
        while queue:
            current = queue.popleft()
            if current in seen:
                raise LineageGraphError(
                    f"lineage cycle detected while walking descendants of {version_id!r}"
                )
            seen.add(current)
            result.append(self._nodes[current])
            queue.extend(self._children.get(current, []))
        return result

    def root(self, version_id: str) -> DatasetVersion:
        chain = self.ancestors(version_id)
        return chain[-1] if chain else self.get(version_id)

    def path_to(self, version_id: str) -> list[DatasetVersion]:
        """Ordered lineage path ``[root, ..., version_id]``."""
        node = self.get(version_id)
        return [*reversed(self.ancestors(version_id)), node]

    def same_family(self, version_a: str, version_b: str) -> bool:
        return self.has(version_a) and self.has(version_b)

    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        return any(a.dataset_version_id == ancestor_id for a in self.ancestors(descendant_id))
