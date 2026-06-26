"""Runs over a shared provenance graph: fork, diff, and a serializable manifest.

A run is identified by its head node; its membership is the head together with all
of its ancestors. Because nodes are content-addressed and immutable, forking is
copy-free: a fork simply names an earlier node as a new head and shares the entire
prefix. Diffing two runs compares their canonical topological orders; the first
position at which the id sequences differ is the divergence point, which is
meaningful precisely because ids exclude security labels (architecture finding F1).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import zip_longest

from warden.core.graph import Graph
from warden.core.nodes import NodeId

__all__ = ["Divergence", "Run", "RunManifest", "diff", "fork"]

type JsonScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Run:
    """A run over a shared graph, identified by its head node."""

    graph: Graph
    head: NodeId

    def node_ids(self) -> set[NodeId]:
        return self.graph.ancestors_inclusive(self.head)

    def topo_order(self) -> list[NodeId]:
        return self.graph.topo_order(self.node_ids())


def fork(run: Run, at: NodeId) -> Run:
    """Fork ``run`` at ``at``, sharing the immutable prefix with no node copy.

    ``at`` must be the head or an ancestor of it.
    """
    if at not in run.node_ids():
        raise ValueError("fork point is not part of the run")
    return Run(run.graph, at)


@dataclass(frozen=True, slots=True)
class Divergence:
    """Where two runs first differ in canonical topological order."""

    index: int | None
    a: NodeId | None
    b: NodeId | None

    @property
    def identical(self) -> bool:
        return self.index is None


def diff(a: Run, b: Run) -> Divergence:
    """Return the first divergence between two runs' canonical topo orders."""
    sequence_a = a.topo_order()
    sequence_b = b.topo_order()
    for index, (left, right) in enumerate(zip_longest(sequence_a, sequence_b)):
        if left != right:
            return Divergence(index, left, right)
    return Divergence(None, None, None)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """A serializable record of a run: head, canonical node-id list, metadata."""

    head: NodeId
    node_ids: tuple[NodeId, ...]
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    @classmethod
    def of(cls, run: Run, metadata: Mapping[str, JsonScalar] | None = None) -> RunManifest:
        return cls(run.head, tuple(run.topo_order()), dict(metadata or {}))

    def to_json(self) -> str:
        return json.dumps(
            {
                "head": self.head.hex(),
                "nodes": [node_id.hex() for node_id in self.node_ids],
                "metadata": dict(self.metadata),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, data: str) -> RunManifest:
        parsed = json.loads(data)
        return cls(
            head=NodeId(bytes.fromhex(parsed["head"])),
            node_ids=tuple(NodeId(bytes.fromhex(h)) for h in parsed["nodes"]),
            metadata=parsed["metadata"],
        )
