"""The in-memory provenance DAG over content-addressed nodes (Layer A).

The graph holds nodes by id and exposes the structural reads both higher layers
need: roots, ancestors, and a canonical topological order. Edges point from a
child to its parents; a node can only be added once all of its parents are
present, which guarantees the structure stays a DAG (a cycle is impossible by
construction, since a node's id commits to its parents' ids and a hash cannot
contain itself).
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from collections.abc import Set as AbstractSet

from warden.core.nodes import Node, NodeId

__all__ = ["Graph", "MissingParentError", "UnknownNodeError"]


class MissingParentError(Exception):
    """A node was added before one of its parents was present in the graph."""


class UnknownNodeError(KeyError):
    """A node id was referenced that is not present in the graph."""


class Graph:
    """A set of content-addressed nodes forming a provenance DAG."""

    __slots__ = ("_nodes",)

    def __init__(self) -> None:
        self._nodes: dict[NodeId, Node] = {}

    def add(self, node: Node) -> NodeId:
        for parent in node.parents:
            if parent not in self._nodes:
                raise MissingParentError(parent)
        # Idempotent: a node with the same id has identical content.
        self._nodes[node.id] = node
        return node.id

    def get(self, node_id: NodeId) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise UnknownNodeError(node_id) from None

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[NodeId]:
        return iter(self._nodes)

    def roots(self) -> list[NodeId]:
        """Return ids of nodes with no parents, in canonical (bytewise) order."""
        return sorted(nid for nid, node in self._nodes.items() if not node.parents)

    def ancestors_inclusive(self, node_id: NodeId) -> set[NodeId]:
        """Return ``node_id`` together with all of its transitive parents."""
        if node_id not in self._nodes:
            raise UnknownNodeError(node_id)
        seen: set[NodeId] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._nodes[current].parents)
        return seen

    def topo_order(self, restrict: AbstractSet[NodeId] | None = None) -> list[NodeId]:
        """Return a canonical topological order (parents before children).

        Among nodes that are simultaneously ready, ties are broken by id (bytewise),
        so the order is fully deterministic. With ``restrict`` the order is computed
        over only that subset, considering only edges internal to it.
        """
        members = set(self._nodes) if restrict is None else set(restrict)
        for member in members:
            if member not in self._nodes:
                raise UnknownNodeError(member)

        indegree: dict[NodeId, int] = {}
        children: dict[NodeId, list[NodeId]] = {member: [] for member in members}
        for member in members:
            internal_parents = [p for p in self._nodes[member].parents if p in members]
            indegree[member] = len(internal_parents)
            for parent in internal_parents:
                children[parent].append(member)

        ready = [member for member in members if indegree[member] == 0]
        heapq.heapify(ready)
        order: list[NodeId] = []
        while ready:
            current = heapq.heappop(ready)
            order.append(current)
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        return order
