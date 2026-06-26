"""Tests for the provenance graph: structure, ordering, determinism."""

from __future__ import annotations

import pytest

from warden.core.canonical import CanonicalValue
from warden.core.graph import Graph, MissingParentError, UnknownNodeError
from warden.core.nodes import Node, NodeId, NodeKind


def _node(
    kind: NodeKind, parents: tuple[NodeId, ...] = (), payload: CanonicalValue = None
) -> Node:
    return Node(kind, parents, payload)


def test_add_requires_parents_present() -> None:
    graph = Graph()
    root = _node(NodeKind.USER_INPUT, payload=1)
    child = _node(NodeKind.LLM_RESPONSE, (root.id,))
    with pytest.raises(MissingParentError):
        graph.add(child)
    graph.add(root)
    graph.add(child)
    assert child.id in graph


def test_topo_order_respects_dependencies() -> None:
    graph = Graph()
    a = _node(NodeKind.USER_INPUT, payload=1)
    b = _node(NodeKind.LLM_REQUEST, (a.id,))
    c = _node(NodeKind.LLM_RESPONSE, (b.id,))
    for node in (a, b, c):
        graph.add(node)
    order = graph.topo_order()
    assert order.index(a.id) < order.index(b.id) < order.index(c.id)


def test_topo_order_is_deterministic_with_tiebreak() -> None:
    graph = Graph()
    root = _node(NodeKind.USER_INPUT, payload=1)
    graph.add(root)
    leaves = [_node(NodeKind.TOOL_RESULT, (root.id,), payload=i) for i in range(5)]
    for leaf in leaves:
        graph.add(leaf)
    first = graph.topo_order()
    assert first == graph.topo_order()
    leaf_ids = {leaf.id for leaf in leaves}
    emitted = [nid for nid in first if nid in leaf_ids]
    assert emitted == sorted(leaf_ids)


def test_roots_and_ancestors() -> None:
    graph = Graph()
    a = _node(NodeKind.USER_INPUT, payload=1)
    b = _node(NodeKind.SYSTEM_PROMPT, payload=2)
    c = _node(NodeKind.LLM_REQUEST, (a.id, b.id))
    for node in (a, b, c):
        graph.add(node)
    assert set(graph.roots()) == {a.id, b.id}
    assert graph.ancestors_inclusive(c.id) == {a.id, b.id, c.id}


def test_unknown_node_raises() -> None:
    graph = Graph()
    node = _node(NodeKind.USER_INPUT, payload=1)
    with pytest.raises(UnknownNodeError):
        graph.get(node.id)
