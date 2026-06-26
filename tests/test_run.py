"""Tests for runs: fork (prefix sharing), diff (divergence), manifest roundtrip."""

from __future__ import annotations

import pytest

from warden.core.canonical import CanonicalValue
from warden.core.graph import Graph
from warden.core.nodes import Node, NodeKind
from warden.core.run import Run, RunManifest, diff, fork


def _build_chain(graph: Graph, payloads: list[CanonicalValue]) -> list[Node]:
    nodes: list[Node] = []
    previous: Node | None = None
    for payload in payloads:
        if previous is None:
            node = Node(NodeKind.USER_INPUT, (), payload)
        else:
            node = Node(NodeKind.LLM_RESPONSE, (previous.id,), payload)
        graph.add(node)
        nodes.append(node)
        previous = node
    return nodes


def test_run_topo_order_is_the_chain() -> None:
    graph = Graph()
    nodes = _build_chain(graph, [1, 2, 3])
    run = Run(graph, nodes[-1].id)
    assert run.topo_order() == [node.id for node in nodes]


def test_fork_shares_prefix() -> None:
    graph = Graph()
    nodes = _build_chain(graph, [1, 2, 3])
    run = Run(graph, nodes[-1].id)
    forked = fork(run, nodes[1].id)
    assert forked.node_ids() == {nodes[0].id, nodes[1].id}
    assert forked.graph is run.graph  # copy-free: the prefix is shared


def test_fork_rejects_non_member() -> None:
    graph = Graph()
    nodes = _build_chain(graph, [1, 2])
    stray = Node(NodeKind.USER_INPUT, (), 99)
    graph.add(stray)
    run = Run(graph, nodes[-1].id)
    with pytest.raises(ValueError):
        fork(run, stray.id)


def test_diff_identical_runs() -> None:
    graph = Graph()
    nodes = _build_chain(graph, [1, 2, 3])
    assert diff(Run(graph, nodes[-1].id), Run(graph, nodes[-1].id)).identical


def test_diff_finds_first_divergence() -> None:
    graph = Graph()
    base = _build_chain(graph, [1, 2])  # shared prefix
    branch_x = Node(NodeKind.LLM_RESPONSE, (base[-1].id,), "x")
    branch_y = Node(NodeKind.LLM_RESPONSE, (base[-1].id,), "y")
    graph.add(branch_x)
    graph.add(branch_y)
    result = diff(Run(graph, branch_x.id), Run(graph, branch_y.id))
    assert not result.identical
    assert result.index == 2
    assert {result.a, result.b} == {branch_x.id, branch_y.id}


def test_manifest_roundtrip() -> None:
    graph = Graph()
    nodes = _build_chain(graph, [1, 2, 3])
    run = Run(graph, nodes[-1].id)
    manifest = RunManifest.of(run, {"label": "demo", "steps": 3})
    assert RunManifest.from_json(manifest.to_json()) == manifest
