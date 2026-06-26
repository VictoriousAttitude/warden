"""Tests for propagation (arch section 6): INV-3 monotonicity, whole-context, declassification."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from warden.core import Graph, Node, NodeId, NodeKind
from warden.labels import Confidentiality, Label, Taint, join_all
from warden.propagate import (
    LabelOverlay,
    UnlabeledError,
    ValueEnv,
    propagate,
)

_SOURCES = ["a", "b", "c"]

label_strategy = st.builds(
    Label,
    integrity=st.sampled_from(Taint),
    confidentiality=st.sampled_from(Confidentiality),
    provenance=st.frozensets(st.sampled_from(_SOURCES), max_size=3),
)


@st.composite
def dags(draw: st.DrawFn) -> list[tuple[Node, Label]]:
    """Build a random DAG: node i's parents are a subset of earlier nodes."""
    n = draw(st.integers(min_value=1, max_value=8))
    built: list[tuple[Node, Label]] = []
    for i in range(n):
        if i == 0:
            parent_idxs: list[int] = []
        else:
            parent_idxs = draw(
                st.lists(
                    st.integers(min_value=0, max_value=i - 1),
                    unique=True,
                    max_size=min(i, 3),
                )
            )
        parents = tuple(built[j][0].id for j in parent_idxs)
        node = Node(NodeKind.AGENT_STATE, parents, {"i": i})
        built.append((node, draw(label_strategy)))
    return built


# --- core propagation -------------------------------------------------------


def test_untrusted_source_taints_descendants() -> None:
    user = Node(NodeKind.USER_INPUT, (), {"q": "summarize my inbox"})
    fetched = Node(NodeKind.TOOL_RESULT, (user.id,), {"body": "...injection..."})
    plan = Node(NodeKind.LLM_RESPONSE, (user.id, fetched.id), {"action": "send"})

    graph = Graph()
    for node in (user, fetched, plan):
        graph.add(node)

    sources = {fetched.id: Label(Taint.UNTRUSTED, provenance=frozenset({"fetch_url"}))}
    overlay = propagate(graph, sources)

    assert overlay.label_of(user.id).integrity is Taint.TRUSTED
    assert overlay.label_of(fetched.id).integrity is Taint.UNTRUSTED
    # Whole-context: the plan derived from the fetched result is now untrusted.
    plan_label = overlay.label_of(plan.id)
    assert plan_label.integrity is Taint.UNTRUSTED
    assert "fetch_url" in plan_label.provenance


def test_whole_context_joins_all_parents() -> None:
    a = Node(NodeKind.TOOL_RESULT, (), {"x": 1})
    b = Node(NodeKind.TOOL_RESULT, (), {"x": 2})
    merged = Node(NodeKind.AGENT_STATE, (a.id, b.id), {"m": 1})
    graph = Graph()
    for node in (a, b, merged):
        graph.add(node)

    sources = {
        a.id: Label(Taint.UNTRUSTED),
        b.id: Label(confidentiality=Confidentiality.SECRET),
    }
    overlay = propagate(graph, sources)
    label = overlay.label_of(merged.id)
    assert label.integrity is Taint.UNTRUSTED
    assert label.confidentiality is Confidentiality.SECRET


@given(dags())
def test_inv3_monotonicity_holds_without_declassification(
    built: list[tuple[Node, Label]],
) -> None:
    graph = Graph()
    sources: dict[NodeId, Label] = {}
    for node, src in built:
        graph.add(node)
        sources[node.id] = src
    overlay = propagate(graph, sources)

    for node, src in built:
        label = overlay.label_of(node.id)
        parent_labels = [overlay.label_of(p) for p in node.parents]
        # Equality form of INV-3 (no declassification anywhere).
        assert label == join_all(parent_labels).join(src)
        # Order form: derived label dominates every parent.
        for parent in node.parents:
            assert overlay.label_of(parent).leq(label)


# --- declassification (the only sanctioned break) ---------------------------


def test_declassification_lowers_across_the_node() -> None:
    secret = Node(NodeKind.TOOL_RESULT, (), {"ssn": "..."})
    declassed = Node(NodeKind.DECLASSIFICATION, (secret.id,), {"reason": "user ok"})
    graph = Graph()
    graph.add(secret)
    graph.add(declassed)

    sources = {
        secret.id: Label(confidentiality=Confidentiality.SECRET),
        declassed.id: Label(confidentiality=Confidentiality.PUBLIC),
    }
    overlay = propagate(graph, sources)
    assert overlay.label_of(secret.id).confidentiality is Confidentiality.SECRET
    # Across the declassification node the label drops below join(parents).
    assert overlay.label_of(declassed.id).confidentiality is Confidentiality.PUBLIC
    assert not overlay.label_of(secret.id).leq(overlay.label_of(declassed.id))


def test_declassification_without_authority_falls_back_to_join() -> None:
    secret = Node(NodeKind.TOOL_RESULT, (), {"ssn": "..."})
    # A declassification node with NO declared authority must not silently lower.
    declassed = Node(NodeKind.DECLASSIFICATION, (secret.id,), {"reason": "forgot"})
    graph = Graph()
    graph.add(secret)
    graph.add(declassed)

    sources = {secret.id: Label(confidentiality=Confidentiality.SECRET)}
    overlay = propagate(graph, sources)
    assert overlay.label_of(declassed.id).confidentiality is Confidentiality.SECRET


# --- value environment & overlay accessors ----------------------------------


def test_value_env_resolves_and_fails_closed() -> None:
    node = Node(NodeKind.TOOL_RESULT, (), {"x": 1})
    label = Label(Taint.UNTRUSTED, provenance=frozenset({"fetch_url"}))
    env = ValueEnv({"h1": (node.id, label)})
    assert env.resolve("h1") == (node.id, label)
    assert env.label_of("h1") == label
    with pytest.raises(UnlabeledError):
        env.label_of("missing")


def test_overlay_label_of_missing_raises_but_get_defaults() -> None:
    overlay = LabelOverlay({})
    missing = Node(NodeKind.USER_INPUT, (), {"x": 1}).id
    with pytest.raises(UnlabeledError):
        overlay.label_of(missing)
    assert overlay.get(missing, Label.bottom()) == Label.bottom()
    assert missing not in overlay
    assert len(overlay) == 0
