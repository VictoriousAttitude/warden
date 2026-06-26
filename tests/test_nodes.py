"""Tests for node identity (Layer A): determinism, domain separation, equality."""

from __future__ import annotations

from warden.core.nodes import Node, NodeKind


def test_identity_is_deterministic() -> None:
    a = Node(NodeKind.USER_INPUT, (), {"text": "hi"})
    b = Node(NodeKind.USER_INPUT, (), {"text": "hi"})
    assert a.id == b.id
    assert a == b
    assert hash(a) == hash(b)


def test_payload_key_order_does_not_affect_identity() -> None:
    a = Node(NodeKind.LLM_REQUEST, (), {"a": 1, "b": 2})
    b = Node(NodeKind.LLM_REQUEST, (), dict([("b", 2), ("a", 1)]))
    assert a.id == b.id


def test_kind_provides_domain_separation() -> None:
    payload = {"x": 1}
    assert Node(NodeKind.USER_INPUT, (), payload).id != Node(
        NodeKind.TOOL_RESULT, (), payload
    ).id


def test_parents_are_part_of_identity() -> None:
    p1 = Node(NodeKind.USER_INPUT, (), {"x": 1})
    p2 = Node(NodeKind.USER_INPUT, (), {"x": 2})
    c1 = Node(NodeKind.LLM_RESPONSE, (p1.id,), {"y": 1})
    c2 = Node(NodeKind.LLM_RESPONSE, (p2.id,), {"y": 1})
    assert c1.id != c2.id


def test_parent_order_is_significant() -> None:
    p1 = Node(NodeKind.USER_INPUT, (), {"x": 1})
    p2 = Node(NodeKind.USER_INPUT, (), {"x": 2})
    a = Node(NodeKind.AGENT_STATE, (p1.id, p2.id), None)
    b = Node(NodeKind.AGENT_STATE, (p2.id, p1.id), None)
    assert a.id != b.id


def test_label_free_identity_is_payload_only() -> None:
    # Identity must not depend on anything outside (kind, parents, payload).
    # Two nodes that differ only in object identity still share an id.
    a = Node(NodeKind.FINAL_OUTPUT, (), {"answer": 42})
    b = Node(NodeKind.FINAL_OUTPUT, (), {"answer": 42})
    assert a is not b
    assert a.id == b.id


def test_node_is_hashable_by_identity() -> None:
    a = Node(NodeKind.USER_INPUT, (), {"text": "hi"})
    b = Node(NodeKind.USER_INPUT, (), {"text": "hi"})
    assert len({a, b}) == 1


def test_nodeid_hex_and_repr() -> None:
    node = Node(NodeKind.USER_INPUT, (), None)
    assert node.id.hex() == node.id.multihash.hex()
    assert repr(node.id).startswith("NodeId(")
