"""Tests for the content-addressed filesystem object store."""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest

from warden.core.nodes import Node, NodeKind, encode_node
from warden.core.store import IntegrityError, ObjectStore


def test_put_get_roundtrip(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path)
    key = store.put(b"hello")
    assert store.has(key)
    assert store.get(key) == b"hello"


def test_content_addressing_is_stable(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path)
    assert store.put(b"abc") == store.put(b"abc")
    assert store.put(b"abc") != store.put(b"abd")


def test_put_node_key_equals_node_id(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path)
    node = Node(NodeKind.USER_INPUT, (), {"text": "hi"})
    key = store.put_node(node)
    assert key == node.id
    assert store.get(key) == encode_node(node)


def test_missing_key_raises(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path)
    node = Node(NodeKind.USER_INPUT, (), None)
    with pytest.raises(KeyError):
        store.get(node.id)


def test_corruption_is_detected(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path)
    key = store.put(b"genuine")
    digest = key.hex()
    path = tmp_path / "objects" / digest[:2] / digest
    path.write_bytes(zlib.compress(b"tampered"))
    with pytest.raises(IntegrityError):
        store.get(key)
