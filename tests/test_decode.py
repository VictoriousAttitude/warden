"""Tests for the Warden CBOR decoder and node rehydration (M2.1, arch sections 4.3-4.6).

The headline is the round-trip law that a content-addressed store relies on:
``canonical_cbor(decode_canonical(b)) == b`` for every accepted ``b``. We prove it
as a Hypothesis property over the closed value domain, then pin that the decoder
rejects non-canonical and out-of-domain bytes, and that nodes survive a store
write/read cycle with their content id intact.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from warden.core import (
    CborDecodeError,
    Node,
    NodeKind,
    canonical_cbor,
    decode_canonical,
    decode_node,
    encode_node,
    loads,
)
from warden.core.store import ObjectStore


def _utf8_encodable(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


_text = st.text().filter(_utf8_encodable)
_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    _text,
    st.binary(),
)
_values = st.recursive(
    _scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(_text, children, max_size=4),
    ),
    max_leaves=20,
)


@given(_values)
def test_byte_round_trip_is_canonical(value: object) -> None:
    encoded = canonical_cbor(value)
    assert canonical_cbor(decode_canonical(encoded)) == encoded


@given(_values)
def test_value_round_trip(value: object) -> None:
    assert decode_canonical(canonical_cbor(value)) == value


@given(st.integers())
def test_integers_decode_as_int_not_float(value: int) -> None:
    decoded = decode_canonical(canonical_cbor(value))
    assert isinstance(decoded, int) and not isinstance(decoded, bool)
    assert decoded == value


def test_int_and_float_stay_distinct_types() -> None:
    assert isinstance(decode_canonical(canonical_cbor(1)), int)
    assert isinstance(decode_canonical(canonical_cbor(1.0)), float)


def test_loads_rejects_trailing_bytes() -> None:
    with pytest.raises(CborDecodeError):
        loads(canonical_cbor(1) + b"\x00")


def test_decode_canonical_rejects_non_shortest_integer() -> None:
    # 0x18 0x05 encodes uint 5 in a 1-byte argument; canonical form is just 0x05.
    assert loads(b"\x18\x05") == 5
    with pytest.raises(CborDecodeError):
        decode_canonical(b"\x18\x05")


def test_decode_canonical_rejects_unsorted_map_keys() -> None:
    # A two-key map with keys out of canonical (bytewise) order.
    blob = bytearray()
    blob.append((5 << 5) | 2)  # map, 2 pairs
    blob += canonical_cbor("b")
    blob += canonical_cbor(1)
    blob += canonical_cbor("a")
    blob += canonical_cbor(2)
    assert loads(bytes(blob)) == {"b": 1, "a": 2}
    with pytest.raises(CborDecodeError):
        decode_canonical(bytes(blob))


def test_decoder_rejects_indefinite_length() -> None:
    with pytest.raises(CborDecodeError):
        loads(b"\x9f\x01\xff")  # indefinite-length array


def test_decoder_rejects_unsupported_simple_value() -> None:
    with pytest.raises(CborDecodeError):
        loads(b"\xf7")  # 'undefined'


def test_decoder_rejects_unsupported_tag() -> None:
    with pytest.raises(CborDecodeError):
        loads(b"\xc0" + canonical_cbor("2020"))  # tag 0 (date/time)


def test_decoder_rejects_half_float() -> None:
    with pytest.raises(CborDecodeError):
        loads(b"\xf9\x3c\x00")  # half-precision 1.0


def test_decoder_rejects_truncated_input() -> None:
    with pytest.raises(CborDecodeError):
        loads(struct.pack(">B", (2 << 5) | 4) + b"\x00")  # bytes of length 4, only 1 given


def test_node_round_trips_through_encode_decode() -> None:
    child = Node(NodeKind.USER_INPUT, (), {"text": "hello"})
    node = Node(NodeKind.TOOL_RESULT, (child.id,), {"body": "<web>", "n": 3})
    rehydrated = decode_node(encode_node(node))
    assert rehydrated.id == node.id
    assert rehydrated.kind is NodeKind.TOOL_RESULT
    assert rehydrated.parents == (child.id,)
    assert rehydrated.payload == {"body": "<web>", "n": 3}


def test_decode_node_rejects_non_envelope() -> None:
    with pytest.raises(ValueError):
        decode_node(canonical_cbor([1, 2, 3]))


def test_store_get_node_round_trip(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path)
    node = Node(NodeKind.TOOL_CALL, (), {"tool": "fetch_url", "url": "http://x"})
    key = store.put_node(node)
    loaded = store.get_node(key)
    assert loaded.id == node.id
    assert loaded.payload == node.payload
