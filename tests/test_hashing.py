"""Tests for content hashing, multihash framing, and varint codec."""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from warden.core.hashing import (
    Blake2b256,
    HashAlgo,
    default_hash,
    multihash,
    read_uvarint,
    uvarint,
)


def test_uvarint_known_values() -> None:
    assert uvarint(0) == b"\x00"
    assert uvarint(127) == b"\x7f"
    assert uvarint(128) == b"\x80\x01"
    assert uvarint(300) == b"\xac\x02"


@given(st.integers(min_value=0, max_value=2**64))
def test_uvarint_roundtrip(value: int) -> None:
    encoded = uvarint(value)
    decoded, offset = read_uvarint(encoded)
    assert decoded == value
    assert offset == len(encoded)


def test_uvarint_rejects_negative() -> None:
    with pytest.raises(ValueError):
        uvarint(-1)


def test_read_uvarint_rejects_truncated() -> None:
    with pytest.raises(ValueError):
        read_uvarint(b"\x80")  # continuation bit set but no following byte


def test_default_hash_is_fixed_blake2b256() -> None:
    algo = default_hash()
    assert isinstance(algo, HashAlgo)
    assert algo.multicodec == 0xB220
    assert algo.digest_size == 32


def test_multihash_framing_and_digest() -> None:
    data = b"warden"
    prefix = uvarint(0xB220) + uvarint(32)
    mh = multihash(Blake2b256(), data)
    assert mh.startswith(prefix)
    assert mh[len(prefix) :] == hashlib.blake2b(data, digest_size=32).digest()
    assert len(mh) == len(prefix) + 32


def test_multihash_is_deterministic() -> None:
    assert multihash(Blake2b256(), b"x") == multihash(Blake2b256(), b"x")


def test_multihash_distinguishes_input() -> None:
    assert multihash(Blake2b256(), b"a") != multihash(Blake2b256(), b"b")
