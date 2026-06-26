"""Conformance and property tests for the Warden CBOR profile."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from warden.core.canonical import CanonicalEncodingError, canonical_cbor

# --- RFC 8949 Appendix A known-answer vectors -------------------------------
# Integers, strings, byte strings, arrays, maps, and simple values are taken
# directly from RFC 8949. Floats deviate intentionally: this profile always uses
# the 64-bit form (additional info 27), not RFC 8949's shortest-float guidance.

_INTEGER_AND_STRUCTURE_VECTORS = [
    (0, "00"),
    (1, "01"),
    (10, "0a"),
    (23, "17"),
    (24, "1818"),
    (25, "1819"),
    (100, "1864"),
    (1000, "1903e8"),
    (1000000, "1a000f4240"),
    (1000000000000, "1b000000e8d4a51000"),
    (-1, "20"),
    (-10, "29"),
    (-100, "3863"),
    (-1000, "3903e7"),
    (False, "f4"),
    (True, "f5"),
    (None, "f6"),
    ("", "60"),
    ("a", "6161"),
    ("IETF", "6449455446"),
    (b"", "40"),
    (b"\x01\x02\x03\x04", "4401020304"),
    ([], "80"),
    ([1, 2, 3], "83010203"),
    ({}, "a0"),
    ({"a": 1}, "a1616101"),
    ({"a": 1, "b": [2, 3]}, "a26161016162820203"),
]


@pytest.mark.parametrize(("value", "expected"), _INTEGER_AND_STRUCTURE_VECTORS)
def test_known_vectors(value: object, expected: str) -> None:
    assert canonical_cbor(value).hex() == expected  # type: ignore[arg-type]


_FLOAT_VECTORS = [
    (0.0, "fb0000000000000000"),
    (1.0, "fb3ff0000000000000"),
    (1.5, "fb3ff8000000000000"),
    (-2.0, "fbc000000000000000"),
    (-4.1, "fbc010666666666666"),
]


@pytest.mark.parametrize(("value", "expected"), _FLOAT_VECTORS)
def test_float_vectors_are_always_float64(value: float, expected: str) -> None:
    assert canonical_cbor(value).hex() == expected


# --- Type distinctness (no coalescing across types) -------------------------


def test_bool_is_not_int() -> None:
    assert canonical_cbor(True) != canonical_cbor(1)
    assert canonical_cbor(False) != canonical_cbor(0)


def test_int_is_not_float() -> None:
    assert canonical_cbor(1) != canonical_cbor(1.0)


def test_bytes_is_not_text() -> None:
    assert canonical_cbor(b"a") != canonical_cbor("a")


# --- Canonicalisation guarantees --------------------------------------------


def test_map_key_order_is_irrelevant() -> None:
    a = canonical_cbor({"a": 1, "b": 2, "c": 3})
    b = canonical_cbor(dict([("c", 3), ("a", 1), ("b", 2)]))
    assert a == b


def test_map_keys_are_sorted_bytewise() -> None:
    # Keys must come out in bytewise order of their encoding regardless of input.
    assert canonical_cbor({"b": 2, "a": 1}).hex() == "a2616101616202"


def test_negative_zero_normalised() -> None:
    assert canonical_cbor(-0.0) == canonical_cbor(0.0)


# --- Rejections (strictness is a security property) -------------------------


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_rejected(value: float) -> None:
    with pytest.raises(CanonicalEncodingError):
        canonical_cbor(value)


def test_unsupported_type_rejected() -> None:
    with pytest.raises(CanonicalEncodingError):
        canonical_cbor({1, 2, 3})  # type: ignore[arg-type]


def test_non_string_map_key_rejected() -> None:
    with pytest.raises(CanonicalEncodingError):
        canonical_cbor({1: "x"})  # type: ignore[dict-item]


def test_excessive_nesting_rejected() -> None:
    value: object = 0
    for _ in range(300):
        value = [value]
    with pytest.raises(CanonicalEncodingError):
        canonical_cbor(value)  # type: ignore[arg-type]


def test_large_integers_use_bignums() -> None:
    # 2**64 cannot fit a 64-bit head; it must round-trip through a bignum.
    big = 2**80
    cbor2 = pytest.importorskip("cbor2")
    assert cbor2.loads(canonical_cbor(big)) == big
    assert cbor2.loads(canonical_cbor(-big)) == -big


# --- Property tests ----------------------------------------------------------


def _utf8_encodable(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


_safe_text = st.text().filter(_utf8_encodable)


def _canonical_values() -> st.SearchStrategy[object]:
    leaves = (
        st.none()
        | st.booleans()
        | st.integers(min_value=-(2**80), max_value=2**80)
        | st.floats(allow_nan=False, allow_infinity=False)
        | _safe_text
        | st.binary()
    )
    return st.recursive(
        leaves,
        lambda children: st.lists(children, max_size=5)
        | st.dictionaries(_safe_text, children, max_size=5),
        max_leaves=20,
    )


@given(_canonical_values())
def test_encoding_is_deterministic(value: object) -> None:
    assert canonical_cbor(value) == canonical_cbor(value)  # type: ignore[arg-type]


def _normalise(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value == 0.0:
        return 0.0
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    return value


@given(_canonical_values())
def test_output_decodes_to_input(value: object) -> None:
    # cbor2 is an INDEPENDENT decoder used purely as an oracle: our bytes must be
    # valid CBOR that decodes back to the (normalised) input structure.
    cbor2 = pytest.importorskip("cbor2")
    assert cbor2.loads(canonical_cbor(value)) == _normalise(value)  # type: ignore[arg-type]
