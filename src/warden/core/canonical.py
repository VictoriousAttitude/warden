"""Deterministic CBOR encoding for content-addressing (the Warden CBOR profile).

This module is the load-bearing root of node identity. Every node id is
``multihash(hash(canonical_cbor(envelope)))`` (see ``nodes.py``), so two values
that are semantically identical MUST encode to identical bytes, and two values
that differ in type or content MUST NOT. Ambiguity here is a security defect: it
would let an attacker forge or collide provenance.

The encoder implements a restricted, stricter subset of RFC 8949 Core
Deterministic Encoding (RFC 8949 section 4.2). It is intentionally hand-written
rather than delegated to a third-party "canonical mode", because the exact output
bytes are part of the trust boundary and must be auditable in one small file.

Profile rules (pinned; tests in tests/test_canonical.py enforce each one):

  * Definite-length items only.
  * Integers in shortest form; values outside the unsigned/negative 64-bit range
    use CBOR bignums (tags 2 and 3).
  * ``bool`` is encoded as the CBOR simple values true/false -- never as an
    integer (note that in Python ``bool`` is a subclass of ``int``).
  * Floats are ALWAYS encoded as 64-bit IEEE-754 (CBOR additional info 27). This
    deviates from RFC 8949's "shortest float" guidance in favour of one
    unambiguous rule. Non-finite floats (NaN, +-Inf) are rejected. Negative zero
    is normalised to positive zero.
  * ``int`` and ``float`` are distinct value types and never coalesce (1 != 1.0).
  * ``bytes`` (major type 2) and ``str`` (major type 3) are distinct.
  * Map keys must be ``str`` and are sorted bytewise-lexicographically by their
    encoded representation (RFC 8949 section 4.2.1).
  * Text is encoded as its UTF-8 bytes verbatim. Unicode normalisation (NFC) is
    the responsibility of the ingestion boundary, not this encoder, so that the
    encoder stays a pure, total function of its input.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from typing import Final

__all__ = ["CanonicalEncodingError", "CanonicalValue", "canonical_cbor"]

# The closed value domain Warden will canonicalise. Anything else is rejected.
type CanonicalValue = (
    None
    | bool
    | int
    | float
    | str
    | bytes
    | Sequence["CanonicalValue"]
    | Mapping[str, "CanonicalValue"]
)


class CanonicalEncodingError(ValueError):
    """A value cannot be deterministically encoded under the Warden CBOR profile."""


# CBOR major types (RFC 8949 section 3.1), shifted into the high 3 bits of the
# initial byte.
_MT_UINT: Final = 0
_MT_NINT: Final = 1
_MT_BYTES: Final = 2
_MT_TEXT: Final = 3
_MT_ARRAY: Final = 4
_MT_MAP: Final = 5
_MT_TAG: Final = 6
_MT_SIMPLE: Final = 7

_UINT64_MAX: Final = (1 << 64) - 1
_TAG_POS_BIGNUM: Final = 2
_TAG_NEG_BIGNUM: Final = 3

# Simple-value initial bytes (major type 7).
_FALSE: Final = (_MT_SIMPLE << 5) | 20
_TRUE: Final = (_MT_SIMPLE << 5) | 21
_NULL: Final = (_MT_SIMPLE << 5) | 22
_FLOAT64: Final = (_MT_SIMPLE << 5) | 27

# Bound on recursion depth: defends against pathological nesting and against
# accidental reference cycles (which a content-addressed value can never contain,
# but a caller's malformed input might).
_MAX_DEPTH: Final = 256


def canonical_cbor(value: CanonicalValue) -> bytes:
    """Encode ``value`` to its canonical CBOR byte string, or raise.

    The result is deterministic: equal inputs (up to the normalisations described
    in the module docstring) always produce identical bytes.
    """
    out = bytearray()
    _encode(value, out, depth=0)
    return bytes(out)


def _encode(value: object, out: bytearray, depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise CanonicalEncodingError("maximum nesting depth exceeded")
    # Order matters: bool must be tested before int because bool subclasses int.
    if value is None:
        out.append(_NULL)
    elif isinstance(value, bool):
        out.append(_TRUE if value else _FALSE)
    elif isinstance(value, int):
        _encode_int(value, out)
    elif isinstance(value, float):
        _encode_float(value, out)
    elif isinstance(value, str):
        _encode_text(value, out)
    elif isinstance(value, (bytes, bytearray)):
        _encode_byte_string(bytes(value), out)
    elif isinstance(value, Mapping):
        _encode_map(value, out, depth)
    elif isinstance(value, (list, tuple)):
        _encode_array(value, out, depth)
    else:
        raise CanonicalEncodingError(f"unsupported type: {type(value).__name__}")


def _encode_head(major: int, arg: int, out: bytearray) -> None:
    """Emit the initial byte plus minimal-length argument for ``major``/``arg``."""
    prefix = major << 5
    if arg < 24:
        out.append(prefix | arg)
    elif arg < 0x100:
        out.append(prefix | 24)
        out.append(arg)
    elif arg < 0x10000:
        out.append(prefix | 25)
        out.extend(struct.pack(">H", arg))
    elif arg < 0x100000000:
        out.append(prefix | 26)
        out.extend(struct.pack(">I", arg))
    elif arg <= _UINT64_MAX:
        out.append(prefix | 27)
        out.extend(struct.pack(">Q", arg))
    else:  # pragma: no cover - guarded by callers
        raise CanonicalEncodingError("argument does not fit in a 64-bit head")


def _encode_int(value: int, out: bytearray) -> None:
    if value >= 0:
        if value <= _UINT64_MAX:
            _encode_head(_MT_UINT, value, out)
        else:
            _encode_bignum(_TAG_POS_BIGNUM, value, out)
    else:
        magnitude = -1 - value  # CBOR negative integers store -1 - n
        if magnitude <= _UINT64_MAX:
            _encode_head(_MT_NINT, magnitude, out)
        else:
            _encode_bignum(_TAG_NEG_BIGNUM, magnitude, out)


def _encode_bignum(tag: int, magnitude: int, out: bytearray) -> None:
    # Tags 2/3 are < 24, so the tag head is a single byte.
    out.append((_MT_TAG << 5) | tag)
    n_bytes = (magnitude.bit_length() + 7) // 8
    _encode_byte_string(magnitude.to_bytes(n_bytes, "big"), out)


def _encode_float(value: float, out: bytearray) -> None:
    if not math.isfinite(value):
        raise CanonicalEncodingError("non-finite floats are not canonical")
    if value == 0.0:
        value = 0.0  # normalise -0.0 to +0.0 (distinct bit pattern, equal value)
    out.append(_FLOAT64)
    out.extend(struct.pack(">d", value))


def _encode_text(value: str, out: bytearray) -> None:
    data = value.encode("utf-8")
    _encode_head(_MT_TEXT, len(data), out)
    out.extend(data)


def _encode_byte_string(value: bytes, out: bytearray) -> None:
    _encode_head(_MT_BYTES, len(value), out)
    out.extend(value)


def _encode_array(value: Sequence[object], out: bytearray, depth: int) -> None:
    _encode_head(_MT_ARRAY, len(value), out)
    for item in value:
        _encode(item, out, depth + 1)


def _encode_map(value: Mapping[object, object], out: bytearray, depth: int) -> None:
    encoded: list[tuple[bytes, bytes]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise CanonicalEncodingError("map keys must be str")
        key_buf = bytearray()
        _encode_text(key, key_buf)
        item_buf = bytearray()
        _encode(item, item_buf, depth + 1)
        encoded.append((bytes(key_buf), bytes(item_buf)))
    # Distinct str keys produce distinct encoded keys, so this order is total.
    encoded.sort(key=lambda pair: pair[0])
    _encode_head(_MT_MAP, len(encoded), out)
    for key_bytes, item_bytes in encoded:
        out.extend(key_bytes)
        out.extend(item_bytes)
