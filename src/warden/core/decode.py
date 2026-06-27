"""Decoder for the Warden CBOR profile: the inverse of ``canonical_cbor``.

Node identity only ever needs the ENCODER (``canonical.py``); this module is what
the Harness uses to read content BACK -- to rehydrate node payloads and recorded
boundary results from the object store. It is deliberately separate from the
encoder so that the encoder stays a small, auditable trust-boundary file.

The decoder accepts the profile's closed value domain and nothing else: it rejects
indefinite-length items, reserved additional-information values, unsupported simple
values and tags, and non-text map keys. ``decode_canonical`` goes further: it
re-encodes the decoded value and checks the result reproduces the input bytes, so
non-canonical encodings (non-shortest integers, unsorted or duplicate map keys,
negative zero, ...) are rejected. Thus for any ``b`` that ``decode_canonical``
accepts, ``canonical_cbor(decode_canonical(b)) == b`` holds by construction --
which is exactly the integrity guarantee a content-addressed store relies on.
"""

from __future__ import annotations

import struct
from typing import Final

from warden.core.canonical import (
    CanonicalEncodingError,
    CanonicalValue,
    canonical_cbor,
)

__all__ = ["CborDecodeError", "decode_canonical", "loads"]

_MT_UINT: Final = 0
_MT_NINT: Final = 1
_MT_BYTES: Final = 2
_MT_TEXT: Final = 3
_MT_ARRAY: Final = 4
_MT_MAP: Final = 5
_MT_TAG: Final = 6
_MT_SIMPLE: Final = 7

_TAG_POS_BIGNUM: Final = 2
_TAG_NEG_BIGNUM: Final = 3

_FALSE_INFO: Final = 20
_TRUE_INFO: Final = 21
_NULL_INFO: Final = 22
_FLOAT64_INFO: Final = 27

_INDEFINITE_INFO: Final = 31

# Mirror the encoder's bound so pathological nesting cannot exhaust the stack.
_MAX_DEPTH: Final = 256


class CborDecodeError(ValueError):
    """Bytes are not a valid encoding in the Warden CBOR value domain."""


class _Reader:
    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def at_end(self) -> bool:
        return self._pos == len(self._data)

    def _take(self, n: int) -> bytes:
        end = self._pos + n
        if end > len(self._data):
            raise CborDecodeError("unexpected end of input")
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def _byte(self) -> int:
        return self._take(1)[0]

    def _argument(self, info: int) -> int:
        if info < 24:
            return info
        if info == 24:
            return self._byte()
        if info == 25:
            return int.from_bytes(self._take(2), "big")
        if info == 26:
            return int.from_bytes(self._take(4), "big")
        if info == 27:
            return int.from_bytes(self._take(8), "big")
        raise CborDecodeError(f"reserved additional information {info}")

    def read(self, depth: int = 0) -> CanonicalValue:
        if depth > _MAX_DEPTH:
            raise CborDecodeError("maximum nesting depth exceeded")
        initial = self._byte()
        major = initial >> 5
        info = initial & 0x1F
        if major == _MT_SIMPLE:
            return self._read_simple(info)
        if info == _INDEFINITE_INFO:
            raise CborDecodeError("indefinite-length items are not allowed")
        arg = self._argument(info)
        if major == _MT_UINT:
            return arg
        if major == _MT_NINT:
            return -1 - arg
        if major == _MT_BYTES:
            return self._take(arg)
        if major == _MT_TEXT:
            return self._read_text(arg)
        if major == _MT_ARRAY:
            return [self.read(depth + 1) for _ in range(arg)]
        if major == _MT_MAP:
            return self._read_map(arg, depth)
        # major == _MT_TAG (the only remaining 3-bit value)
        return self._read_bignum(arg, depth)

    def _read_simple(self, info: int) -> CanonicalValue:
        if info == _FALSE_INFO:
            return False
        if info == _TRUE_INFO:
            return True
        if info == _NULL_INFO:
            return None
        if info == _FLOAT64_INFO:
            return float(struct.unpack(">d", self._take(8))[0])
        raise CborDecodeError(f"unsupported simple or float value {info}")

    def _read_text(self, length: int) -> str:
        try:
            return self._take(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise CborDecodeError("invalid UTF-8 in text string") from error

    def _read_map(self, count: int, depth: int) -> dict[str, CanonicalValue]:
        result: dict[str, CanonicalValue] = {}
        for _ in range(count):
            key = self.read(depth + 1)
            if not isinstance(key, str):
                raise CborDecodeError("map keys must be text strings")
            result[key] = self.read(depth + 1)
        return result

    def _read_bignum(self, tag: int, depth: int) -> int:
        if tag not in (_TAG_POS_BIGNUM, _TAG_NEG_BIGNUM):
            raise CborDecodeError(f"unsupported tag {tag}")
        payload = self.read(depth + 1)
        if not isinstance(payload, bytes):
            raise CborDecodeError("bignum payload must be a byte string")
        magnitude = int.from_bytes(payload, "big")
        return magnitude if tag == _TAG_POS_BIGNUM else -1 - magnitude


def loads(data: bytes) -> CanonicalValue:
    """Decode one item from ``data`` in the Warden CBOR value domain.

    Raises ``CborDecodeError`` on malformed input or trailing bytes. This does NOT
    check that ``data`` is in canonical form -- use ``decode_canonical`` for that.
    """
    reader = _Reader(data)
    value = reader.read()
    if not reader.at_end():
        raise CborDecodeError("trailing bytes after top-level item")
    return value


def decode_canonical(data: bytes) -> CanonicalValue:
    """Decode ``data`` and verify it is in canonical form, or raise.

    Guarantees ``canonical_cbor(decode_canonical(b)) == b`` for every accepted
    ``b``: the decoded value is re-encoded and checked byte-for-byte, so any
    non-canonical encoding is rejected rather than silently normalized.
    """
    value = loads(data)
    try:
        reencoded = canonical_cbor(value)
    except CanonicalEncodingError as error:
        raise CborDecodeError(f"decoded value is not canonical: {error}") from error
    if reencoded != data:
        raise CborDecodeError("input is not in canonical form")
    return value
