"""Content hashing and self-describing digests (multihash) for node identity.

A node id is a multihash: ``uvarint(code) || uvarint(length) || digest``
(the multiformats convention). Carrying the algorithm code in the id itself means
a recording made under one hash function is never silently confused with another,
and migrating the production hash is a detectable, mechanical change rather than a
silent break in content-addressing.

The fixed default is BLAKE2b-256 from the standard library, so node ids are
reproducible on any interpreter with no native build step. BLAKE3 -- the intended
production hash for its speed and parallelism -- is provided behind the same
``HashAlgo`` protocol and is opt-in via the ``blake3`` extra. The default is fixed
(never "BLAKE3 if importable"): an environment-dependent default would make node
ids non-reproducible, which is a bug for a content-addressed store.
"""

from __future__ import annotations

import hashlib
from typing import Final, Protocol, runtime_checkable

__all__ = [
    "Blake2b256",
    "Blake3",
    "HashAlgo",
    "default_hash",
    "multihash",
    "read_uvarint",
    "uvarint",
]

# multicodec codes (the multiformats table).
_CODE_BLAKE3: Final = 0x1E
_CODE_BLAKE2B_256: Final = 0xB220


@runtime_checkable
class HashAlgo(Protocol):
    """A content hash function tagged with its multicodec identity."""

    @property
    def multicodec(self) -> int: ...

    @property
    def digest_size(self) -> int: ...

    def digest(self, data: bytes) -> bytes: ...


class Blake2b256:
    """BLAKE2b truncated to 256 bits (standard library; the fixed default)."""

    multicodec: Final = _CODE_BLAKE2B_256
    digest_size: Final = 32

    def digest(self, data: bytes) -> bytes:
        return hashlib.blake2b(data, digest_size=32).digest()


class Blake3:
    """BLAKE3-256 (requires the ``blake3`` extra; the intended production hash)."""

    multicodec: Final = _CODE_BLAKE3
    digest_size: Final = 32

    def digest(self, data: bytes) -> bytes:
        from blake3 import blake3  # imported lazily so the extra stays optional

        return bytes(blake3(data).digest(length=32))


_DEFAULT: Final[HashAlgo] = Blake2b256()


def default_hash() -> HashAlgo:
    """Return the fixed default hash algorithm used for node identity."""
    return _DEFAULT


def uvarint(value: int) -> bytes:
    """Encode a non-negative integer as an unsigned LEB128 varint."""
    if value < 0:
        raise ValueError("uvarint cannot encode a negative integer")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def read_uvarint(data: bytes, offset: int = 0) -> tuple[int, int]:
    """Decode an unsigned LEB128 varint, returning ``(value, next_offset)``."""
    value = 0
    shift = 0
    pos = offset
    while True:
        if pos >= len(data):
            raise ValueError("truncated uvarint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def multihash(algo: HashAlgo, data: bytes) -> bytes:
    """Hash ``data`` with ``algo`` and wrap it as a self-describing multihash."""
    digest = algo.digest(data)
    if len(digest) != algo.digest_size:
        raise ValueError("hash algorithm returned an unexpected digest size")
    return uvarint(algo.multicodec) + uvarint(algo.digest_size) + digest
