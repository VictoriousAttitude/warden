"""A content-addressed object store on the local filesystem.

Objects are keyed by the multihash of their bytes and laid out git-style with a
two-character fan-out: ``<root>/objects/<hh>/<full-hash-hex>``. Writes are atomic
(temp file + fsync + rename) and idempotent (identical content writes the same
bytes to the same path), so concurrent writers cannot conflict. Reads verify the
stored bytes against the requested key, turning silent on-disk corruption into a
loud error.

On-disk bytes are zlib-compressed (standard library, zero extra dependency). The
hash is always computed over the UNCOMPRESSED canonical content, never the
compressed form, so compression is purely a storage detail and can change without
affecting identity.
"""

from __future__ import annotations

import os
import zlib
from pathlib import Path

from warden.core.hashing import HashAlgo, default_hash, multihash
from warden.core.nodes import Node, NodeId, decode_node, encode_node

__all__ = ["IntegrityError", "ObjectStore"]


class IntegrityError(Exception):
    """Stored bytes do not hash to the key under which they were requested."""


class ObjectStore:
    """A filesystem-backed, content-addressed blob store."""

    def __init__(self, root: str | os.PathLike[str], algo: HashAlgo | None = None) -> None:
        self._root = Path(root)
        self._algo = algo or default_hash()
        (self._root / "objects").mkdir(parents=True, exist_ok=True)

    def _path(self, key: NodeId) -> Path:
        digest = key.hex()
        return self._root / "objects" / digest[:2] / digest

    def put(self, content: bytes) -> NodeId:
        """Store ``content`` and return its content-addressed key."""
        key = NodeId(multihash(self._algo, content))
        path = self._path(key)
        if path.exists():
            return key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f"{path.name}.tmp-{os.urandom(8).hex()}"
        with open(tmp, "wb") as handle:
            handle.write(zlib.compress(content))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return key

    def put_node(self, node: Node) -> NodeId:
        """Store a node by its canonical preimage; the key equals ``node.id``."""
        return self.put(encode_node(node))

    def has(self, key: NodeId) -> bool:
        return self._path(key).exists()

    def get(self, key: NodeId) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise KeyError(key)
        content = zlib.decompress(path.read_bytes())
        if NodeId(multihash(self._algo, content)) != key:
            raise IntegrityError(f"content at {path} does not match {key!r}")
        return content

    def get_node(self, key: NodeId) -> Node:
        """Read and rehydrate a node, verifying its recomputed id matches ``key``."""
        node = decode_node(self.get(key))
        if node.id != key:
            raise IntegrityError(f"rehydrated node id does not match {key!r}")
        return node
