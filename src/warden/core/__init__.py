"""Warden Core: the immutable, content-addressed provenance substrate (Layer A).

This package carries no security logic and no record/replay logic. It provides the
canonical encoding, content hashing, and node identity that both the Guard and the
Harness build on. See WARDEN_ARCHITECTURE_v0.1.txt section 4.
"""

from warden.core.canonical import (
    CanonicalEncodingError,
    CanonicalValue,
    canonical_cbor,
)
from warden.core.graph import Graph, MissingParentError, UnknownNodeError
from warden.core.hashing import (
    Blake2b256,
    Blake3,
    HashAlgo,
    default_hash,
    multihash,
)
from warden.core.nodes import Node, NodeId, NodeKind, compute_node_id, encode_node
from warden.core.run import Divergence, Run, RunManifest, diff, fork
from warden.core.store import IntegrityError, ObjectStore

__all__ = [
    "Blake2b256",
    "Blake3",
    "CanonicalEncodingError",
    "CanonicalValue",
    "Divergence",
    "Graph",
    "HashAlgo",
    "IntegrityError",
    "MissingParentError",
    "Node",
    "NodeId",
    "NodeKind",
    "ObjectStore",
    "Run",
    "RunManifest",
    "UnknownNodeError",
    "canonical_cbor",
    "compute_node_id",
    "default_hash",
    "diff",
    "encode_node",
    "fork",
    "multihash",
]
