"""Provenance nodes and their content-addressed identity (Layer A).

A run is a directed acyclic graph of content nodes. This module defines the node
schema and the one function that turns content into identity. Per finding F1 in
WARDEN_ARCHITECTURE_v0.1.txt, identity is computed over CONTENT ONLY -- never over
security labels, which are a separate derived overlay (Layer B). This is what lets
the replay cache and cassette dedup key on content regardless of policy.

The identity preimage is a canonical envelope:

    {"kind": <int>, "parents": [<parent multihash bytes>, ...], "payload": <value>}

The integer ``kind`` tag provides domain separation: a ToolResult and a UserInput
carrying byte-identical payloads receive different ids, so one node kind can never
be substituted for another by content collision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

from warden.core.canonical import CanonicalValue, canonical_cbor
from warden.core.decode import decode_canonical
from warden.core.hashing import HashAlgo, default_hash, multihash

__all__ = ["Node", "NodeId", "NodeKind", "compute_node_id", "decode_node", "encode_node"]


class NodeKind(IntEnum):
    """The kinds of provenance node.

    These integer values are part of node identity and MUST remain stable: never
    renumber or reuse a value, or previously computed ids would change.
    """

    USER_INPUT = 1
    SYSTEM_PROMPT = 2
    LLM_REQUEST = 3
    LLM_RESPONSE = 4
    TOOL_CALL = 5
    TOOL_RESULT = 6
    AGENT_STATE = 7
    FINAL_OUTPUT = 8
    DECLASSIFICATION = 9


@dataclass(frozen=True, slots=True, order=True)
class NodeId:
    """A self-describing content hash (a multihash) identifying a node.

    Ordering is bytewise over the multihash, which gives a total, deterministic
    tie-break for canonical topological ordering.
    """

    multihash: bytes

    def hex(self) -> str:
        return self.multihash.hex()

    def __repr__(self) -> str:
        return f"NodeId({self.multihash[:6].hex()}\u2026)"


def _envelope(
    kind: NodeKind, parents: tuple[NodeId, ...], payload: CanonicalValue
) -> dict[str, CanonicalValue]:
    return {
        "kind": int(kind),
        "parents": [parent.multihash for parent in parents],
        "payload": payload,
    }


def encode_node(node: Node) -> bytes:
    """Return the canonical byte preimage of a node (the bytes its id hashes)."""
    return canonical_cbor(_envelope(node.kind, node.parents, node.payload))


def decode_node(data: bytes) -> Node:
    """Rehydrate a node from its canonical byte preimage (inverse of ``encode_node``).

    ``data`` must be the canonical envelope ``encode_node`` produced; the input is
    validated as canonical (``decode_canonical``) and structurally, so corrupt or
    forged bytes raise rather than yielding a malformed node. The returned node
    recomputes its own id, so callers can confirm it matches the key it was read
    under (see ``ObjectStore.get_node``).
    """
    envelope = decode_canonical(data)
    if not isinstance(envelope, Mapping):
        raise ValueError("node preimage is not a map")
    if "payload" not in envelope:
        raise ValueError("node preimage is missing 'payload'")
    kind_raw = envelope.get("kind")
    parents_raw = envelope.get("parents")
    if isinstance(kind_raw, bool) or not isinstance(kind_raw, int):
        raise ValueError("node 'kind' must be an integer")
    if isinstance(parents_raw, (str, bytes)) or not isinstance(parents_raw, Sequence):
        raise ValueError("node 'parents' must be a list")
    parents: list[NodeId] = []
    for parent in parents_raw:
        if not isinstance(parent, bytes):
            raise ValueError("each parent must be a multihash byte string")
        parents.append(NodeId(parent))
    return Node(NodeKind(kind_raw), tuple(parents), envelope["payload"])


def compute_node_id(
    kind: NodeKind,
    parents: tuple[NodeId, ...],
    payload: CanonicalValue,
    algo: HashAlgo | None = None,
) -> NodeId:
    """Compute the content-addressed identity of a node.

    Parent order is significant: it is part of the canonical envelope, so callers
    must present parents in a stable, meaningful order.
    """
    algo = algo or default_hash()
    return NodeId(multihash(algo, canonical_cbor(_envelope(kind, parents, payload))))


@dataclass(frozen=True, slots=True, eq=False)
class Node:
    """An immutable content node. Equality and hashing are by content id."""

    kind: NodeKind
    parents: tuple[NodeId, ...]
    payload: CanonicalValue
    id: NodeId = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "id", compute_node_id(self.kind, self.parents, self.payload)
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
