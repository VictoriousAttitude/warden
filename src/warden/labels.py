"""Label algebra: the product join-semilattice the Guard propagates (Layer B).

This module is pure algebra -- no graph, no policy, no I/O. It defines the security
label attached to every value and the single ``join`` that combines labels when a
value is derived from others. See WARDEN_ARCHITECTURE_v0.1.txt section 5.

Single propagation orientation (finding F2). Combining values always moves UP the
lattice toward the more-restrictive label, so ``label(derived) = join(parents)``
holds uniformly on every axis and monotonicity (INV-3) is true by construction.
Internally, therefore:

  * integrity is a TAINT level oriented TRUSTED (bottom) below UNTRUSTED (top);
  * confidentiality is oriented PUBLIC (bottom) below SECRET (top);
  * provenance is a set that only grows (join = union).

The policy DSL reads integrity back in the human "trusted is good" direction
(section 5.3); that translation happens at the monitor boundary, never in here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum

__all__ = [
    "Confidentiality",
    "Label",
    "SourceId",
    "Taint",
    "join_all",
]

type SourceId = str
"""A provenance source identifier (e.g. a tool or origin name)."""


class Taint(IntEnum):
    """Integrity oriented for propagation: TRUSTED below UNTRUSTED, top = UNTRUSTED.

    The integer order IS the lattice order, so ``join`` is ``max``: combining a
    trusted value with an untrusted one yields untrusted. Values are stable.
    """

    TRUSTED = 0
    UNTRUSTED = 1


class Confidentiality(IntEnum):
    """Confidentiality levels: PUBLIC below INTERNAL below SECRET, top = SECRET.

    The integer order IS the lattice order, so ``join`` is ``max``: combining a
    public value with a secret one yields secret. Values are stable.
    """

    PUBLIC = 0
    INTERNAL = 1
    SECRET = 2


@dataclass(frozen=True, slots=True)
class Label:
    """A security label: the product of the three component lattices.

    Equality is structural. The partial order ``<=`` is componentwise and is a
    genuine partial order (provenance compares by subset), NOT a total order.
    """

    integrity: Taint = Taint.TRUSTED
    confidentiality: Confidentiality = Confidentiality.PUBLIC
    provenance: frozenset[SourceId] = field(default_factory=frozenset)

    @classmethod
    def bottom(cls) -> Label:
        """The identity element of ``join``: trusted, public, no provenance."""
        return cls(Taint.TRUSTED, Confidentiality.PUBLIC, frozenset())

    def join(self, other: Label) -> Label:
        """The least upper bound: most-restrictive componentwise (commutative)."""
        return Label(
            integrity=Taint(max(self.integrity, other.integrity)),
            confidentiality=Confidentiality(
                max(self.confidentiality, other.confidentiality)
            ),
            provenance=self.provenance | other.provenance,
        )

    def leq(self, other: Label) -> bool:
        """Partial order: ``self`` is no more restrictive than ``other`` on every axis."""
        return (
            self.integrity <= other.integrity
            and self.confidentiality <= other.confidentiality
            and self.provenance <= other.provenance
        )

    def __le__(self, other: Label) -> bool:
        return self.leq(other)


def join_all(labels: Iterable[Label]) -> Label:
    """Fold ``join`` over ``labels``, starting from ``Label.bottom()``.

    The empty join is ``Label.bottom()`` (the lattice identity), so a value with no
    labeled parents and no source label is trusted and public.
    """
    acc = Label.bottom()
    for label in labels:
        acc = acc.join(label)
    return acc
