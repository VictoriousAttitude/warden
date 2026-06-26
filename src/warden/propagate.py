"""Eager label propagation over the content DAG (Layer B). See arch section 6.

Propagation reads the immutable content DAG (Layer A) and produces a LabelOverlay:
a map ``NodeId -> Label`` for one (run, policy version). It never mutates nodes
(INV-1). A node's label is its source label joined with the labels of its parents,
so by construction ``label(node) >= join(parents)`` -- INV-3 monotonicity. The one
sanctioned break is a Declassification node, where an audited authority sets a
lower label explicitly (section 5.4).

The crux of label creep (finding F5) lives in the PropagationStrategy seam. M1
ships WHOLE_CONTEXT: sound and conservative (no laundering escapes the join) at the
cost of higher creep. M3 can swap in a handle-based strategy without touching the
monitor or policy. The value environment (CaMeL-style handles) data model lives
here from M1 so that swap is purely additive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from warden.core import Graph, Node, NodeId, NodeKind
from warden.labels import Label, join_all

__all__ = [
    "WHOLE_CONTEXT",
    "Handle",
    "LabelOverlay",
    "PropagationStrategy",
    "SourceLabels",
    "UnlabeledError",
    "ValueEnv",
    "WholeContext",
    "propagate",
]

type SourceLabels = Mapping[NodeId, Label]
"""Intrinsic labels declared for boundary nodes (e.g. an untrusted ToolResult)."""

type Handle = str
"""An opaque reference to a value held in the value environment."""


class UnlabeledError(KeyError):
    """A label was requested for a node or handle that has none (fail-closed)."""


@runtime_checkable
class PropagationStrategy(Protocol):
    """How a node's label is derived from its parents' labels and its source label."""

    def node_label(
        self, node: Node, parent_labels: Sequence[Label], source_label: Label
    ) -> Label: ...


@dataclass(frozen=True, slots=True)
class WholeContext:
    """M1 default: a derived value carries the join of every input plus its source.

    Sound and conservative: because every parent's label is joined in, taint cannot
    be laundered out of a derivation. The cost is creep, which is the release-gated
    metric the strategy seam exists to let us trade away later (section 6.2).
    """

    def node_label(
        self, node: Node, parent_labels: Sequence[Label], source_label: Label
    ) -> Label:
        return join_all(parent_labels).join(source_label)


WHOLE_CONTEXT: Final[PropagationStrategy] = WholeContext()


@dataclass(frozen=True, slots=True)
class LabelOverlay:
    """A derived map ``NodeId -> Label`` for one run under one policy version."""

    _labels: Mapping[NodeId, Label]

    def label_of(self, node_id: NodeId) -> Label:
        """Return the propagated label, or raise UnlabeledError (fail-closed)."""
        try:
            return self._labels[node_id]
        except KeyError:
            raise UnlabeledError(node_id) from None

    def get(self, node_id: NodeId, default: Label) -> Label:
        return self._labels.get(node_id, default)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._labels

    def __len__(self) -> int:
        return len(self._labels)


@dataclass(frozen=True, slots=True)
class ValueEnv:
    """CaMeL-style binding of opaque handles to (content node, label) pairs.

    The monitor resolves a tool argument that is a handle reference to the label of
    the value it points at -- per-value labeling rather than per-context. In M1 the
    planner may still see raw content (so WHOLE_CONTEXT is the sound backstop); in M3
    the planner sees only handles and this becomes the low-creep path.
    """

    bindings: Mapping[Handle, tuple[NodeId, Label]] = field(default_factory=dict)

    def resolve(self, handle: Handle) -> tuple[NodeId, Label]:
        try:
            return self.bindings[handle]
        except KeyError:
            raise UnlabeledError(handle) from None

    def label_of(self, handle: Handle) -> Label:
        return self.resolve(handle)[1]


def propagate(
    graph: Graph,
    sources: SourceLabels | None = None,
    strategy: PropagationStrategy = WHOLE_CONTEXT,
) -> LabelOverlay:
    """Compute the label overlay for ``graph`` in canonical topological order.

    Parents are labeled before children (canonical topo order), so each node's
    parent labels are already final when it is visited. A Declassification node
    takes its label directly from ``sources`` (the audited authority's decision);
    absent such a declaration it falls back to the normal join, so a missing
    authority can only over-label, never silently lower (fail-closed, INV-5).
    """
    declared = sources or {}
    labels: dict[NodeId, Label] = {}
    for node_id in graph.topo_order():
        node = graph.get(node_id)
        source_label = declared.get(node_id, Label.bottom())
        if node.kind is NodeKind.DECLASSIFICATION and node_id in declared:
            labels[node_id] = source_label
            continue
        parent_labels = [labels[parent] for parent in node.parents]
        labels[node_id] = strategy.node_label(node, parent_labels, source_label)
    return LabelOverlay(labels)
