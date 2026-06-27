"""Deterministic replay of a recorded run against its cassette.

Replay re-runs the agent's own logic but, at every tool boundary, serves the
recorded result instead of executing the tool. This is the operational form of the
Determinism Theorem (arch section 10.1): if boundary results are fixed by the
request's content id, delivered in the recorded logical-sequence order, and the
non-boundary code is deterministic, the replayed run is byte-identical to the
original -- same node ids, same topological order.

The ``Replayer`` is the logical-sequence boundary scheduler (finding F6) in its
synchronous projection: it holds a cursor over the recording's boundary events and,
for each presented request, requires it to match the next recorded boundary in
order. Two failure modes both fail closed (INV-5):

* a request that does not match the expected boundary (a cassette miss, or an
  out-of-order / divergent call), and
* a request after the recording is exhausted.

Because ``resolve`` returns the recorded result without running the tool, a
consequential tool's side effect never recurs on replay (INV-8).

A ``CounterfactualReplayer`` perturbs the recording: it substitutes one boundary's
result and replays the rest, asking "what would the agent have done if this tool had
returned something else?". It is the adversarial fuzzer -- inject hostile content at
a boundary and watch where the flow ends up. The Guard consults either replayer
through the ``BoundaryOracle`` seam.
"""

from __future__ import annotations

from typing import Protocol

from warden.core import CanonicalValue, Node, NodeKind
from warden.harness.record import Recording

__all__ = [
    "BoundaryOracle",
    "CounterfactualReplayer",
    "ReplayError",
    "Replayer",
]


class BoundaryOracle(Protocol):
    """What the Guard consults at each boundary.

    ``resolve`` returns the result node to serve for ``call``, or ``None`` to signal
    "no recorded answer -- run the tool live". A strict ``Replayer`` never returns
    ``None`` (it fails closed instead); a ``CounterfactualReplayer`` returns ``None``
    once the perturbation has diverged, so the Guard re-executes the suffix.
    """

    def resolve(self, call: Node) -> Node | None: ...


class ReplayError(Exception):
    """A replayed run diverged from its recording or hit a cassette miss (INV-5)."""


class Replayer:
    """Serves recorded boundary results in recorded logical-sequence order."""

    __slots__ = ("_cursor", "_recording")

    def __init__(self, recording: Recording) -> None:
        self._recording = recording
        self._cursor = 0

    def resolve(self, call: Node) -> Node:
        """Return the recorded result node for ``call``, advancing the schedule.

        ``call`` must equal the next expected boundary's request; otherwise replay
        has diverged and we fail closed rather than re-executing the tool.
        """
        events = self._recording.events
        if self._cursor >= len(events):
            raise ReplayError(
                f"replay diverged: boundary {call.id!r} after the recording was exhausted"
            )
        event = events[self._cursor]
        if call.id != event.call:
            raise ReplayError(
                f"replay diverged at seq {event.seq}: "
                f"expected request {event.call!r}, got {call.id!r}"
            )
        self._cursor += 1
        return self._recording.graph.get(event.result)

    @property
    def exhausted(self) -> bool:
        """True once every recorded boundary has been served."""
        return self._cursor == len(self._recording.events)


class CounterfactualReplayer:
    """Replays a recording with one boundary's result substituted (the counterfactual).

    This is fork + suffix-replay made operational. Boundaries before the injection are
    served from the cassette -- the shared, unchanged prefix. The injected boundary
    returns a substituted result whose new content id forces every dependent request
    to change. From the first request that no longer matches the recording onward,
    ``resolve`` returns ``None`` so the Guard re-executes the real tools: the suffix is
    re-run under the perturbation. ``diff`` (arch finding F7) then locates the first
    divergence between the baseline and counterfactual runs.

    WARNING: the suffix runs LIVE -- consequential tools' side effects DO happen. Run
    counterfactuals only against tools that are safe to execute (a sandbox), exactly as
    you would any adversarial fuzz harness.
    """

    __slots__ = ("_cursor", "_diverged", "_payload", "_recording", "_target_seq")

    def __init__(self, recording: Recording, *, at: int, payload: CanonicalValue) -> None:
        """Perturb the boundary at logical sequence ``at`` to return ``payload``."""
        if not 1 <= at <= len(recording.events):
            raise ValueError(
                f"injection seq {at} is out of range [1, {len(recording.events)}]"
            )
        self._recording = recording
        self._target_seq = at
        self._payload = payload
        self._cursor = 0
        self._diverged = False

    def resolve(self, call: Node) -> Node | None:
        """Serve the recorded result, the substituted result, or ``None`` to run live."""
        if self._diverged or self._cursor >= len(self._recording.events):
            return None
        event = self._recording.events[self._cursor]
        if call.id != event.call:
            # The perturbation changed this request; re-execute from here on.
            self._diverged = True
            return None
        self._cursor += 1
        if event.seq == self._target_seq:
            return Node(NodeKind.TOOL_RESULT, (call.id,), self._payload)
        return self._recording.graph.get(event.result)
