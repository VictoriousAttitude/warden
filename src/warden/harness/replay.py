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
"""

from __future__ import annotations

from warden.core import Node
from warden.harness.record import Recording

__all__ = ["ReplayError", "Replayer"]


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
