"""Record a live execution as an immutable content DAG plus a replay cassette.

A recording captures Layer A only: the content nodes a run produced and, for each
mediated tool boundary, an event binding the request node to its response node. It
holds no security labels -- those are the Guard's derived overlay (Layer B), re-
derived on replay -- so the recorder depends only on ``warden.core`` and the Guard
never bleeds into the Harness (INV-9).

Two structures carry the recording:

* The **graph** is the provenance DAG of every node the run committed, optionally
  persisted to a content-addressed ``ObjectStore`` so a recording survives the
  process that made it.
* The **cassette** is the replay table: ``{call_id -> result_id}``. A boundary's
  call node is the content-addressed identity of the *request* (tool name + bound
  arguments); replay looks a request up by that id and serves the recorded result
  without re-executing the tool (INV-8). Because the key is a content hash, two
  byte-identical calls share one cassette entry -- the dedup the replay cache needs.

Events also carry a monotonic logical sequence number (F6): the boundary order is
the ground truth a deterministic replay must reproduce, independent of wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass

from warden.core import Graph, Node, NodeId, ObjectStore, Run

__all__ = ["BoundaryEvent", "Recorder", "Recording"]


@dataclass(frozen=True, slots=True)
class BoundaryEvent:
    """One mediated tool boundary: a request node and the result it produced.

    ``seq`` is the logical order in which boundaries occurred; ``call`` is the
    content id of the request and the cassette key, ``result`` the recorded response.
    """

    seq: int
    call: NodeId
    result: NodeId


@dataclass(frozen=True, slots=True)
class Recording:
    """The immutable artifact of a recorded run: its DAG, head, and boundaries."""

    graph: Graph
    head: NodeId | None
    events: tuple[BoundaryEvent, ...]

    def run(self) -> Run:
        """View the recording as a ``Run`` rooted at the last committed node."""
        if self.head is None:
            raise ValueError("cannot build a Run from an empty recording")
        return Run(self.graph, self.head)

    def cassette(self) -> dict[NodeId, NodeId]:
        """Return the replay table mapping each request id to its recorded result."""
        return {event.call: event.result for event in self.events}


class Recorder:
    """Accumulates a live run into a ``Graph`` (and optional store) plus events."""

    __slots__ = ("_events", "_graph", "_head", "_seq", "_store")

    def __init__(self, store: ObjectStore | None = None) -> None:
        self._graph = Graph()
        self._store = store
        self._head: NodeId | None = None
        self._seq = 0
        self._events: list[BoundaryEvent] = []

    def _commit(self, node: Node) -> NodeId:
        self._graph.add(node)
        if self._store is not None:
            self._store.put_node(node)
        self._head = node.id
        return node.id

    def record_source(self, node: Node) -> NodeId:
        """Commit a source node (a value entering the run at a trust boundary)."""
        return self._commit(node)

    def record_boundary(self, call: Node, result: Node) -> BoundaryEvent:
        """Commit a request/result pair and append the boundary event.

        The result must name the call as a parent so the DAG records that the
        response was produced by exactly this request.
        """
        self._commit(call)
        self._commit(result)
        self._seq += 1
        event = BoundaryEvent(self._seq, call.id, result.id)
        self._events.append(event)
        return event

    def recording(self) -> Recording:
        """Snapshot the run recorded so far as an immutable ``Recording``."""
        return Recording(self._graph, self._head, tuple(self._events))
