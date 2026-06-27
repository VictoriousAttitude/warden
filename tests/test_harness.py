"""Tests for the Harness recorder (M2.2, arch section 10).

Two layers: the recorder in isolation (it commits content nodes, advances the head,
numbers boundaries monotonically, and persists to an ObjectStore), and the recorder
driven by a live Guard run (it captures the provenance DAG -- sources, a TOOL_CALL
request node, and the TOOL_RESULT it produced -- and a cassette keyed by the request
id). The recorder records Layer A only; no labels leak into it (INV-9).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warden import Guard, Recorder
from warden.core import Node, NodeKind, ObjectStore
from warden.harness import BoundaryEvent


def test_empty_recording_has_no_head_or_events() -> None:
    rec = Recorder().recording()
    assert rec.head is None
    assert rec.events == ()
    assert rec.cassette() == {}
    with pytest.raises(ValueError):
        rec.run()


def test_record_source_commits_and_advances_head() -> None:
    recorder = Recorder()
    node = Node(NodeKind.USER_INPUT, (), "hi")
    assert recorder.record_source(node) == node.id
    rec = recorder.recording()
    assert rec.head == node.id
    assert node.id in rec.graph
    assert rec.events == ()


def test_record_boundary_builds_event_and_cassette() -> None:
    recorder = Recorder()
    call = Node(NodeKind.TOOL_CALL, (), {"tool": "fetch", "args": "()"})
    result = Node(NodeKind.TOOL_RESULT, (call.id,), "<web>")
    event = recorder.record_boundary(call, result)
    assert event == BoundaryEvent(1, call.id, result.id)
    rec = recorder.recording()
    assert rec.head == result.id
    assert rec.cassette() == {call.id: result.id}
    assert rec.run().head == result.id


def test_sequence_numbers_are_monotonic() -> None:
    recorder = Recorder()
    seqs: list[int] = []
    for i in range(3):
        call = Node(NodeKind.TOOL_CALL, (), {"tool": "t", "args": str(i)})
        result = Node(NodeKind.TOOL_RESULT, (call.id,), str(i))
        seqs.append(recorder.record_boundary(call, result).seq)
    assert seqs == [1, 2, 3]


def test_store_persists_recorded_nodes(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path)
    recorder = Recorder(store)
    call = Node(NodeKind.TOOL_CALL, (), {"tool": "fetch", "args": "()"})
    result = Node(NodeKind.TOOL_RESULT, (call.id,), "<web>")
    recorder.record_boundary(call, result)
    assert store.get_node(call.id).id == call.id
    assert store.get_node(result.id).id == result.id


def test_guard_records_the_full_provenance_dag(tmp_path: Path) -> None:
    recorder = Recorder(ObjectStore(tmp_path))
    guard = Guard(
        "allow send_email if body.integrity == trusted",
        recorder=recorder,
    )

    @guard.tool
    def send_email(body: str, recipient: str) -> str:
        return f"sent to {recipient}"

    body = guard.source("hi Alice")
    recipient = guard.source("alice@corp")
    receipt = send_email(body=body, recipient=recipient)

    rec = recorder.recording()
    # Two source nodes, one TOOL_CALL, one TOOL_RESULT.
    assert len(rec.graph) == 4
    assert len(rec.events) == 1
    event = rec.events[0]
    assert event.seq == 1
    assert event.result == receipt.id
    assert rec.head == receipt.id
    assert rec.cassette() == {event.call: event.result}

    result_node = rec.graph.get(receipt.id)
    assert result_node.kind is NodeKind.TOOL_RESULT
    assert result_node.parents == (event.call,)

    call_node = rec.graph.get(event.call)
    assert call_node.kind is NodeKind.TOOL_CALL
    # The request commits to its labeled inputs, in parameter order.
    assert call_node.parents == (body.id, recipient.id)


def test_identical_calls_share_one_cassette_entry() -> None:
    recorder = Recorder()
    guard = Guard("allow echo if text.integrity == trusted", recorder=recorder)

    @guard.tool(name="echo")
    def echo(text: str) -> str:
        return text

    arg = guard.source("same")
    echo(text=arg)
    echo(text=arg)

    rec = recorder.recording()
    assert len(rec.events) == 2
    # Byte-identical requests hash to one cassette key (content-addressed dedup).
    assert len(rec.cassette()) == 1
