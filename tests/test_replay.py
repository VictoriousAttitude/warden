"""Tests for deterministic replay (M2.3, arch section 10.1).

The headline is the Determinism Theorem: a recorded run, re-run with its boundaries
served from the cassette in recorded order, reproduces the same content DAG byte for
byte. Around it we pin the fail-closed contracts: a consequential tool body never
re-executes on replay (INV-8), and a cassette miss, an out-of-order request, or a
request past the end of the recording all fail closed (INV-5).
"""

from __future__ import annotations

import pytest

from warden import (
    Guard,
    Handle,
    Label,
    Recorder,
    Replayer,
    ReplayError,
    Taint,
    ToolClass,
)
from warden.core import Node, NodeKind, diff
from warden.harness import BoundaryEvent, Recording

_POLICY = """
allow fetch if url.integrity == trusted
allow store_note if text.confidentiality <= internal
"""

_WEB = Label(Taint.UNTRUSTED, provenance=frozenset({"fetch_url"}))


def _wire_and_run(guard: Guard) -> tuple[Handle, list[str]]:
    """Register a fixed tool set and run a fixed two-boundary flow on ``guard``."""
    calls: list[str] = []

    @guard.tool(name="fetch", cls=ToolClass.READ_ONLY, emits=_WEB)
    def fetch(url: str) -> str:
        calls.append("fetch")
        return "<web>"

    @guard.tool(name="store_note")
    def store_note(text: str) -> str:
        calls.append("store_note")
        return "stored"

    web = fetch(guard.source("http://x"))
    note = store_note(text=web)
    return note, calls


# --- The boundary scheduler in isolation -------------------------------------


def _hand_recording() -> Recording:
    recorder = Recorder()
    call = Node(NodeKind.TOOL_CALL, (), {"tool": "fetch", "args": "()"})
    result = Node(NodeKind.TOOL_RESULT, (call.id,), "<web>")
    recorder.record_boundary(call, result)
    return recorder.recording()


def test_replayer_serves_recorded_result_and_advances() -> None:
    recording = _hand_recording()
    event = recording.events[0]
    replayer = Replayer(recording)
    assert not replayer.exhausted
    served = replayer.resolve(recording.graph.get(event.call))
    assert served.id == event.result
    assert replayer.exhausted


def test_replayer_rejects_unexpected_request() -> None:
    replayer = Replayer(_hand_recording())
    stranger = Node(NodeKind.TOOL_CALL, (), {"tool": "other", "args": "()"})
    with pytest.raises(ReplayError):
        replayer.resolve(stranger)


def test_replayer_rejects_request_past_the_end() -> None:
    recording = _hand_recording()
    replayer = Replayer(recording)
    replayer.resolve(recording.graph.get(recording.events[0].call))
    extra = Node(NodeKind.TOOL_CALL, (), {"tool": "fetch", "args": "(again)"})
    with pytest.raises(ReplayError):
        replayer.resolve(extra)


def test_out_of_order_boundary_fails_closed() -> None:
    recorder = Recorder()
    events: list[BoundaryEvent] = []
    for i in range(2):
        call = Node(NodeKind.TOOL_CALL, (), {"tool": "t", "args": str(i)})
        result = Node(NodeKind.TOOL_RESULT, (call.id,), str(i))
        events.append(recorder.record_boundary(call, result))
    recording = recorder.recording()
    replayer = Replayer(recording)
    # Present the SECOND boundary first: order diverges from the recording.
    with pytest.raises(ReplayError):
        replayer.resolve(recording.graph.get(events[1].call))


# --- Replay through the Guard ------------------------------------------------


def test_replay_reproduces_the_recorded_dag() -> None:
    record = Recorder()
    note1, _ = _wire_and_run(Guard(_POLICY, recorder=record))
    recording = record.recording()

    replay = Recorder()
    note2, _ = _wire_and_run(
        Guard(_POLICY, recorder=replay, replayer=Replayer(recording))
    )

    # Byte-stable replay (Theorem 10.1): identical content DAG, identical final id.
    assert diff(recording.run(), replay.recording().run()).identical
    assert note2.id == note1.id


def test_replay_does_not_re_execute_tools() -> None:
    record = Recorder()
    _, live_calls = _wire_and_run(Guard(_POLICY, recorder=record))
    assert live_calls == ["fetch", "store_note"]

    _, replay_calls = _wire_and_run(
        Guard(_POLICY, replayer=Replayer(record.recording()))
    )
    assert replay_calls == []  # INV-8: no tool body runs on replay


def test_replay_cassette_miss_fails_closed() -> None:
    record = Recorder()
    _wire_and_run(Guard(_POLICY, recorder=record))

    guard = Guard(_POLICY, replayer=Replayer(record.recording()))

    @guard.tool(name="fetch", cls=ToolClass.READ_ONLY, emits=_WEB)
    def fetch(url: str) -> str:
        return "<web>"

    # A different request (different URL => different call id) is not in the cassette.
    with pytest.raises(ReplayError):
        fetch(guard.source("http://DIFFERENT"))
