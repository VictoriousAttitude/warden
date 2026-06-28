"""Record-once, replay-hermetically for the model layer of an AgentDojo run.

The only non-deterministic, network-bound, paid part of a real AgentDojo run is the
LLM. The tools are pure in-memory mutations of the task environment, so given the same
sequence of model requests they reproduce the same results deterministically. Hermetic
CI therefore replays the MODEL's turns -- the tool *requests* -- and lets the tools
re-run; it never contacts a provider.

This is a distinct layer from the Harness, which replays tool *results* over the
provenance DAG (INV-8). Here recording is simply running the live pipeline once and
keeping its message list; ``ReplayModel`` is the pipeline element that re-emits the
recorded assistant turns. The cursor rides ``extra_args`` so one instance serves both
the pre-loop and in-loop model positions, and running past the recording fails closed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentdojo.functions_runtime import FunctionCall

__all__ = [
    "ReplayModel",
    "TranscriptExhaustedError",
    "load_transcript",
    "save_transcript",
]

# Cursor over the recorded turns, threaded through the pipeline's extra_args so the
# replay element stays stateless and reentrant across its two pipeline positions.
_CURSOR_KEY = "warden.replay_cursor"


class TranscriptExhaustedError(Exception):
    """A replayed run asked for more model turns than were recorded (fail-closed)."""


def _dump_turn(message: Any) -> dict[str, Any]:
    calls = message["tool_calls"]
    return {
        "content": message["content"],
        "tool_calls": [call.model_dump() for call in calls] if calls else None,
    }


def _load_turn(raw: Mapping[str, Any]) -> dict[str, Any]:
    calls = raw["tool_calls"]
    return {
        "role": "assistant",
        "content": raw["content"],
        "tool_calls": [FunctionCall.model_validate(c) for c in calls] if calls else None,
    }


def save_transcript(messages: Sequence[Any], path: Path | str) -> None:
    """Serialize the assistant turns of a finished run to a JSON fixture.

    Only assistant messages are kept, in order: they are the model's outputs, the sole
    non-deterministic part of the run. Tool results are deliberately NOT stored -- they
    are reproduced when the tools re-run under replay.
    """
    turns = [_dump_turn(m) for m in messages if m.get("role") == "assistant"]
    Path(path).write_text(json.dumps(turns, indent=2), encoding="utf-8")


def load_transcript(path: Path | str) -> list[dict[str, Any]]:
    """Load a transcript fixture back into replayable assistant turns."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_load_turn(turn) for turn in raw]


class ReplayModel:
    """A model-shaped pipeline element that re-emits recorded turns, never a live call.

    Each ``query`` appends the next recorded assistant message instead of contacting a
    provider, so a recorded run replays with no network and no API key. The cursor
    lives in ``extra_args``; asking for a turn past the recording fails closed.
    """

    __slots__ = ("_turns",)

    def __init__(self, turns: Sequence[Any]) -> None:
        self._turns = list(turns)

    def query(
        self,
        query: str,
        runtime: Any,
        env: Any = None,
        messages: Any = (),
        extra_args: Any = None,
    ) -> tuple[str, Any, Any, list[Any], dict[str, Any]]:
        extra_args = dict(extra_args or {})
        cursor = extra_args.get(_CURSOR_KEY, 0)
        if cursor >= len(self._turns):
            raise TranscriptExhaustedError(
                f"replay needs turn {cursor} but only {len(self._turns)} were recorded"
            )
        extra_args[_CURSOR_KEY] = cursor + 1
        return query, runtime, env, [*messages, self._turns[cursor]], extra_args
