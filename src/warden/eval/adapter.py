"""Warden's reference monitor as an AgentDojo pipeline element (arch sections 8, 12).

AgentDojo drives a real LLM through a composable pipeline; the one place it actually
invokes a tool is ``ToolsExecutor``, which calls ``runtime.run_function`` for each
tool call the model requested. This module supplies ``WardenToolsExecutor``, a
drop-in replacement that mediates every call through Warden's ``Monitor`` BEFORE the
side effect runs. It never patches AgentDojo: the pipeline only duck-calls
``.query(...)``, so a plain object of the right shape slots straight in.

The hard part is soundness against a real model (finding F5). Warden's WHOLE_CONTEXT
strategy propagates taint over the DAG of *handles*; but an LLM reads an untrusted
value and RE-TYPES it as a fresh literal argument, which breaks the handle chain. So
this adapter models taint at the CONVERSATION level instead: it accumulates the label
of every tool result the model has been shown, and labels each subsequent call's
arguments with that running join. That is WHOLE_CONTEXT realized over the transcript
-- sound (no read value escapes the join) and deliberately conservative. Its
false-positive (label-creep) cost is exactly the quantity the eval measures, and the
quantity the low-creep per-handle alternative (M3 dual-plane) exists to reduce.

The context is threaded through the pipeline's ``extra_args`` rather than held on the
element, so the executor is stateless and reentrant across tasks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentdojo.agent_pipeline.tool_execution import (
    is_string_list,
    literal_eval,
    tool_result_to_str,
)
from agentdojo.types import ChatToolResultMessage, text_content_block_from_string

from warden.labels import Label
from warden.monitor import Monitor, WardenPolicyViolation
from warden.policy import Policy, ToolClass, compile_policy

__all__ = [
    "ToolSpec",
    "WardenToolsExecutor",
    "mediating_executor",
]

# Key under which the running conversation-context label rides the pipeline's
# extra_args dict (namespaced to avoid colliding with AgentDojo's own keys).
_CONTEXT_KEY = "warden.context"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """How Warden views one AgentDojo tool.

    ``tool_class`` decides whether the call is a mediated sink (CONSEQUENTIAL) or a
    default-allowed read. ``emits`` is the intrinsic label the tool introduces into
    the conversation when it returns -- e.g. UNTRUSTED for a tool that reads
    attacker-reachable data, or SECRET for one that reads a vault. ``None`` means the
    tool introduces no new taint of its own.
    """

    tool_class: ToolClass
    emits: Label | None = None


def _tool_message(call: Any, text: str, error: str | None) -> Any:
    """Build the AgentDojo tool-result message the model will see next."""
    return ChatToolResultMessage(
        role="tool",
        content=[text_content_block_from_string(text)],
        tool_call_id=call.id,
        tool_call=call,
        error=error,
    )


class WardenToolsExecutor:
    """A mediating drop-in for AgentDojo's ``ToolsExecutor``.

    For each tool call in the latest assistant message it resolves the call's
    arguments to labels (the conversation context for a sink), asks the ``Monitor``
    to decide, and only then runs the tool via ``runtime.run_function``. A denial is
    surfaced to the model as an error tool-result -- the side effect never runs
    (complete mediation, fail-closed) -- exactly mirroring AgentDojo's own error
    convention so the agent loop continues unperturbed.
    """

    __slots__ = ("_format", "_monitor", "_specs")

    def __init__(
        self,
        monitor: Monitor,
        specs: Mapping[str, ToolSpec],
        *,
        output_formatter: Any = tool_result_to_str,
    ) -> None:
        self._monitor = monitor
        self._specs = dict(specs)
        self._format = output_formatter

    def query(
        self,
        query: str,
        runtime: Any,
        env: Any = None,
        messages: Any = (),
        extra_args: Any = None,
    ) -> tuple[str, Any, Any, list[Any], dict[str, Any]]:
        extra_args = dict(extra_args or {})
        if not messages:
            return query, runtime, env, list(messages), extra_args
        last = messages[-1]
        if last.get("role") != "assistant" or not last.get("tool_calls"):
            return query, runtime, env, list(messages), extra_args

        context: Label = extra_args.get(_CONTEXT_KEY, Label.bottom())
        results: list[Any] = []
        for call in last["tool_calls"]:
            # Coerce stringified lists exactly as AgentDojo's executor does.
            for name, value in call.args.items():
                if isinstance(value, str) and is_string_list(value):
                    call.args[name] = literal_eval(value)

            # Every argument the model produced is tainted by all it has read.
            arg_labels = {name: context for name in call.args}
            try:
                self._monitor.mediate(call.function, arg_labels)
            except WardenPolicyViolation as denial:
                results.append(_tool_message(call, "", str(denial)))
                continue

            result, error = runtime.run_function(env, call.function, call.args)
            results.append(_tool_message(call, self._format(result), error))

            spec = self._specs.get(call.function)
            if spec is not None and spec.emits is not None:
                context = context.join(spec.emits)

        extra_args[_CONTEXT_KEY] = context
        return query, runtime, env, [*messages, *results], extra_args


def mediating_executor(
    policy: Policy | str,
    specs: Mapping[str, ToolSpec],
    *,
    output_formatter: Any = tool_result_to_str,
) -> WardenToolsExecutor:
    """Build a ``WardenToolsExecutor`` for ``policy`` over the given tool specs.

    Tool classes are registered with the ``Monitor`` up front so unregistered tools
    fail closed to CONSEQUENTIAL (default-deny).
    """
    compiled = compile_policy(policy) if isinstance(policy, str) else policy
    monitor = Monitor(compiled)
    for name, spec in specs.items():
        monitor.register(name, spec.tool_class)
    return WardenToolsExecutor(monitor, specs, output_formatter=output_formatter)
