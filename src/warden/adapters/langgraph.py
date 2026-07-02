"""LangGraph adapter: a drop-in ``ToolNode`` that mediates every tool call.

LangGraph's prebuilt ``ToolNode`` is the one component that actually runs tools:
given the graph state, it reads the last ``AIMessage.tool_calls``, executes each
tool, and returns ``{"messages": [ToolMessage, ...]}``. That is exactly where a
reference monitor belongs, so ``WardenToolNode`` is a drop-in for it: swap one
node, change one line, and every tool call in the graph is mediated (fail-closed)
and every labeled result is masked behind an opaque token before the model sees it.

The node adds no new trust machinery -- it reuses the in-process Guard verbatim:

  * each tool is an ``@guard.tool``-decorated callable, so mediation happens inside
    the wrapper and a denial raises ``WardenPolicyViolation`` BEFORE the side effect
    (complete mediation); the wrapper returns a labeled ``Handle``;
  * a per-thread ``Session`` masks each labeled result to the model as a token and
    resolves token-shaped arguments back to their exact handle. Anything the model
    typed itself resolves to a bottom (trusted/public) literal, because a model that
    only ever saw tokens never received labeled bytes to launder into a literal --
    the F5 defense, recovered here for a real graph runtime.

Tokens minted in turn N must resolve in turn N+1, so the session persists across
ReAct iterations within a thread; sessions are keyed by ``thread_id`` from the
runnable config (one default session when absent).

Handling a denial (``on_denial``)
---------------------------------
``on_denial="error"`` (the default) surfaces the denial to the model as an error
``ToolMessage`` and lets the graph run on -- no checkpointer needed, and the
behavior every other graph already relies on.

``on_denial="interrupt"`` turns fail-closed into *fail-closed with an audited
escalation*. A denied call calls LangGraph's ``interrupt()`` with the explainable
provenance path (never the raw argument bytes -- the payload cannot itself leak a
labeled value), pausing the graph for a human. Resume with
``Command(resume=...)``: an approval declassifies the call's arguments to bottom
via ``Guard.declassify`` -- the sanctioned INV-3 downgrade, recorded as
DECLASSIFICATION provenance -- and re-runs the call *through the monitor*, so the
approval LOWERS labels rather than bypassing mediation; anything else proceeds as
the rejected error. Approval is ``resume is True``, ``"approve"``, or a mapping
with ``{"approve": True}`` / ``{"decision": "approve"}``. This mode requires the
graph be compiled with a checkpointer (LangGraph's precondition for ``interrupt``).

Re-execution safety. ``interrupt()`` re-runs the whole node from the top on every
resume and matches interrupts by position, so this node memoizes each tool's
result by ``tool_call_id`` (in-process, across the pause): an already-run body --
allowed or approved -- is never re-executed (no double side effect), while a call
that consumed an interrupt still re-issues it on re-execution to hold its counter
slot. Declassification is coarse by design: an approval clears *every* argument of
that one call (the reviewer saw the full provenance and cleared the action); it
does not relabel upstream handles or affect other calls.

Surviving a process restart
---------------------------
Compile the graph with a persistent ``store=`` (any LangGraph ``BaseStore``) and
the node persists each thread's token->handle bindings under the namespace
``("warden", <thread_id>, "bindings")``, written through at mint time. Store puts
are durable immediately -- unlike state-channel writes, they are not part of the
checkpoint transaction -- so bindings minted before an ``interrupt()`` pauses the
super-step survive it. A fresh process resuming the thread rebuilds the session by
re-masking the persisted handles in mint order: tokens are minted deterministically
by count, so the exact names reproduce through the public API, and a mismatch fails
closed. Labels round-trip exactly (persistence never launders trust); values ride
the store's serializer, so cross-process durability assumes JSON-representable tool
values. The store is part of the trusted base exactly like the checkpointer: an
attacker who can rewrite it can already rewrite the message history itself.

Residual (documented, not hidden): without a store the bindings are process-local
as before -- a token minted before a cross-process resume degrades to a bottom
literal (sound: it can only LOWER trust, never forge it). The re-execution memo is
in-memory either way, so an in-flight escalation cannot yet be resumed in a fresh
process. Separately, the model still needs each tool's *schema* to call it (via
``.bind_tools``); for this first cut you define the LangChain tools for the schema
and hand Warden the decorated callables keyed by name.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from warden import (
    Confidentiality,
    Guard,
    Handle,
    Label,
    Session,
    Taint,
    WardenPolicyViolation,
)
from warden.core import NodeId

# Intra-package: the session's token wire format ("<warden-handle:N>") is minted
# by intercept.Session; the adapter parses N back out to persist mint order.
from warden.intercept import _HANDLE_PREFIX

__all__ = ["WardenToolNode"]

# The thread key a session is filed under when the runnable config carries none.
_DEFAULT_THREAD = "__warden_default__"

# How a denial is surfaced. "error" -> error ToolMessage (default, no checkpointer
# needed); "interrupt" -> pause for human review via LangGraph's interrupt().
_ON_DENIAL: tuple[str, ...] = ("error", "interrupt")

# Per-call memo entry: (interrupt payload or None, the resolved ToolMessage). A
# non-None payload marks a call that consumed an interrupt and so must re-issue it
# on node re-execution to keep LangGraph's positional interrupt counter aligned.
type _MemoEntry = tuple[dict[str, Any] | None, Any]

# Page size for exhausting a store namespace (BaseStore.search defaults to a
# LIMIT of 10 and silently truncates -- a naive search would drop bindings).
_PAGE = 64


def _items(store: BaseStore, namespace: tuple[str, ...]) -> list[Any]:
    """Every item under ``namespace``, paginating past the search default limit."""
    items: list[Any] = []
    offset = 0
    while True:
        page = store.search(namespace, limit=_PAGE, offset=offset)
        items.extend(page)
        if len(page) < _PAGE:
            return items
        offset += _PAGE


def _encode_handle(handle: Handle, value: Any) -> dict[str, Any]:
    """Project a handle to a store-safe record: exact label, id hex, raw value.

    The label round-trips EXACTLY (enum values plus the provenance set), so a
    restored token carries precisely the trust it had -- persistence must never
    launder a label. The value rides through the store's own serializer; the
    production backends serialize JSON, so cross-process durability assumes
    JSON-representable tool values (strings, in practice).
    """
    return {
        "id": handle.id.hex(),
        "integrity": handle.label.integrity.value,
        "confidentiality": handle.label.confidentiality.value,
        "provenance": sorted(handle.label.provenance),
        "value": value,
    }


def _decode_handle(encoded: Mapping[str, Any]) -> Handle:
    label = Label(
        Taint(encoded["integrity"]),
        Confidentiality(encoded["confidentiality"]),
        frozenset(encoded["provenance"]),
    )
    return Handle(NodeId(bytes.fromhex(encoded["id"])), label, encoded["value"])


def _is_approval(decision: Any) -> bool:
    """Whether a resume value approves the escalation (see ``on_denial``)."""
    if decision is True or decision == "approve":
        return True
    if isinstance(decision, Mapping):
        return decision.get("approve") is True or decision.get("decision") == "approve"
    return False


class WardenToolNode:
    """A mediating, handle-masking drop-in for LangGraph's prebuilt ``ToolNode``.

    ``tools`` maps each tool name to its already ``@guard.tool``-decorated callable
    (the documented product API -- no new spec type). The key is matched against
    ``tool_call["name"]``. ``on_denial`` selects how a policy denial is surfaced:
    ``"error"`` (default) or ``"interrupt"`` (human review; see the module docstring).
    Construct it once and use it as the graph's tool node.
    """

    __slots__ = ("_guard", "_memo", "_on_denial", "_sessions", "_tools")

    def __init__(
        self,
        guard: Guard,
        tools: Mapping[str, Callable[..., Handle]],
        *,
        on_denial: Literal["error", "interrupt"] = "error",
    ) -> None:
        if on_denial not in _ON_DENIAL:
            raise ValueError(f"on_denial must be one of {_ON_DENIAL}, got {on_denial!r}")
        self._guard = guard
        self._tools = dict(tools)
        self._on_denial = on_denial
        self._sessions: dict[str, Session] = {}
        self._memo: dict[str, dict[str, _MemoEntry]] = {}

    def _thread_id(self, config: RunnableConfig | None) -> str:
        configurable = (config or {}).get("configurable") or {}
        thread_id: str = configurable.get("thread_id", _DEFAULT_THREAD)
        return thread_id

    def _namespace(self, thread_id: str, plane: str) -> tuple[str, str, str]:
        return ("warden", thread_id, plane)

    def _session(self, thread_id: str, store: BaseStore | None) -> Session:
        """Return this thread's session, opening (and hydrating) one on first use.

        Bindings must outlive a single node call so a token minted in one ReAct
        iteration resolves in the next, so the session is keyed by ``thread_id``
        and reused, not reopened per call. In a fresh process the session is
        rebuilt from the store before first use; thereafter every mask writes
        through, so the in-process session stays authoritative.
        """
        session = self._sessions.get(thread_id)
        if session is None:
            session = self._guard.session()
            if store is not None:
                self._restore_bindings(session, thread_id, store)
            self._sessions[thread_id] = session
        return session

    def _restore_bindings(
        self, session: Session, thread_id: str, store: BaseStore
    ) -> None:
        """Replay persisted bindings into a fresh session, in mint order.

        Tokens are minted deterministically by count, so re-masking the restored
        handles in their original order reproduces the exact token names through
        the public ``mask`` API alone. A name mismatch means the store disagrees
        with the session discipline (a lost or forged entry); fail closed rather
        than resolve tokens to the wrong labels.
        """
        namespace = self._namespace(thread_id, "bindings")
        for item in sorted(_items(store, namespace), key=lambda it: it.value["index"]):
            token = session.mask(_decode_handle(item.value))
            if token != item.key:
                raise RuntimeError(
                    f"warden binding store for thread {thread_id!r} is inconsistent: "
                    f"expected token {item.key!r}, minted {token!r}"
                )

    def _result_message(
        self,
        handle: Handle,
        session: Session,
        call_id: str,
        thread_id: str,
        store: BaseStore | None,
    ) -> Any:
        """Render a successful result: mask a labeled handle, show a bottom one raw."""
        if handle.label != Label.bottom():
            content = session.mask(handle)
            if store is not None:
                # Write the binding through immediately: store puts are durable
                # even if a later interrupt() pauses this very super-step.
                index = int(content[len(_HANDLE_PREFIX) : -1])
                store.put(
                    self._namespace(thread_id, "bindings"),
                    content,
                    {"index": index, **_encode_handle(handle, self._guard.value(handle))},
                )
        else:
            content = str(self._guard.value(handle))
        return ToolMessage(content=content, tool_call_id=call_id)

    def _denial_payload(
        self, call: Mapping[str, Any], denial: WardenPolicyViolation
    ) -> dict[str, Any]:
        """The human-review payload for a denied call.

        Deliberately provenance-only: it carries the action and the explainable
        path (INV-6), never the raw argument values, so the approval prompt itself
        cannot leak a labeled value.
        """
        return {
            "action": denial.action,
            "tool_call_id": call["id"],
            "reason": str(denial),
            "provenance": list(denial.path),
        }

    def _resolve(
        self,
        call: Mapping[str, Any],
        session: Session,
        memo: dict[str, _MemoEntry],
        thread_id: str,
        store: BaseStore | None,
    ) -> Any:
        """Resolve one tool call to a ``ToolMessage`` (may pause via ``interrupt``)."""
        call_id = call["id"]
        cached = memo.get(call_id)
        if cached is not None:
            payload, message = cached
            if payload is not None:
                # Re-issue the interrupt so LangGraph's positional counter stays
                # aligned on re-execution; the stored resume is returned, no pause.
                interrupt(payload)
            return message

        handles = {name: session.unmask(value) for name, value in call["args"].items()}
        try:
            # Mediation runs INSIDE the wrapper and raises before the side effect.
            handle = self._tools[call["name"]](**handles)
        except WardenPolicyViolation as denial:
            if self._on_denial == "error":
                message = ToolMessage(
                    content=str(denial), tool_call_id=call_id, status="error"
                )
                memo[call_id] = (None, message)
                return message
            payload = self._denial_payload(call, denial)
            decision = interrupt(payload)
            if _is_approval(decision):
                # Authority-gated downgrade: lower each argument to bottom (recorded
                # as DECLASSIFICATION provenance) and re-run THROUGH the monitor, so
                # the approval lowers labels rather than bypassing mediation.
                cleared = {
                    name: self._guard.declassify(handle, to=Label.bottom())
                    for name, handle in handles.items()
                }
                approved = self._tools[call["name"]](**cleared)
                message = self._result_message(
                    approved, session, call_id, thread_id, store
                )
            else:
                message = ToolMessage(
                    content=f"{denial} (escalation rejected by reviewer)",
                    tool_call_id=call_id,
                    status="error",
                )
            memo[call_id] = (payload, message)
            return message

        message = self._result_message(handle, session, call_id, thread_id, store)
        memo[call_id] = (None, message)
        return message

    def __call__(
        self,
        state: Mapping[str, Any],
        # These annotations are LOAD-BEARING and matched TEXTUALLY: under postponed
        # evaluation LangGraph injects the runnable config and the compiled store
        # only for the literal spellings "RunnableConfig"/"Optional[RunnableConfig]"
        # and "BaseStore"/"Optional[BaseStore]". Any other spelling (including
        # "RunnableConfig | None") silently receives None, which would collapse
        # every thread onto the default session -- cross-thread token leakage on a
        # shared node -- and silently disable binding persistence.
        config: Optional[RunnableConfig] = None,  # noqa: UP045
        store: Optional[BaseStore] = None,  # noqa: UP045
    ) -> dict[str, list[Any]]:
        last = state["messages"][-1]
        thread_id = self._thread_id(config)
        session = self._session(thread_id, store)
        memo = self._memo.setdefault(thread_id, {})
        # A pending interrupt raises out of _resolve; the memo is retained so the
        # already-run calls are not re-executed on resume. Reaching the end means
        # the super-step completed, so the memo is dropped.
        messages = [
            self._resolve(call, session, memo, thread_id, store)
            for call in last.tool_calls
        ]
        self._memo.pop(thread_id, None)
        return {"messages": messages}
