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

Residual (documented, not hidden): the per-thread session bindings live in memory,
not in a LangGraph checkpoint. A cross-process resume therefore loses them, and a
token minted before the resume degrades to a bottom literal carrying its own text.
That is sound -- it can only LOWER trust, never forge it -- but it costs precision
on resume; checkpoint-backed bindings are the follow-up. Separately, the model
still needs each tool's *schema* to call it (via ``.bind_tools``); for this first
cut you define the LangChain tools for the schema and hand Warden the decorated
callables keyed by name. Deriving both from one definition is a future helper.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import ToolMessage

from warden import Guard, Handle, Label, Session, WardenPolicyViolation

__all__ = ["WardenToolNode"]

# The thread key a session is filed under when the runnable config carries none.
_DEFAULT_THREAD = "__warden_default__"


class WardenToolNode:
    """A mediating, handle-masking drop-in for LangGraph's prebuilt ``ToolNode``.

    ``tools`` maps each tool name to its already ``@guard.tool``-decorated callable
    (the documented product API -- no new spec type). The key is matched against
    ``tool_call["name"]``. Construct it once and use it as the graph's tool node.
    """

    __slots__ = ("_guard", "_sessions", "_tools")

    def __init__(self, guard: Guard, tools: Mapping[str, Callable[..., Handle]]) -> None:
        self._guard = guard
        self._tools = dict(tools)
        self._sessions: dict[str, Session] = {}

    def _session(self, config: Mapping[str, Any] | None) -> Session:
        """Return this thread's session, opening one on first use.

        Bindings must outlive a single node call so a token minted in one ReAct
        iteration resolves in the next, so the session is keyed by ``thread_id``
        and reused, not reopened per call.
        """
        configurable = (config or {}).get("configurable") or {}
        thread_id = configurable.get("thread_id", _DEFAULT_THREAD)
        session = self._sessions.get(thread_id)
        if session is None:
            session = self._guard.session()
            self._sessions[thread_id] = session
        return session

    def __call__(
        self,
        state: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, list[Any]]:
        last = state["messages"][-1]
        session = self._session(config)
        out: list[Any] = []
        for call in last.tool_calls:
            # Resolve each argument through the session: a token unmasks to its exact
            # labeled handle; anything the model typed itself becomes a bottom literal.
            handles = {name: session.unmask(value) for name, value in call["args"].items()}
            try:
                # Mediation runs INSIDE the wrapper and raises before the side effect.
                handle = self._tools[call["name"]](**handles)
            except WardenPolicyViolation as denial:
                out.append(
                    ToolMessage(
                        content=str(denial), tool_call_id=call["id"], status="error"
                    )
                )
                continue
            # Mask any labeled result so the model never sees its bytes; a benign,
            # bottom-labeled result is shown raw to preserve utility (no creep).
            if handle.label != Label.bottom():
                content = session.mask(handle)
            else:
                content = str(self._guard.value(handle))
            out.append(ToolMessage(content=content, tool_call_id=call["id"]))
        return {"messages": out}
