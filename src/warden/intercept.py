"""Mode 2 in-process interception: the shim that wires existing tools through the Guard.

See arch section 9. Mode 2 is the ergonomic, zero-network surface: you decorate the
callables an agent already has with ``@guard.tool`` and label external inputs with
``guard.source``. Everything else -- mediation, propagation, fail-closed denial --
is inherited from the Guard core, which this module only orchestrates; it adds no
new trust assumptions.

Labels enter at exactly two boundaries (finding F5):

  * ``guard.source(value, ...)`` -- an external input (user text, a DB row) is born
    with an explicitly declared label;
  * ``@guard.tool(..., emits=...)`` -- a tool that INTRODUCES taint (a web fetch, a
    secret read) stamps its result with an intrinsic source label.

Between those boundaries, a result's label is the WHOLE_CONTEXT join of its argument
labels and the tool's source label, so taint cannot be laundered out of a derivation
(INV-3 by construction). Every wrapped call is mediated BEFORE the underlying
function runs (complete mediation, arch section 8); a denial raises
``WardenPolicyViolation`` and the side effect never happens.

Residual (documented, not hidden): a Handle nested inside a composite argument (a
list, dict, or dataclass passed as one parameter) is not unwrapped, so its label is
not traced -- pass handles as direct arguments. Top-level ``*args``/``**kwargs`` ARE
folded in. Soundness of WHOLE_CONTEXT holds over directly-passed handles; structural
laundering through opaque containers is the M3 dual-plane / static-analysis frontier.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, overload

from warden.core import Node, NodeId, NodeKind
from warden.harness import Recorder
from warden.labels import Confidentiality, Label, SourceId, Taint, join_all
from warden.monitor import Monitor
from warden.policy import Policy, ToolClass, compile_policy
from warden.propagate import WHOLE_CONTEXT, PropagationStrategy

__all__ = ["Guard", "Handle"]


@dataclass(frozen=True, slots=True)
class Handle:
    """An opaque, labeled reference to a value flowing through the Guard.

    Identity is the content node id (Layer A), so two handles to byte-identical
    content under the same label compare and hash equal. The underlying value
    travels with the handle so wrapped tools can compute on it, but it is NEVER
    part of identity or the label plane; read it back only via ``Guard.value``.
    """

    id: NodeId
    label: Label = field(compare=False)
    _value: Any = field(compare=False, repr=False)


def _payload(value: Any) -> str:
    """Project an arbitrary value to a canonical-safe provenance payload.

    A deterministic ``repr`` stands in for full content capture, which is the
    Harness's job (M2). It gives handles a stable content id without forcing every
    tool's return type to be CBOR-encodable.
    """
    return repr(value)


def _classify(value: Any) -> tuple[Label, Any, NodeId | None]:
    """Split a call argument into (label, underlying value, provenance parent)."""
    if isinstance(value, Handle):
        return value.label, value._value, value.id
    return Label.bottom(), value, None


class Guard:
    """A policy plus the in-process registry of tools wrapped through it.

    One Guard owns one Monitor and one propagation strategy. Decorating a callable
    registers its tool class with the Monitor and returns a wrapper that mediates
    every call and returns a labeled Handle.
    """

    __slots__ = ("_monitor", "_recorder", "_strategy")

    def __init__(
        self,
        policy: Policy | str,
        *,
        strategy: PropagationStrategy = WHOLE_CONTEXT,
        recorder: Recorder | None = None,
    ) -> None:
        compiled = compile_policy(policy) if isinstance(policy, str) else policy
        self._monitor = Monitor(compiled)
        self._strategy = strategy
        self._recorder = recorder

    def source(
        self,
        value: Any,
        *,
        integrity: Taint = Taint.TRUSTED,
        confidentiality: Confidentiality = Confidentiality.PUBLIC,
        provenance: Iterable[SourceId] = (),
    ) -> Handle:
        """Mint a labeled handle for an external input (the first label boundary)."""
        label = Label(integrity, confidentiality, frozenset(provenance))
        node = Node(NodeKind.USER_INPUT, (), _payload(value))
        if self._recorder is not None:
            self._recorder.record_source(node)
        return Handle(node.id, label, value)

    def value(self, handle: Handle) -> Any:
        """Reveal the underlying value of a handle (trusted egress read)."""
        return handle._value

    @overload
    def tool(self, fn: Callable[..., Any], /) -> Callable[..., Handle]: ...

    @overload
    def tool(
        self,
        *,
        name: str | None = None,
        cls: ToolClass = ToolClass.CONSEQUENTIAL,
        emits: Label | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Handle]]: ...

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        cls: ToolClass = ToolClass.CONSEQUENTIAL,
        emits: Label | None = None,
    ) -> Any:
        """Wrap a callable so its calls are mediated and its results are labeled.

        ``cls`` declares the tool class (consequential default-deny, read-only
        default-allow). ``emits`` is the tool's intrinsic source label: set it for
        tools that introduce taint (a web fetch is untrusted; a secret read is
        confidential). Usable bare (``@guard.tool``) or parameterized.
        """
        source_label = emits or Label.bottom()

        def decorate(func: Callable[..., Any]) -> Callable[..., Handle]:
            action = name or func.__name__
            self._monitor.register(action, cls)
            signature = inspect.signature(func)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Handle:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                arg_labels: dict[str, Label] = {}
                parent_ids: list[NodeId] = []
                parent_labels: list[Label] = []
                for pname, param in signature.parameters.items():
                    raw = bound.arguments[pname]
                    if param.kind is inspect.Parameter.VAR_POSITIONAL:
                        items = [_classify(item) for item in raw]
                        bound.arguments[pname] = tuple(value for _, value, _ in items)
                    elif param.kind is inspect.Parameter.VAR_KEYWORD:
                        items = [_classify(item) for item in raw.values()]
                        bound.arguments[pname] = {
                            key: value
                            for key, (_, value, _) in zip(raw, items, strict=True)
                        }
                    else:
                        label, value, parent = _classify(raw)
                        items = [(label, value, parent)]
                        bound.arguments[pname] = value
                    arg_labels[pname] = join_all(label for label, _, _ in items)
                    parent_labels.extend(label for label, _, _ in items)
                    parent_ids.extend(pid for _, _, pid in items if pid is not None)

                # Complete mediation: decide BEFORE the side effect; deny raises.
                self._monitor.mediate(action, arg_labels)

                # The call node is the content-addressed identity of the REQUEST
                # (tool + bound arguments) -- the cassette key a replay looks up.
                call_node = Node(
                    NodeKind.TOOL_CALL,
                    tuple(parent_ids),
                    {"tool": action, "args": _payload(dict(bound.arguments))},
                )
                result = func(*bound.args, **bound.kwargs)
                result_node = Node(NodeKind.TOOL_RESULT, (call_node.id,), _payload(result))
                label = self._strategy.node_label(result_node, parent_labels, source_label)
                if self._recorder is not None:
                    self._recorder.record_boundary(call_node, result_node)
                return Handle(result_node.id, label, result)

            return wrapper

        return decorate if fn is None else decorate(fn)
