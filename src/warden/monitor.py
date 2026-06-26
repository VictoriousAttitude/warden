"""The reference monitor: the one gate every consequential tool call passes through.

See arch section 8. Before a side effect runs, the monitor resolves the labels of
the call's arguments, evaluates the compiled policy, and either allows the call or
raises WardenPolicyViolation carrying a provenance path (INV-6). It implements
Saltzer & Schroeder complete mediation with fail-safe defaults.

Complete mediation is SCOPED, not absolute (finding F4): in a Python library we
cannot stop agent code from calling a tool's raw callable directly. The guarantee
is "complete mediation over registered sinks invoked through this monitor," to be
backed by a static bypass-lint in a later milestone. Here we provide the runtime
gate and the fail-closed defaults (INV-5): an unregistered tool is treated as
consequential (default-deny), and a policy that cannot be evaluated denies.
"""

from __future__ import annotations

from collections.abc import Mapping

from warden.labels import Confidentiality, Label, Taint
from warden.policy import Decision, Policy, PolicyError, ToolClass, decide
from warden.propagate import UnlabeledError, ValueEnv

__all__ = ["Monitor", "WardenPolicyViolation"]


class WardenPolicyViolation(Exception):  # noqa: N818  # spec-mandated public name (arch 8.1)
    """Raised when the monitor denies a call. Always carries a provenance path."""

    def __init__(self, action: str, decision: Decision, path: tuple[str, ...]) -> None:
        self.action = action
        self.decision = decision
        self.path = path
        super().__init__(self._render())

    def _render(self) -> str:
        trail = " -> ".join(self.path)
        return f"Warden denied {self.action!r}: {trail}"


def _build_path(action: str, args: Mapping[str, Label], decision: Decision) -> tuple[str, ...]:
    """Render the contributing labels and the deciding reason (INV-6: non-empty)."""
    steps: list[str] = []
    for name, label in sorted(args.items()):
        markers: list[str] = []
        if label.integrity is Taint.UNTRUSTED:
            markers.append("UNTRUSTED")
        if label.confidentiality is not Confidentiality.PUBLIC:
            markers.append(label.confidentiality.name)
        if label.provenance:
            markers.append("from " + ", ".join(sorted(label.provenance)))
        if markers:
            steps.append(f"`{name}` ({'; '.join(markers)})")
    steps.append(decision.reason)
    return tuple(steps)


class Monitor:
    """A policy plus a registry of which tools are consequential sinks."""

    __slots__ = ("_policy", "_tools")

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._tools: dict[str, ToolClass] = {}

    def register(self, name: str, tool_class: ToolClass) -> None:
        """Declare a tool's class. Unregistered tools are treated as consequential."""
        self._tools[name] = tool_class

    def mediate(self, action: str, args: Mapping[str, Label]) -> Decision:
        """Decide a call; raise WardenPolicyViolation on denial (fail-closed).

        An unregistered tool defaults to CONSEQUENTIAL (most restrictive). A policy
        that references a value not present in ``args`` cannot establish safety, so
        it denies rather than letting the call through.
        """
        tool_class = self._tools.get(action, ToolClass.CONSEQUENTIAL)
        try:
            decision = decide(self._policy, action, args, tool_class)
        except PolicyError as exc:
            decision = Decision(False, None, f"fail-closed: {exc}")
        if not decision.allowed:
            raise WardenPolicyViolation(action, decision, _build_path(action, args, decision))
        return decision

    def mediate_handles(
        self, action: str, args: Mapping[str, str], env: ValueEnv
    ) -> Decision:
        """Resolve handle-referenced arguments to their labels, then mediate.

        An argument whose handle is unknown to the value environment is unlabeled at
        a sink, which fail-closes to a denial (INV-5).
        """
        resolved: dict[str, Label] = {}
        for name, handle in args.items():
            try:
                resolved[name] = env.label_of(handle)
            except UnlabeledError as exc:
                decision = Decision(
                    False, None, f"fail-closed: unlabeled argument {name!r}"
                )
                raise WardenPolicyViolation(
                    action, decision, (f"`{name}`", decision.reason)
                ) from exc
        return self.mediate(action, resolved)
