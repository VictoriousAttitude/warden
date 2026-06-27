"""Static bypass-lint: AST verification of mediation completeness (INV-4).

Implements mechanism (2) of the three that together make up complete mediation
(arch section 8.2; resolves finding F4). Python has no enforced encapsulation, so
"no bypass by construction" is unattainable in a library: agent code can reach a
tool's raw callable around the monitor. This lint makes that residual VISIBLE by
flagging, statically, every way a registered sink's unmediated callable is reached
on a source corpus. It is not a vulnerability scanner for target programs; it is
how we PROVE -- over the analyzed corpus, under the threat-model assumption that
tools are wired through Warden (arch section 8.2(3)) -- that no consequential side
effect escapes the gate.

Two bypass surfaces survive the ``@guard.tool`` design, and the lint flags both:

  * W1 -- the ``functools.wraps`` backdoor. The decorator rebinds a tool's name to
    the mediated wrapper, but ``functools.wraps`` re-exposes the raw callable as
    ``wrapper.__wrapped__``. Any ``.__wrapped__`` access reaches it unmediated.
  * W2 -- the functional registration form. ``wrapped = guard.tool(raw)`` leaves
    the original ``def raw`` bound and directly callable, so a later ``raw(...)``
    bypasses the wrapper. Decorator syntax shadows the raw name and avoids this.

Residual (the threat-model assumption, arch section 8.2(3)): dynamic dispatch
(getattr, dict lookup), cross-module aliasing, and reflection are out of scope --
this is single-module, name-based analysis. The guarantee is "complete mediation
over registered sinks invoked through the boundary," verified here, not absolute.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Finding", "lint_file", "lint_paths", "lint_source"]

_WRAPPED = "__wrapped__"
_GUARD_CTOR = "Guard"
_TOOL_METHOD = "tool"


@dataclass(frozen=True, slots=True)
class Finding:
    """One mediation-bypass site located by the lint."""

    code: str  # "W1" (__wrapped__ backdoor) | "W2" (raw functional-form sink)
    message: str
    filename: str
    lineno: int
    col: int

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}:{self.col}: {self.code} {self.message}"


def _guard_names(tree: ast.Module) -> set[str]:
    """Names bound to a ``Guard(...)`` instance, for precise tool-call attribution."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Name) and func.id == _GUARD_CTOR:
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _is_tool_registration(call: ast.Call, guards: set[str]) -> bool:
    """True if ``call`` is ``<guard>.tool(...)``.

    When at least one ``Guard`` instance is resolvable in the module, the receiver
    must be one of those names (precise). Otherwise we fall back to matching the
    ``.tool`` method by name -- conservative, and documented as a possible source
    of false positives on unrelated ``.tool`` methods.
    """
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == _TOOL_METHOD):
        return False
    if guards:
        return isinstance(func.value, ast.Name) and func.value.id in guards
    return True


def _functional_raw_sinks(tree: ast.Module, guards: set[str]) -> set[str]:
    """Raw callables left bound by the functional form ``w = guard.tool(raw)``.

    Decorator usage rebinds the function's own name to the wrapper, so it never
    appears here; only the non-decorator form leaks a directly-callable raw name.
    """
    sinks: set[str] = set()
    for node in ast.walk(tree):
        value: ast.expr | None = None
        if isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
        if isinstance(value, ast.Call) and _is_tool_registration(value, guards):
            sinks.update(arg.id for arg in value.args if isinstance(arg, ast.Name))
    return sinks


def lint_source(source: str, *, filename: str = "<unknown>") -> list[Finding]:
    """Return every mediation-bypass site in one module's source."""
    tree = ast.parse(source, filename=filename)
    guards = _guard_names(tree)
    raw_sinks = _functional_raw_sinks(tree, guards)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == _WRAPPED:
            findings.append(
                Finding(
                    "W1",
                    "access to '__wrapped__' reaches a tool's unmediated callable",
                    filename,
                    node.lineno,
                    node.col_offset,
                )
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in raw_sinks
        ):
            findings.append(
                Finding(
                    "W2",
                    f"direct call to raw sink {node.func.id!r} bypasses the monitor",
                    filename,
                    node.lineno,
                    node.col_offset,
                )
            )
    findings.sort(key=lambda f: (f.lineno, f.col, f.code))
    return findings


def lint_file(path: str | Path) -> list[Finding]:
    """Lint a single source file."""
    p = Path(path)
    return lint_source(p.read_text(encoding="utf-8"), filename=str(p))


def lint_paths(paths: Iterable[str | Path]) -> list[Finding]:
    """Lint every file in ``paths`` and concatenate the findings."""
    findings: list[Finding] = []
    for path in paths:
        findings.extend(lint_file(path))
    return findings
