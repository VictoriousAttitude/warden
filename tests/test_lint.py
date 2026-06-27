"""Tests for the static bypass-lint (INV-4, arch section 8.2(2)).

Pins the two bypass surfaces the lint must catch (W1 ``__wrapped__`` backdoor, W2
functional-form raw sink), that clean decorator-style code is silent, that the
registration site itself is not flagged, that guard-name tracking gives precision,
and -- the operational INV-4 gate -- that Warden's own source corpus has none.
"""

from __future__ import annotations

from pathlib import Path

from warden.lint import lint_paths, lint_source

_CLEAN = """
from warden import Guard, ToolClass

guard = Guard("allow send_email if body.integrity == trusted")

@guard.tool
def send_email(body, recipient):
    return "sent"

@guard.tool(cls=ToolClass.READ_ONLY)
def fetch(url):
    return "data"

result = send_email(body=fetch("http://x"), recipient="a@b")
"""

_WRAPPED_BACKDOOR = """
from warden import Guard

guard = Guard("deny send_email if body.integrity != trusted")

@guard.tool
def send_email(body, recipient):
    return "sent"

send_email.__wrapped__("hijacked", "victim@evil")
"""

_FUNCTIONAL_RAW = """
from warden import Guard

guard = Guard("deny send_email if body.integrity != trusted")

def send_email(body, recipient):
    return "sent"

wrapped = guard.tool(send_email)
send_email("hijacked", "victim@evil")
"""

_UNRELATED_TOOL_METHOD = """
from warden import Guard

guard = Guard("allow x if y.integrity == trusted")
toolbox = SomethingElse()

def raw(a):
    return a

handle = toolbox.tool(raw)
raw("fine, toolbox is not a Guard")
"""


def test_clean_decorator_code_is_silent() -> None:
    assert lint_source(_CLEAN, filename="clean.py") == []


def test_w1_flags_wrapped_backdoor() -> None:
    findings = lint_source(_WRAPPED_BACKDOOR, filename="bd.py")
    assert [f.code for f in findings] == ["W1"]
    assert findings[0].lineno == 10


def test_w2_flags_functional_raw_sink_call_only() -> None:
    findings = lint_source(_FUNCTIONAL_RAW, filename="fn.py")
    # The registration `guard.tool(send_email)` must NOT be flagged (send_email is
    # an argument there, not the call target); only the later direct call is.
    assert [f.code for f in findings] == ["W2"]
    assert "send_email" in findings[0].message
    assert findings[0].lineno == 10


def test_guard_name_tracking_ignores_unrelated_tool_method() -> None:
    # A Guard exists, so `toolbox.tool(raw)` (toolbox is not the guard) is not a
    # registration and `raw(...)` is therefore not a bypass.
    assert lint_source(_UNRELATED_TOOL_METHOD, filename="u.py") == []


def test_warden_corpus_has_no_mediation_bypass() -> None:
    root = Path(__file__).resolve().parent.parent
    corpus = sorted((root / "src" / "warden").rglob("*.py")) + sorted(
        (root / "tests").glob("*.py")
    )
    findings = lint_paths(corpus)
    assert findings == [], "\n".join(str(f) for f in findings)
