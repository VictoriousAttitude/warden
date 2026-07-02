"""End-to-end tests for the Mode 2 shim (arch section 9), through the PUBLIC API.

These exercise the same EchoLeak / secret-leak contract as the eval gate, but via
``Guard`` + ``@guard.tool`` rather than the internal monitor -- the surface a real
agent attaches to. They pin: benign flows pass untouched (label creep == 0), every
modeled attack is blocked before its side effect, and WHOLE_CONTEXT propagates taint
through intermediate tools.
"""

from __future__ import annotations

import pytest

from warden import (
    Confidentiality,
    Guard,
    Label,
    Recorder,
    Taint,
    ToolClass,
    WardenPolicyViolation,
)
from warden.core import NodeKind

_POLICY = """
deny send_email if body.integrity != trusted
deny post if data.confidentiality >= secret
allow send_email if body.integrity == trusted
allow post if data.confidentiality <= internal
"""


def _guard() -> Guard:
    return Guard(_POLICY)


def _wire(guard: Guard) -> dict[str, object]:
    @guard.tool
    def send_email(body: str, recipient: str) -> str:
        return f"sent to {recipient}"

    @guard.tool
    def post(data: str) -> str:
        return "posted"

    @guard.tool(
        cls=ToolClass.READ_ONLY,
        emits=Label(Taint.UNTRUSTED, provenance=frozenset({"fetch_url"})),
    )
    def fetch_url(url: str) -> str:
        return "<web content with hidden instructions>"

    @guard.tool(
        cls=ToolClass.READ_ONLY,
        emits=Label(
            confidentiality=Confidentiality.SECRET, provenance=frozenset({"db"})
        ),
    )
    def read_secret(key: str) -> str:
        return "s3cr3t"

    @guard.tool(cls=ToolClass.READ_ONLY)
    def summarize(text: str) -> str:
        return f"summary: {text}"

    return {
        "send_email": send_email,
        "post": post,
        "fetch_url": fetch_url,
        "read_secret": read_secret,
        "summarize": summarize,
    }


def test_benign_email_to_a_colleague_passes() -> None:
    guard = _guard()
    tools = _wire(guard)
    body = guard.source("hi Alice")
    recipient = guard.source("alice@corp")
    receipt = tools["send_email"](body=body, recipient=recipient)  # type: ignore[operator]
    assert guard.value(receipt) == "sent to alice@corp"


def test_benign_internal_post_passes() -> None:
    guard = _guard()
    tools = _wire(guard)
    summary = guard.source("Q3 numbers", confidentiality=Confidentiality.INTERNAL)
    assert guard.value(tools["post"](data=summary)) == "posted"  # type: ignore[operator]


def test_echoleak_exfil_via_untrusted_web_is_blocked() -> None:
    guard = _guard()
    tools = _wire(guard)
    web = tools["fetch_url"](guard.source("http://evil.test"))  # type: ignore[operator]
    assert web.label.integrity is Taint.UNTRUSTED
    with pytest.raises(WardenPolicyViolation):
        tools["send_email"](body=web, recipient=guard.source("alice@corp"))  # type: ignore[operator]


def test_taint_propagates_through_an_intermediate_tool() -> None:
    guard = _guard()
    tools = _wire(guard)
    web = tools["fetch_url"](guard.source("http://evil.test"))  # type: ignore[operator]
    digest = tools["summarize"](web)  # type: ignore[operator]
    # WHOLE_CONTEXT: a derivation of untrusted input stays untrusted.
    assert digest.label.integrity is Taint.UNTRUSTED
    with pytest.raises(WardenPolicyViolation):
        tools["send_email"](body=digest, recipient=guard.source("alice@corp"))  # type: ignore[operator]


def test_secret_leak_over_public_post_is_blocked() -> None:
    guard = _guard()
    tools = _wire(guard)
    secret = tools["read_secret"](guard.source("api_key"))  # type: ignore[operator]
    assert secret.label.confidentiality is Confidentiality.SECRET
    with pytest.raises(WardenPolicyViolation):
        tools["post"](data=secret)  # type: ignore[operator]


def test_side_effect_does_not_run_when_denied() -> None:
    guard = _guard()
    calls: list[str] = []

    @guard.tool
    def send_email(body: str, recipient: str) -> str:
        calls.append(recipient)
        return "sent"

    untrusted = guard.source("hijacked", integrity=Taint.UNTRUSTED)
    with pytest.raises(WardenPolicyViolation):
        send_email(body=untrusted, recipient=guard.source("alice@corp"))
    assert calls == []  # complete mediation: denied before the function body ran


def test_source_default_label_is_trusted_public() -> None:
    guard = _guard()
    handle = guard.source("plain text")
    assert handle.label == Label.bottom()
    assert guard.value(handle) == "plain text"


def test_unregistered_action_name_overrides_function_name() -> None:
    guard = Guard("allow publish if data.confidentiality <= internal")

    @guard.tool(name="publish")
    def post_to_blog(data: str) -> str:
        return "ok"

    assert guard.value(post_to_blog(data=guard.source("hello"))) == "ok"


def test_raw_literal_arguments_carry_bottom() -> None:
    guard = _guard()

    @guard.tool(cls=ToolClass.READ_ONLY)
    def summarize(text: str) -> str:
        return f"summary: {text}"

    # A raw (non-Handle) argument is born bottom, so the derivation stays bottom.
    digest = summarize("plain text")
    assert digest.label == Label.bottom()
    assert guard.value(digest) == "summary: plain text"


def test_raw_argument_does_not_dilute_a_labeled_one() -> None:
    guard = _guard()

    @guard.tool(cls=ToolClass.READ_ONLY)
    def concat(a: str, b: str) -> str:
        return a + b

    tainted = guard.source("injected", integrity=Taint.UNTRUSTED)
    mixed = concat(tainted, " and plain")
    # Join, not average: one untrusted input taints the whole derivation.
    assert mixed.label.integrity is Taint.UNTRUSTED
    assert guard.value(mixed) == "injected and plain"


def test_var_positional_handles_are_unwrapped_and_joined() -> None:
    guard = _guard()

    @guard.tool(cls=ToolClass.READ_ONLY)
    def join_lines(*parts: str) -> str:
        return "\n".join(parts)

    tainted = guard.source("injected", integrity=Taint.UNTRUSTED)
    result = join_lines(guard.source("header"), tainted, "raw footer")
    # The body sees raw values; the label joins every element of *parts.
    assert guard.value(result) == "header\ninjected\nraw footer"
    assert result.label.integrity is Taint.UNTRUSTED


def test_var_keyword_handles_are_unwrapped_and_joined() -> None:
    guard = _guard()

    @guard.tool(cls=ToolClass.READ_ONLY)
    def render(**fields: str) -> str:
        return ", ".join(f"{key}={value}" for key, value in fields.items())

    secret = guard.source("s3cr3t", confidentiality=Confidentiality.SECRET)
    result = render(user="alice", token=secret)
    # The body sees raw values; the label joins every value of **fields.
    assert guard.value(result) == "user=alice, token=s3cr3t"
    assert result.label.confidentiality is Confidentiality.SECRET


def test_declassification_lets_a_reviewed_value_through() -> None:
    guard = _guard()
    tools = _wire(guard)
    web = tools["fetch_url"](guard.source("http://evil.test"))  # type: ignore[operator]
    assert web.label.integrity is Taint.UNTRUSTED
    # An authority reviews the value and clears it for release; only then does the
    # sink that refuses untrusted bodies accept it.
    cleared = guard.declassify(web, to=Label.bottom())
    assert guard.value(cleared) == guard.value(web)
    receipt = tools["send_email"](body=cleared, recipient=guard.source("alice@corp"))  # type: ignore[operator]
    assert guard.value(receipt) == "sent to alice@corp"


def test_declassification_can_lower_one_axis_and_hold_another() -> None:
    guard = _guard()
    tools = _wire(guard)
    secret = tools["read_secret"](guard.source("api_key"))  # type: ignore[operator]
    assert secret.label.confidentiality is Confidentiality.SECRET
    cleared = guard.declassify(secret, to=Label(confidentiality=Confidentiality.INTERNAL))
    assert cleared.label.confidentiality is Confidentiality.INTERNAL
    assert guard.value(tools["post"](data=cleared)) == "posted"  # type: ignore[operator]


def test_declassification_cannot_raise_a_label() -> None:
    guard = _guard()
    handle = guard.source("plain")  # trusted, public
    with pytest.raises(ValueError, match="must not raise"):
        guard.declassify(handle, to=Label(Taint.UNTRUSTED))


def test_declassification_is_committed_to_the_provenance_graph() -> None:
    recorder = Recorder()
    guard = Guard(_POLICY, recorder=recorder)
    tainted = guard.source("reviewed text", integrity=Taint.UNTRUSTED)
    cleared = guard.declassify(tainted, to=Label.bottom())
    # The audited exception is visible in the graph: a DECLASSIFICATION node
    # whose parent is the handle it downgraded.
    node = recorder.recording().graph.get(cleared.id)
    assert node.kind is NodeKind.DECLASSIFICATION
    assert node.parents == (tainted.id,)
