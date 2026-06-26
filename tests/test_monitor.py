"""Tests for the reference monitor (arch section 8): mediation, fail-closed, INV-6."""

from __future__ import annotations

import pytest

from warden.core import Node, NodeKind
from warden.labels import Confidentiality, Label, Taint
from warden.monitor import Monitor, WardenPolicyViolation
from warden.policy import ToolClass, compile_policy
from warden.propagate import ValueEnv

UNTRUSTED = Label(Taint.UNTRUSTED, provenance=frozenset({"fetch_url"}))
TRUSTED = Label(Taint.TRUSTED)


def _monitor(source: str) -> Monitor:
    return Monitor(compile_policy(source))


def test_allow_rule_permits_registered_consequential() -> None:
    monitor = _monitor("allow send_email if body.integrity == trusted")
    monitor.register("send_email", ToolClass.CONSEQUENTIAL)
    decision = monitor.mediate("send_email", {"body": TRUSTED})
    assert decision.allowed


def test_deny_raises_with_nonempty_provenance_path() -> None:
    monitor = _monitor("deny send_email if body.integrity != trusted")
    monitor.register("send_email", ToolClass.CONSEQUENTIAL)
    with pytest.raises(WardenPolicyViolation) as exc:
        monitor.mediate("send_email", {"body": UNTRUSTED})
    violation = exc.value
    assert violation.path  # INV-6: never empty
    assert any("UNTRUSTED" in step for step in violation.path)
    assert any("fetch_url" in step for step in violation.path)
    assert "send_email" in str(violation)


def test_unregistered_tool_defaults_consequential_and_denies() -> None:
    monitor = _monitor("allow known if x.integrity == trusted")
    with pytest.raises(WardenPolicyViolation):
        monitor.mediate("unknown_sink", {"x": TRUSTED})


def test_read_only_tool_defaults_allow() -> None:
    monitor = _monitor("deny send_email if body.integrity != trusted")
    monitor.register("read_doc", ToolClass.READ_ONLY)
    assert monitor.mediate("read_doc", {}).allowed


def test_fail_closed_when_policy_references_undefined_value() -> None:
    monitor = _monitor("deny send_email if body.integrity != trusted")
    monitor.register("send_email", ToolClass.CONSEQUENTIAL)
    with pytest.raises(WardenPolicyViolation) as exc:
        monitor.mediate("send_email", {})  # `body` is undefined
    assert "fail-closed" in exc.value.decision.reason


def test_mediate_handles_resolves_value_env() -> None:
    monitor = _monitor("deny send_email if body.integrity != trusted")
    monitor.register("send_email", ToolClass.CONSEQUENTIAL)
    node_id = Node(NodeKind.TOOL_RESULT, (), {"body": "..."}).id
    env = ValueEnv({"h_body": (node_id, UNTRUSTED)})
    with pytest.raises(WardenPolicyViolation):
        monitor.mediate_handles("send_email", {"body": "h_body"}, env)


def test_mediate_handles_fail_closed_on_unknown_handle() -> None:
    monitor = _monitor("allow send_email if body.integrity == trusted")
    monitor.register("send_email", ToolClass.CONSEQUENTIAL)
    with pytest.raises(WardenPolicyViolation) as exc:
        monitor.mediate_handles("send_email", {"body": "missing"}, ValueEnv())
    assert "unlabeled" in exc.value.decision.reason


def test_secret_confidentiality_appears_in_path() -> None:
    monitor = _monitor("deny post if data.confidentiality >= secret")
    monitor.register("post", ToolClass.CONSEQUENTIAL)
    secret = Label(confidentiality=Confidentiality.SECRET, provenance=frozenset({"db"}))
    with pytest.raises(WardenPolicyViolation) as exc:
        monitor.mediate("post", {"data": secret})
    assert any("SECRET" in step for step in exc.value.path)
