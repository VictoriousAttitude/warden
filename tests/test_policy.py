"""Tests for the policy DSL and capability decision (arch section 7)."""

from __future__ import annotations

import pytest

from warden.labels import Confidentiality, Label, Taint
from warden.policy import (
    Effect,
    PolicyError,
    PolicySyntaxError,
    PolicyTypeError,
    ToolClass,
    compile_policy,
    decide,
)

UNTRUSTED = Label(Taint.UNTRUSTED, provenance=frozenset({"fetch_url"}))
TRUSTED = Label(Taint.TRUSTED)
SECRET = Label(confidentiality=Confidentiality.SECRET)


# --- parsing ----------------------------------------------------------------


def test_compiles_rule_with_condition() -> None:
    policy = compile_policy("deny send_email if body.integrity != trusted")
    assert len(policy.rules) == 1
    rule = policy.rules[0]
    assert rule.effect is Effect.DENY
    assert rule.action == "send_email"
    assert rule.condition is not None


def test_unknown_field_is_type_error() -> None:
    with pytest.raises(PolicyTypeError):
        compile_policy("deny x if y.readers == secret")


def test_bad_integrity_literal_is_type_error() -> None:
    with pytest.raises(PolicyTypeError):
        compile_policy("deny x if y.integrity == secret")


def test_unterminated_string_is_syntax_error() -> None:
    with pytest.raises(PolicySyntaxError):
        compile_policy("deny x if 'oops in y.provenance")


def test_missing_operator_is_syntax_error() -> None:
    with pytest.raises(PolicySyntaxError):
        compile_policy("deny x if y.integrity trusted")


# --- decision semantics -----------------------------------------------------


def test_deny_rule_fires_on_untrusted() -> None:
    policy = compile_policy("deny send_email if body.integrity != trusted")
    decision = decide(policy, "send_email", {"body": UNTRUSTED}, ToolClass.CONSEQUENTIAL)
    assert not decision.allowed
    assert decision.rule is not None


def test_deny_rule_does_not_fire_on_trusted_but_default_deny_does() -> None:
    policy = compile_policy("deny send_email if body.integrity != trusted")
    # Condition false (trusted), but consequential with no allow rule => default-deny.
    decision = decide(policy, "send_email", {"body": TRUSTED}, ToolClass.CONSEQUENTIAL)
    assert not decision.allowed
    assert decision.rule is None


def test_allow_rule_permits_consequential() -> None:
    policy = compile_policy("allow send_email if body.integrity == trusted")
    decision = decide(policy, "send_email", {"body": TRUSTED}, ToolClass.CONSEQUENTIAL)
    assert decision.allowed


def test_read_only_defaults_allow() -> None:
    policy = compile_policy("deny send_email if body.integrity != trusted")
    decision = decide(policy, "read_doc", {}, ToolClass.READ_ONLY)
    assert decision.allowed
    assert decision.rule is None


def test_deny_beats_allow() -> None:
    policy = compile_policy(
        "allow send_email if body.integrity == untrusted\n"
        "deny send_email if body.integrity == untrusted"
    )
    decision = decide(policy, "send_email", {"body": UNTRUSTED}, ToolClass.CONSEQUENTIAL)
    assert not decision.allowed
    assert decision.rule is not None and decision.rule.effect is Effect.DENY


def test_undefined_value_reference_raises() -> None:
    policy = compile_policy("deny send_email if body.integrity != trusted")
    with pytest.raises(PolicyError):
        decide(policy, "send_email", {}, ToolClass.CONSEQUENTIAL)


# --- expressions ------------------------------------------------------------


def test_boolean_and_or_not() -> None:
    policy = compile_policy(
        "deny act if (a.integrity != trusted or b.confidentiality >= secret) "
        "and not c.integrity == untrusted"
    )
    fire = decide(
        policy,
        "act",
        {"a": UNTRUSTED, "b": TRUSTED, "c": TRUSTED},
        ToolClass.CONSEQUENTIAL,
    )
    assert not fire.allowed
    # c is untrusted => the `not` clause is false => whole condition false => no deny.
    no_fire = decide(
        policy,
        "act",
        {"a": UNTRUSTED, "b": TRUSTED, "c": UNTRUSTED},
        ToolClass.READ_ONLY,
    )
    assert no_fire.allowed


def test_provenance_membership() -> None:
    policy = compile_policy("deny send_email if 'fetch_url' in body.provenance")
    fired = decide(policy, "send_email", {"body": UNTRUSTED}, ToolClass.CONSEQUENTIAL)
    assert not fired.allowed
    clean = decide(
        policy, "send_email", {"body": Label(provenance=frozenset({"db"}))}, ToolClass.READ_ONLY
    )
    assert clean.allowed


def test_confidentiality_ordering() -> None:
    policy = compile_policy("deny post if data.confidentiality >= internal")
    assert not decide(
        policy, "post", {"data": SECRET}, ToolClass.CONSEQUENTIAL
    ).allowed
    assert decide(
        policy,
        "post",
        {"data": Label(confidentiality=Confidentiality.PUBLIC)},
        ToolClass.READ_ONLY,
    ).allowed


def test_integrity_human_orientation() -> None:
    # `>= trusted` holds only for trusted (trusted ranks above untrusted).
    policy = compile_policy("allow act if x.integrity >= trusted")
    assert decide(policy, "act", {"x": TRUSTED}, ToolClass.CONSEQUENTIAL).allowed
    assert not decide(policy, "act", {"x": UNTRUSTED}, ToolClass.CONSEQUENTIAL).allowed
