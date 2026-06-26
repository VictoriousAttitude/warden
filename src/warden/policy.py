"""Capability policy: a small total DSL compiled to a side-effect-free predicate.

See arch section 7. The policy evaluator runs on attacker-INFLUENCED labels, so the
language is deliberately incorruptible: TOTAL (always terminates), side-effect-free,
no loops, no calls, no I/O. It is a tiny boolean language over the typed label model.

Per the project's F3 ethos -- own the security-critical parse rather than trust an
external grammar engine -- the tokenizer, recursive-descent parser, and evaluator
are hand-written and dependency-free. The grammar (arch section 7.2):

    policy  := rule*
    rule    := ("deny" | "allow") action ["if" expr]
    expr    := expr ("and" | "or") expr | "not" expr | "(" expr ")" | cmp
    cmp     := lvalue op literal | string "in" lvalue
    lvalue  := ident "." ("integrity" | "confidentiality" | "provenance")
    op      := "==" | "!=" | "<=" | ">=" | "<" | ">"

Integrity is read in the human "trusted is good" orientation (section 5.3): rank
TRUSTED above UNTRUSTED, so ``x.integrity >= trusted`` holds only for trusted
values. Confidentiality ranks PUBLIC < INTERNAL < SECRET. The translation from the
internal propagation lattice happens here, never inside the algebra.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum

from warden.labels import Confidentiality, Label, Taint

__all__ = [
    "Compare",
    "Decision",
    "Effect",
    "Expr",
    "Field",
    "Membership",
    "Op",
    "Policy",
    "PolicyError",
    "PolicySyntaxError",
    "PolicyTypeError",
    "Rule",
    "ToolClass",
    "compile_policy",
    "decide",
]


class PolicyError(Exception):
    """Base class for policy compilation errors."""


class PolicySyntaxError(PolicyError):
    """The policy source could not be tokenized or parsed."""


class PolicyTypeError(PolicyError):
    """The policy parsed but is not well-typed over the label model."""


class Effect(Enum):
    ALLOW = "allow"
    DENY = "deny"


class ToolClass(Enum):
    """Whether a tool merely reads (default-allow) or has side effects (default-deny)."""

    READ_ONLY = "read_only"
    CONSEQUENTIAL = "consequential"


class Field(StrEnum):
    INTEGRITY = "integrity"
    CONFIDENTIALITY = "confidentiality"
    PROVENANCE = "provenance"


class Op(StrEnum):
    EQ = "=="
    NE = "!="
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"


# --- AST ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LValue:
    ident: str
    field: Field


@dataclass(frozen=True, slots=True)
class Compare:
    left: LValue
    op: Op
    right: Taint | Confidentiality


@dataclass(frozen=True, slots=True)
class Membership:
    element: str
    container: LValue


@dataclass(frozen=True, slots=True)
class Not:
    operand: Expr


@dataclass(frozen=True, slots=True)
class And:
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Or:
    left: Expr
    right: Expr


type Expr = Compare | Membership | Not | And | Or


@dataclass(frozen=True, slots=True)
class Rule:
    effect: Effect
    action: str
    condition: Expr | None


@dataclass(frozen=True, slots=True)
class Policy:
    rules: tuple[Rule, ...]


# --- tokenizer ---------------------------------------------------------------

_SYMBOLS = {"==", "!=", "<=", ">=", "<", ">", "(", ")", "."}
_INTEGRITY_LITERALS = {"trusted": Taint.TRUSTED, "untrusted": Taint.UNTRUSTED}
_CONF_LITERALS = {
    "public": Confidentiality.PUBLIC,
    "internal": Confidentiality.INTERNAL,
    "secret": Confidentiality.SECRET,
}


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str  # "name" | "string" | "sym"
    value: str


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
        elif ch in {'"', "'"}:
            j = i + 1
            while j < n and source[j] != ch:
                j += 1
            if j >= n:
                raise PolicySyntaxError(f"unterminated string at offset {i}")
            tokens.append(_Token("string", source[i + 1 : j]))
            i = j + 1
        elif ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            tokens.append(_Token("name", source[i:j]))
            i = j
        elif source[i : i + 2] in _SYMBOLS:
            tokens.append(_Token("sym", source[i : i + 2]))
            i += 2
        elif ch in _SYMBOLS:
            tokens.append(_Token("sym", ch))
            i += 1
        else:
            raise PolicySyntaxError(f"unexpected character {ch!r} at offset {i}")
    return tokens


# --- parser (recursive descent) ---------------------------------------------


class _Parser:
    __slots__ = ("_pos", "_tokens")

    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> _Token | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _next(self) -> _Token:
        token = self._peek()
        if token is None:
            raise PolicySyntaxError("unexpected end of policy")
        self._pos += 1
        return token

    def _expect(self, kind: str, value: str) -> None:
        token = self._next()
        if token.kind != kind or token.value != value:
            raise PolicySyntaxError(f"expected {value!r}, got {token.value!r}")

    def _at(self, kind: str, value: str) -> bool:
        token = self._peek()
        return token is not None and token.kind == kind and token.value == value

    def parse_policy(self) -> Policy:
        rules: list[Rule] = []
        while self._peek() is not None:
            rules.append(self._parse_rule())
        return Policy(tuple(rules))

    def _parse_rule(self) -> Rule:
        head = self._next()
        if head.kind != "name" or head.value not in {"deny", "allow"}:
            raise PolicySyntaxError(f"expected 'deny' or 'allow', got {head.value!r}")
        effect = Effect.DENY if head.value == "deny" else Effect.ALLOW
        action_token = self._next()
        if action_token.kind != "name":
            raise PolicySyntaxError(f"expected an action name, got {action_token.value!r}")
        condition: Expr | None = None
        if self._at("name", "if"):
            self._next()
            condition = self._parse_expr()
        return Rule(effect, action_token.value, condition)

    def _parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._at("name", "or"):
            self._next()
            left = Or(left, self._parse_and())
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        while self._at("name", "and"):
            self._next()
            left = And(left, self._parse_not())
        return left

    def _parse_not(self) -> Expr:
        if self._at("name", "not"):
            self._next()
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> Expr:
        if self._at("sym", "("):
            self._next()
            inner = self._parse_expr()
            self._expect("sym", ")")
            return inner
        return self._parse_comparison()

    def _parse_comparison(self) -> Expr:
        token = self._next()
        if token.kind == "string":
            self._expect("name", "in")
            return Membership(token.value, self._parse_lvalue())
        if token.kind == "name":
            lvalue = self._finish_lvalue(token.value)
            op = self._parse_op()
            return Compare(lvalue, op, self._parse_literal(lvalue.field))
        raise PolicySyntaxError(f"unexpected token {token.value!r} in condition")

    def _parse_lvalue(self) -> LValue:
        token = self._next()
        if token.kind != "name":
            raise PolicySyntaxError(f"expected an identifier, got {token.value!r}")
        return self._finish_lvalue(token.value)

    def _finish_lvalue(self, ident: str) -> LValue:
        self._expect("sym", ".")
        field_token = self._next()
        if field_token.kind != "name":
            raise PolicySyntaxError(f"expected a field name, got {field_token.value!r}")
        try:
            field = Field(field_token.value)
        except ValueError:
            raise PolicyTypeError(
                f"unknown field {field_token.value!r} (M1 supports "
                "integrity, confidentiality, provenance)"
            ) from None
        return LValue(ident, field)

    def _parse_op(self) -> Op:
        token = self._next()
        try:
            return Op(token.value)
        except ValueError:
            raise PolicySyntaxError(
                f"expected a comparison operator, got {token.value!r}"
            ) from None

    def _parse_literal(self, field: Field) -> Taint | Confidentiality:
        token = self._next()
        if token.kind != "name":
            raise PolicySyntaxError(f"expected a literal, got {token.value!r}")
        if field is Field.INTEGRITY:
            if token.value not in _INTEGRITY_LITERALS:
                raise PolicyTypeError(f"{token.value!r} is not an integrity level")
            return _INTEGRITY_LITERALS[token.value]
        if field is Field.CONFIDENTIALITY:
            if token.value not in _CONF_LITERALS:
                raise PolicyTypeError(f"{token.value!r} is not a confidentiality level")
            return _CONF_LITERALS[token.value]
        raise PolicyTypeError("provenance is compared with the 'in' operator, not a level")


def compile_policy(source: str) -> Policy:
    """Tokenize, parse, and type-check ``source`` into a Policy (total, decidable)."""
    return _Parser(_tokenize(source)).parse_policy()


# --- evaluation (total, side-effect-free) ------------------------------------

_INTEGRITY_RANK = {Taint.TRUSTED: 1, Taint.UNTRUSTED: 0}
_CONF_RANK = {
    Confidentiality.PUBLIC: 0,
    Confidentiality.INTERNAL: 1,
    Confidentiality.SECRET: 2,
}


def _compare_ranks(actual: int, op: Op, expected: int) -> bool:
    match op:
        case Op.EQ:
            return actual == expected
        case Op.NE:
            return actual != expected
        case Op.LE:
            return actual <= expected
        case Op.GE:
            return actual >= expected
        case Op.LT:
            return actual < expected
        case Op.GT:
            return actual > expected


def _eval(expr: Expr, values: Mapping[str, Label]) -> bool:
    match expr:
        case Not(operand):
            return not _eval(operand, values)
        case And(left, right):
            return _eval(left, values) and _eval(right, values)
        case Or(left, right):
            return _eval(left, values) or _eval(right, values)
        case Membership(element, container):
            return element in _resolve(container.ident, values).provenance
        case Compare(left, op, right):
            label = _resolve(left.ident, values)
            if isinstance(right, Taint):
                return _compare_ranks(
                    _INTEGRITY_RANK[label.integrity], op, _INTEGRITY_RANK[right]
                )
            return _compare_ranks(
                _CONF_RANK[label.confidentiality], op, _CONF_RANK[right]
            )


def _resolve(ident: str, values: Mapping[str, Label]) -> Label:
    try:
        return values[ident]
    except KeyError:
        raise PolicyError(f"policy references undefined value {ident!r}") from None


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    rule: Rule | None
    reason: str


def decide(
    policy: Policy,
    action: str,
    values: Mapping[str, Label],
    tool_class: ToolClass,
) -> Decision:
    """Evaluate ``policy`` for one call. Deny beats allow; consequential default-deny.

    A referenced-but-undefined value raises PolicyError; the monitor treats that as a
    denial (fail-closed, INV-5). Read-only actions default-allow; consequential
    actions require a matching allow rule (arch section 7.3).
    """
    matched_allow: Rule | None = None
    for rule in policy.rules:
        if rule.action != action:
            continue
        if rule.condition is not None and not _eval(rule.condition, values):
            continue
        if rule.effect is Effect.DENY:
            return Decision(False, rule, f"denied by rule on {action!r}")
        matched_allow = matched_allow or rule

    if matched_allow is not None:
        return Decision(True, matched_allow, f"allowed by rule on {action!r}")
    if tool_class is ToolClass.CONSEQUENTIAL:
        return Decision(False, None, f"default-deny: no allow rule for consequential {action!r}")
    return Decision(True, None, f"default-allow: read-only {action!r}")
