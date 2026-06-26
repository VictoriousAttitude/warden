"""Tests for the label algebra (arch section 5): the semilattice laws and INV-3.

The join must be a commutative idempotent monoid with identity ``bottom`` and must
compute the least upper bound. These are the properties propagation relies on for
monotonicity, so they are checked with Hypothesis over arbitrary labels.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from warden.labels import Confidentiality, Label, Taint, join_all

_SOURCES = ["a", "b", "c", "d"]

labels = st.builds(
    Label,
    integrity=st.sampled_from(Taint),
    confidentiality=st.sampled_from(Confidentiality),
    provenance=st.frozensets(st.sampled_from(_SOURCES), max_size=4),
)


# --- semilattice laws -------------------------------------------------------


@given(labels)
def test_join_is_idempotent(a: Label) -> None:
    assert a.join(a) == a


@given(labels, labels)
def test_join_is_commutative(a: Label, b: Label) -> None:
    assert a.join(b) == b.join(a)


@given(labels, labels, labels)
def test_join_is_associative(a: Label, b: Label, c: Label) -> None:
    assert a.join(b).join(c) == a.join(b.join(c))


@given(labels)
def test_bottom_is_join_identity(a: Label) -> None:
    assert a.join(Label.bottom()) == a
    assert Label.bottom().join(a) == a


# --- join is the least upper bound ------------------------------------------


@given(labels, labels)
def test_join_is_upper_bound(a: Label, b: Label) -> None:
    j = a.join(b)
    assert a.leq(j)
    assert b.leq(j)


@given(labels, labels, labels)
def test_join_is_least_upper_bound(a: Label, b: Label, c: Label) -> None:
    # Any common upper bound of a and b also bounds their join.
    if a.leq(c) and b.leq(c):
        assert a.join(b).leq(c)


@given(labels, labels)
def test_leq_iff_join_absorbs(a: Label, b: Label) -> None:
    # a <= b  <=>  a join b == b  (the defining identity of a join-semilattice).
    assert a.leq(b) == (a.join(b) == b)


# --- partial order laws -----------------------------------------------------


@given(labels)
def test_leq_is_reflexive(a: Label) -> None:
    assert a.leq(a)


@given(labels, labels)
def test_leq_is_antisymmetric(a: Label, b: Label) -> None:
    if a.leq(b) and b.leq(a):
        assert a == b


# --- orientation (finding F2): join moves toward more-restrictive -----------


def test_taint_join_spreads_untrust() -> None:
    trusted = Label(Taint.TRUSTED)
    untrusted = Label(Taint.UNTRUSTED)
    assert trusted.join(untrusted).integrity is Taint.UNTRUSTED


def test_confidentiality_join_takes_max() -> None:
    public = Label(confidentiality=Confidentiality.PUBLIC)
    secret = Label(confidentiality=Confidentiality.SECRET)
    assert public.join(secret).confidentiality is Confidentiality.SECRET


def test_provenance_join_unions_sources() -> None:
    a = Label(provenance=frozenset({"fetch_url"}))
    b = Label(provenance=frozenset({"db"}))
    assert a.join(b).provenance == frozenset({"fetch_url", "db"})


def test_bottom_is_trusted_public_empty() -> None:
    b = Label.bottom()
    assert b.integrity is Taint.TRUSTED
    assert b.confidentiality is Confidentiality.PUBLIC
    assert b.provenance == frozenset()


# --- join_all ---------------------------------------------------------------


def test_join_all_of_empty_is_bottom() -> None:
    assert join_all([]) == Label.bottom()


@given(st.lists(labels, max_size=6))
def test_join_all_bounds_every_input(items: list[Label]) -> None:
    j = join_all(items)
    assert all(item.leq(j) for item in items)
