"""Regressions for malformed evidence at the governance authorization boundary."""

import math
from decimal import Decimal
from fractions import Fraction

import pytest

from ailee.domains.governance import (
    GovernanceConfig,
    GovernanceGovernor,
    GovernanceTrustLevel,
    ScopeStatus,
    TemporalStatus,
    create_strict_governor,
)


def signal(**overrides):
    values = {
        "source": "official",
        "jurisdiction": "state:PA",
        "authority_level": "certified_official",
        "mandate": "voter_registration_update",
        "consent_proof": "proof",
        "valid_from": 1.0,
        "valid_until": 20_000.0,
        "timestamp": 10_000.0,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("field", ["timestamp", "valid_from", "valid_until", "issued_at"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, False])
def test_malformed_temporal_evidence_cannot_authorize_or_mutate_history(field, value):
    governor = create_strict_governor()
    previous = governor.evaluate(**signal())

    with pytest.raises((TypeError, ValueError)):
        governor.evaluate(**signal(**{field: value}))

    assert governor.decision_history == [previous]
    assert governor.get_decision_history() == [previous]
    assert governor.get_events() == [previous]


@pytest.mark.parametrize("depth", [-1, math.nan, math.inf, True, False, 1.5])
def test_malformed_delegation_depth_cannot_authorize_or_mutate_history(depth):
    governor = create_strict_governor()
    previous = governor.evaluate(**signal())

    with pytest.raises((TypeError, ValueError)):
        governor.evaluate(**signal(delegation_depth=depth))

    assert governor.decision_history == [previous]
    assert governor.get_decision_history() == [previous]
    assert governor.get_events() == [previous]


def test_contradictory_temporal_bounds_are_rejected_before_history_mutation():
    governor = create_strict_governor()
    with pytest.raises(ValueError, match="valid_from"):
        governor.evaluate(**signal(valid_from=20_000.0, valid_until=1.0))
    assert governor.decision_history == []
    assert governor.get_decision_history() == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"valid_from": -1.0, "valid_until": 0.0}, TemporalStatus.EXPIRED),
        (
            {"valid_from": 0.0, "valid_until": 20_000.0, "timestamp": -3601.0},
            TemporalStatus.NOT_YET_VALID,
        ),
    ],
)
def test_zero_temporal_bounds_are_evaluated(overrides, expected):
    decision = create_strict_governor().evaluate(**signal(**overrides))
    assert decision.temporal_status == expected
    assert not decision.actionable
    assert decision.authorized_level < GovernanceTrustLevel.CONSTRAINED_TRUST


def test_zero_issued_at_is_used_for_the_default_validity_window():
    governor = GovernanceGovernor(
        GovernanceConfig(
            require_temporal_bounds=False,
            default_validity_window_seconds=10.0,
            grace_period_seconds=0.0,
        )
    )
    decision = governor.evaluate(
        **signal(valid_from=None, valid_until=None, issued_at=0.0, timestamp=100.0)
    )
    assert decision.temporal_status == TemporalStatus.EXPIRED
    assert decision.authorized_level is GovernanceTrustLevel.NO_TRUST
    assert not decision.actionable


@pytest.mark.parametrize(
    "field, value",
    [
        (field, value)
        for field in ("default_validity_window_seconds", "grace_period_seconds")
        for value in (math.nan, math.inf, -math.inf, True, False, -1.0)
    ]
    + [("default_validity_window_seconds", 0.0)],
)
def test_malformed_temporal_configuration_is_rejected_at_construction(field, value):
    with pytest.raises((TypeError, ValueError)):
        GovernanceGovernor(GovernanceConfig(**{field: value}))


@pytest.mark.parametrize("depth", [-1, math.nan, math.inf, True, False, 1.5])
def test_invalid_configured_delegation_limit_is_rejected(depth):
    with pytest.raises((TypeError, ValueError)):
        GovernanceGovernor(GovernanceConfig(max_delegation_depth=depth))


@pytest.mark.parametrize(
    "field", ["default_validity_window_seconds", "grace_period_seconds"]
)
def test_mutated_nonfinite_configuration_cannot_authorize_or_mutate_history(field):
    governor = create_strict_governor()
    previous = governor.evaluate(**signal())
    setattr(governor.cfg, field, math.nan)

    with pytest.raises(ValueError):
        governor.evaluate(**signal(valid_until=2.0))

    assert governor.decision_history == [previous]
    assert governor.get_decision_history() == [previous]


@pytest.mark.parametrize("timestamp, actionable", [(3600.0, True), (3601.0, False)])
def test_existing_clock_grace_boundary_is_preserved(timestamp, actionable):
    decision = create_strict_governor().evaluate(
        **signal(valid_from=-1.0, valid_until=0.0, timestamp=timestamp)
    )
    assert decision.actionable is actionable


def test_registered_nonnegative_integer_delegation_remains_constrained():
    governor = create_strict_governor()
    governor.register_delegation("parent", "official")
    decision = governor.evaluate(**signal(delegation_depth=1, delegated_from="parent"))
    assert decision.delegation_valid
    assert decision.authorized_level is GovernanceTrustLevel.CONSTRAINED_TRUST
    assert decision.actionable


@pytest.mark.parametrize("jurisdiction", [None, "", "not-recognized"])
def test_strict_unknown_scope_never_authorizes_and_history_matches(jurisdiction):
    governor = create_strict_governor()
    decision = governor.evaluate(**signal(jurisdiction=jurisdiction))

    assert decision.scope_status == ScopeStatus.SCOPE_UNKNOWN
    assert decision.authorized_level is GovernanceTrustLevel.NO_TRUST
    assert not decision.actionable
    assert governor.decision_history == [decision]
    assert governor.get_decision_history() == [decision]
    assert governor.get_events() == [decision]


@pytest.mark.parametrize("jurisdiction", [None, "not-recognized"])
def test_explicitly_disabled_scope_enforcement_retains_its_existing_behavior(jurisdiction):
    governor = create_strict_governor(enforce_scope_boundaries=False)
    decision = governor.evaluate(**signal(jurisdiction=jurisdiction))

    assert decision.scope_status == ScopeStatus.SCOPE_UNKNOWN
    assert decision.authorized_level is GovernanceTrustLevel.FULL_TRUST
    assert decision.actionable


def test_optional_missing_jurisdiction_still_uses_existing_advisory_scope_policy():
    governor = create_strict_governor(require_jurisdiction=False)
    decision = governor.evaluate(**signal(jurisdiction=None))

    assert decision.scope_status == ScopeStatus.IN_SCOPE
    assert decision.authorized_level is GovernanceTrustLevel.FULL_TRUST
    assert decision.actionable


def test_strict_incompatible_scope_remains_denied():
    decision = create_strict_governor().evaluate(**signal(target_scope="state:NY"))
    assert decision.scope_status == ScopeStatus.OUT_OF_SCOPE
    assert decision.authorized_level is GovernanceTrustLevel.NO_TRUST
    assert not decision.actionable


def test_partial_required_temporal_evidence_remains_a_nonactionable_decision():
    decision = create_strict_governor().evaluate(**signal(valid_until=None))
    assert decision.temporal_status == TemporalStatus.EXPIRED
    assert decision.authorized_level is GovernanceTrustLevel.NO_TRUST
    assert not decision.actionable


def test_default_validity_overflow_cannot_leave_a_partial_decision():
    governor = create_strict_governor(
        require_temporal_bounds=False, default_validity_window_seconds=1e308
    )
    with pytest.raises(ValueError, match="default valid_until"):
        governor.evaluate(
            **signal(valid_from=None, valid_until=None, issued_at=1e308, timestamp=1e308)
        )
    assert governor.decision_history == []
    assert governor.get_decision_history() == []
    assert governor.get_events() == []


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (
            {
                "valid_from": 2**53 + 1,
                "valid_until": 2**53 + 2,
                "timestamp": 2**53,
            },
            TemporalStatus.NOT_YET_VALID,
        ),
        (
            {
                "valid_from": 2**53 - 1,
                "valid_until": 2**53,
                "timestamp": 2**53 + 1,
            },
            TemporalStatus.EXPIRED,
        ),
    ],
)
def test_large_integer_temporal_evidence_is_not_rounded_into_authorization(bounds, expected):
    governor = create_strict_governor(grace_period_seconds=0)
    decision = governor.evaluate(**signal(**bounds))
    assert decision.temporal_status == expected
    assert not decision.actionable


def test_large_integer_contradictory_bounds_cannot_round_to_equal_values():
    governor = create_strict_governor(grace_period_seconds=0)
    with pytest.raises(ValueError, match="valid_from"):
        governor.evaluate(
            **signal(valid_from=2**53 + 1, valid_until=2**53, timestamp=2**53)
        )
    assert governor.decision_history == []


def test_existing_integer_timestamp_audit_identity_is_preserved():
    decision = create_strict_governor().evaluate(
        **signal(valid_from=1, valid_until=20_000, timestamp=10_000)
    )
    # Recorded from the BEDROCK baseline for this valid integer-valued request.
    assert decision.decision_id == "03830819a3765d40"
    assert type(decision.timestamp) is int
    assert type(decision.metadata["temporal"]["valid_from"]) is int


@pytest.mark.parametrize("number", [Decimal, Fraction])
def test_supported_exact_numeric_evidence_preserves_valid_behavior(number):
    governor = create_strict_governor(grace_period_seconds=0)
    decision = governor.evaluate(
        **signal(
            valid_from=number(1), valid_until=number(20_000), timestamp=number(10_000)
        )
    )
    assert decision.authorized_level is GovernanceTrustLevel.FULL_TRUST
    assert decision.actionable
    assert type(decision.timestamp) is number
    assert type(decision.metadata["temporal"]["valid_from"]) is number


@pytest.mark.parametrize("number", [Decimal, Fraction])
@pytest.mark.parametrize("expired", [False, True])
def test_exact_fractional_temporal_precision_cannot_be_rounded_into_authorization(
    number, expired
):
    governor = create_strict_governor(grace_period_seconds=0)
    point = number(10_000)
    epsilon = number("0.00000000000000000000001")
    bounds = (
        {"valid_from": point - 1, "valid_until": point, "timestamp": point + epsilon}
        if expired
        else {"valid_from": point + epsilon, "valid_until": point + 1, "timestamp": point}
    )
    decision = governor.evaluate(**signal(**bounds))
    assert decision.temporal_status == (
        TemporalStatus.EXPIRED if expired else TemporalStatus.NOT_YET_VALID
    )
    assert not decision.actionable


@pytest.mark.parametrize("field", ["timestamp", "valid_from", "valid_until", "issued_at"])
@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity"), "10000"])
def test_nonfinite_decimal_and_numeric_strings_never_enter_history(field, value):
    governor = create_strict_governor()
    with pytest.raises((TypeError, ValueError)):
        governor.evaluate(**signal(**{field: value}))
    assert governor.decision_history == []
    assert governor.get_decision_history() == []
    assert governor.get_events() == []
