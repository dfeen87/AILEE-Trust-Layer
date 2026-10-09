"""Exact temporal arithmetic across supported governance evidence types."""

import hashlib
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from numbers import Rational, Real

import pytest

from ailee.domains.governance import (
    GovernanceTrustLevel,
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


def assert_temporal_decision(decision, expected):
    assert decision.temporal_status == expected
    assert decision.actionable is (expected == TemporalStatus.VALID)
    assert decision.authorized_level is {
        TemporalStatus.VALID: GovernanceTrustLevel.FULL_TRUST,
        TemporalStatus.EXPIRED: GovernanceTrustLevel.NO_TRUST,
        TemporalStatus.NOT_YET_VALID: GovernanceTrustLevel.ADVISORY_TRUST,
    }[expected]


def assert_history(governor, decisions):
    assert governor.decision_history == decisions
    assert governor.get_decision_history() == decisions
    assert governor.get_events() == decisions


def test_decimal_bounds_work_with_default_float_grace_and_preserve_audit_evidence():
    governor = create_strict_governor()
    evidence = {
        "valid_from": Decimal("1.125"),
        "valid_until": Decimal("20000.125"),
        "issued_at": Decimal("0.125"),
        "timestamp": Decimal("10000.125"),
    }
    decision = governor.evaluate(**signal(**evidence))

    assert_temporal_decision(decision, TemporalStatus.VALID)
    for field, original in evidence.items():
        assert decision.metadata["temporal"][field] is original
    assert decision.timestamp is evidence["timestamp"]
    assert decision.decision_id == hashlib.sha256(
        b"official:10000.125:voter_registration_update:state:PA"
    ).hexdigest()[:16]
    assert decision.metadata["temporal"]["remaining_validity_seconds"] == 10_000
    assert_history(governor, [decision])


def test_fraction_grace_cannot_round_3600_seconds_to_4096_seconds():
    until = Fraction(2**64)
    decision = create_strict_governor().evaluate(
        **signal(valid_from=until - 1, valid_until=until, timestamp=until + 3800)
    )

    assert_temporal_decision(decision, TemporalStatus.EXPIRED)


@pytest.mark.parametrize("bound_type", [int, float, Decimal, Fraction])
@pytest.mark.parametrize("grace_type", [int, float, Decimal, Fraction])
def test_mixed_supported_types_preserve_temporal_evidence_and_exact_remaining_validity(
    bound_type, grace_type
):
    governor = create_strict_governor(grace_period_seconds=grace_type(2))
    evidence = {
        "valid_from": bound_type(10),
        "valid_until": bound_type(20),
        "issued_at": Decimal("8.25"),
        "timestamp": Fraction(49, 4),
    }
    decision = governor.evaluate(**signal(**evidence))

    assert_temporal_decision(decision, TemporalStatus.VALID)
    assert Fraction(decision.metadata["temporal"]["remaining_validity_seconds"]) == Fraction(31, 4)
    for field, original in evidence.items():
        assert decision.metadata["temporal"][field] is original
    assert decision.timestamp is evidence["timestamp"]


@pytest.mark.parametrize("number", [int, float, Decimal, Fraction])
@pytest.mark.parametrize(
    "offset, expected",
    [
        (-3601, TemporalStatus.NOT_YET_VALID),
        (-3600, TemporalStatus.VALID),
        (0, TemporalStatus.VALID),
        (3600, TemporalStatus.VALID),
        (3601, TemporalStatus.EXPIRED),
    ],
)
def test_zero_bounds_and_inclusive_default_grace_boundaries(number, offset, expected):
    decision = create_strict_governor().evaluate(
        **signal(valid_from=number(0), valid_until=number(0), timestamp=number(offset))
    )
    assert_temporal_decision(decision, expected)


@pytest.mark.parametrize("number", [Decimal, Fraction])
@pytest.mark.parametrize("expired", [False, True])
def test_exact_expiry_and_grace_boundaries_at_large_fractional_magnitudes(number, expired):
    with localcontext() as context:
        context.prec = 80
        point = number(2**64)
        epsilon = number("0.000000000000000000000000000001")
        grace = number(3600)
        boundary = point + grace if expired else point - grace
        outside = boundary + epsilon if expired else boundary - epsilon
    governor = create_strict_governor()
    at_boundary = governor.evaluate(
        **signal(valid_from=point, valid_until=point, timestamp=boundary)
    )
    past_boundary = governor.evaluate(
        **signal(valid_from=point, valid_until=point, timestamp=outside)
    )

    assert_temporal_decision(at_boundary, TemporalStatus.VALID)
    assert_temporal_decision(
        past_boundary, TemporalStatus.EXPIRED if expired else TemporalStatus.NOT_YET_VALID
    )
    assert_history(governor, [at_boundary, past_boundary])


@pytest.mark.parametrize("number", [int, Fraction])
@pytest.mark.parametrize(
    "offset, expected",
    [(3600, TemporalStatus.VALID), (3601, TemporalStatus.EXPIRED)],
)
def test_extreme_exact_integer_bounds_do_not_require_float_conversion(number, offset, expected):
    point = number(10**400)
    decision = create_strict_governor().evaluate(
        **signal(valid_from=point, valid_until=point, timestamp=point + offset)
    )
    assert_temporal_decision(decision, expected)


@pytest.mark.parametrize(
    "start, window",
    [
        (Decimal("10000.125"), 0.25),
        (Fraction(2**64), 10.0),
        (10**400, 10.0),
        (Decimal("10000.125"), Fraction(1, 3)),
        (Fraction(2**64), Decimal("0.125")),
    ],
)
def test_default_validity_uses_exact_mixed_arithmetic_and_inclusive_expiry(start, window):
    expiry = Fraction(start) + Fraction(window)
    epsilon = Fraction(1, 10**30)
    governor = create_strict_governor(
        require_temporal_bounds=False,
        default_validity_window_seconds=window,
        grace_period_seconds=0.0,
    )
    at_expiry = governor.evaluate(
        **signal(valid_from=None, valid_until=None, issued_at=start, timestamp=expiry)
    )
    expired = governor.evaluate(
        **signal(valid_from=None, valid_until=None, issued_at=start, timestamp=expiry + epsilon)
    )

    assert_temporal_decision(at_expiry, TemporalStatus.VALID)
    assert_temporal_decision(expired, TemporalStatus.EXPIRED)
    assert at_expiry.metadata["temporal"]["valid_from"] is start
    assert at_expiry.metadata["temporal"]["issued_at"] is start
    assert Fraction(at_expiry.metadata["temporal"]["valid_until"]) == expiry
    assert at_expiry.metadata["temporal"]["remaining_validity_seconds"] == 0
    assert_history(governor, [at_expiry, expired])


def test_float_default_window_does_not_round_a_fractional_expiry_outward():
    start = Fraction(2**64)
    governor = create_strict_governor(
        require_temporal_bounds=False,
        default_validity_window_seconds=3800.0,
        grace_period_seconds=0.0,
    )
    decision = governor.evaluate(
        **signal(valid_from=None, valid_until=None, issued_at=start, timestamp=start + 3900)
    )
    assert_temporal_decision(decision, TemporalStatus.EXPIRED)


def test_missing_issued_at_uses_exact_timestamp_for_default_window():
    timestamp = Decimal("10000.125")
    governor = create_strict_governor(
        require_temporal_bounds=False,
        default_validity_window_seconds=Fraction(1, 3),
    )
    decision = governor.evaluate(
        **signal(valid_from=None, valid_until=None, issued_at=None, timestamp=timestamp)
    )
    assert_temporal_decision(decision, TemporalStatus.VALID)
    assert decision.metadata["temporal"]["valid_from"] is timestamp
    assert Fraction(decision.metadata["temporal"]["valid_until"]) == Fraction(timestamp) + Fraction(1, 3)


def test_decimal_context_cannot_round_default_expiry_into_authorization():
    governor = create_strict_governor(
        require_temporal_bounds=False,
        default_validity_window_seconds=Decimal("0.6"),
        grace_period_seconds=Decimal("0"),
    )
    with localcontext() as context:
        context.prec = 5
        decision = governor.evaluate(
            **signal(
                valid_from=None,
                valid_until=None,
                issued_at=Decimal("10000"),
                timestamp=Decimal("10000.7"),
            )
        )
    assert_temporal_decision(decision, TemporalStatus.EXPIRED)


def test_decimal_context_cannot_round_remaining_validity_audit_evidence():
    governor = create_strict_governor(grace_period_seconds=Decimal("0"))
    until = Decimal("12345.6789012345")
    timestamp = Decimal("0")
    with localcontext() as context:
        context.prec = 5
        decision = governor.evaluate(
            **signal(valid_from=Decimal("0"), valid_until=until, timestamp=timestamp)
        )

    assert_temporal_decision(decision, TemporalStatus.VALID)
    assert decision.metadata["temporal"]["valid_until"] is until
    assert decision.metadata["temporal"]["timestamp"] is timestamp
    assert Fraction(decision.metadata["temporal"]["remaining_validity_seconds"]) == Fraction(until)


@Real.register
class UnsupportedReal:
    """Real-shaped input with a lossy float conversion but no exact encoding."""

    def __float__(self):
        return 10_000.0

    def __lt__(self, other):
        return 10_000.0 < other

    def __gt__(self, other):
        return 10_000.0 > other


@pytest.mark.parametrize("field", ["timestamp", "valid_from", "valid_until", "issued_at"])
@pytest.mark.parametrize("early_exit", ["revoked", "unverified", "out_of_scope"])
def test_unsupported_real_combinations_fail_before_any_decision_history_mutation(field, early_exit):
    governor = create_strict_governor()
    previous = governor.evaluate(**signal())
    overrides = {field: UnsupportedReal()}
    if early_exit == "revoked":
        governor.revoke_source("official")
    elif early_exit == "unverified":
        overrides["authority_level"] = None
    else:
        overrides["jurisdiction"] = "not-recognized"

    with pytest.raises((TypeError, ValueError)):
        governor.evaluate(**signal(**overrides))
    assert_history(governor, [previous])


@pytest.mark.parametrize("field", ["default_validity_window_seconds", "grace_period_seconds"])
def test_mutated_unsupported_numeric_configuration_fails_before_history_mutation(field):
    governor = create_strict_governor()
    previous = governor.evaluate(**signal())
    setattr(governor.cfg, field, UnsupportedReal())

    with pytest.raises((TypeError, ValueError)):
        governor.evaluate(**signal())
    assert_history(governor, [previous])


@Rational.register
class MalformedRational:
    """Registered numeric input whose advertised ratio violates Rational."""

    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def __le__(self, other):
        return False

    def __lt__(self, other):
        return False


@pytest.mark.parametrize("components", [(0, 0), (1.5, 1), (1, 0.5)])
@pytest.mark.parametrize("unverified", [False, True])
def test_malformed_registered_rational_cannot_authorize_or_commit_early_decision(
    components, unverified
):
    # Disabling metadata must not bypass validation of the governing ratio.
    governor = create_strict_governor(enable_audit_metadata=False)
    previous = governor.evaluate(**signal())
    overrides = {"timestamp": MalformedRational(*components)}
    if unverified:
        overrides["authority_level"] = None

    with pytest.raises((TypeError, ValueError)):
        governor.evaluate(**signal(**overrides))
    assert_history(governor, [previous])


@pytest.mark.parametrize("components", [(0, 0), (1.5, 1), (1, 0.5)])
def test_malformed_registered_rational_configuration_fails_before_history_mutation(components):
    governor = create_strict_governor()
    previous = governor.evaluate(**signal())
    governor.cfg.default_validity_window_seconds = MalformedRational(*components)

    with pytest.raises((TypeError, ValueError)):
        governor.evaluate(**signal(authority_level=None))
    assert_history(governor, [previous])


class HalfInt(int):
    """An int subclass advertising a supported non-integral exact ratio."""

    @property
    def numerator(self):
        return int(self)

    @property
    def denominator(self):
        return 2


def test_integer_subclass_ratio_cannot_widen_default_expiry_to_its_numerator():
    window = HalfInt(9)
    assert Fraction(window) == Fraction(9, 2)
    governor = create_strict_governor(
        require_temporal_bounds=False,
        default_validity_window_seconds=window,
        grace_period_seconds=0,
    )
    decision = governor.evaluate(
        **signal(valid_from=None, valid_until=None, issued_at=0, timestamp=5)
    )
    assert_temporal_decision(decision, TemporalStatus.EXPIRED)


@pytest.mark.parametrize("field", ["default_validity_window_seconds", "grace_period_seconds"])
def test_exact_negative_policy_duration_cannot_hide_behind_numeric_comparison_methods(field):
    governor = create_strict_governor()
    previous = governor.evaluate(**signal())
    setattr(governor.cfg, field, MalformedRational(-1, 1))

    with pytest.raises(ValueError, match=field):
        governor.evaluate(**signal(authority_level=None))
    assert_history(governor, [previous])


def test_float_default_validity_overflow_preserves_existing_fail_closed_contract():
    governor = create_strict_governor(
        require_temporal_bounds=False, default_validity_window_seconds=1e308
    )
    previous = governor.evaluate(**signal())
    with pytest.raises(ValueError, match="default valid_until"):
        governor.evaluate(
            **signal(valid_from=None, valid_until=None, issued_at=1e308, timestamp=1e308)
        )
    assert_history(governor, [previous])


@pytest.mark.parametrize(
    "bounds",
    [
        {"valid_from": 0.0, "valid_until": 1e308, "timestamp": 1e308},
        {"valid_from": -1e308, "valid_until": 0.0, "timestamp": -1e308},
    ],
)
def test_float_grace_boundary_overflow_cannot_create_unbounded_authorization(bounds):
    governor = create_strict_governor(grace_period_seconds=1e308)
    previous = governor.evaluate(**signal())
    with pytest.raises(ValueError):
        governor.evaluate(**signal(**bounds))
    assert_history(governor, [previous])


def test_float_remaining_validity_overflow_cannot_enter_successful_audit_evidence():
    governor = create_strict_governor(grace_period_seconds=0.0)
    previous = governor.evaluate(**signal())
    with pytest.raises(ValueError):
        governor.evaluate(
            **signal(valid_from=-1e308, valid_until=1e308, timestamp=-1e308)
        )
    assert_history(governor, [previous])


@pytest.mark.parametrize("timestamp_type", [int, float])
@pytest.mark.parametrize(
    "timestamp, expected",
    [(5, TemporalStatus.VALID), (10, TemporalStatus.VALID), (11, TemporalStatus.EXPIRED)],
)
def test_ordinary_default_window_keeps_int_and_float_policy_behavior(timestamp_type, timestamp, expected):
    governor = create_strict_governor(
        require_temporal_bounds=False,
        default_validity_window_seconds=10.0,
        grace_period_seconds=0.0,
    )
    decision = governor.evaluate(
        **signal(
            valid_from=None,
            valid_until=None,
            issued_at=timestamp_type(0),
            timestamp=timestamp_type(timestamp),
        )
    )
    assert_temporal_decision(decision, expected)


def test_finite_float_evidence_is_compared_as_its_exact_binary_value():
    # 0.1 + 0.2 rounds outward in binary float arithmetic. That rounded sum
    # must not enlarge the authorization interval of the original evidence.
    rounded_expiry = 0.1 + 0.2
    assert Fraction(rounded_expiry) > Fraction(0.1) + Fraction(0.2)
    governor = create_strict_governor(
        require_temporal_bounds=False,
        default_validity_window_seconds=0.2,
        grace_period_seconds=0.0,
    )
    decision = governor.evaluate(
        **signal(valid_from=None, valid_until=None, issued_at=0.1, timestamp=rounded_expiry)
    )
    assert_temporal_decision(decision, TemporalStatus.EXPIRED)


def test_float_expiry_boundary_remains_inclusive_when_exactly_representable():
    governor = create_strict_governor(grace_period_seconds=0.25)
    at_boundary = governor.evaluate(
        **signal(valid_from=0.0, valid_until=0.5, timestamp=0.75)
    )
    expired = governor.evaluate(
        **signal(valid_from=0.0, valid_until=0.5, timestamp=math.nextafter(0.75, math.inf))
    )
    assert_temporal_decision(at_boundary, TemporalStatus.VALID)
    assert_temporal_decision(expired, TemporalStatus.EXPIRED)
