import math
from dataclasses import FrozenInstanceError

import pytest

from ailee import AileeConfig, AileeTrustPipeline
from ailee.math_engine import (
    FORMULA_VERSION,
    DeltaVParameters,
    DeltaVSample,
    MathNumericalError,
    MathValidationError,
    attach_delta_v_evidence,
    build_delta_v_evidence,
    compute_delta_v,
    trapezoidal_integral,
)


def params(**changes):
    values = dict(
        specific_impulse=300.0, efficiency=0.8, alpha=0.0, initial_velocity=2.0
    )
    values.update(changes)
    return DeltaVParameters(**values)


def samples():
    return (
        DeltaVSample(0.0, 10.0, 1.0, 2.0, 5.0),
        DeltaVSample(2.0, 10.0, 1.0, 2.0, 5.0),
    )


@pytest.mark.parametrize(
    "points, expected",
    [
        (((0.0, 3.0), (4.0, 3.0)), 12.0),
        (((0.0, 1.0), (2.0, 5.0)), 6.0),
        (((0.0, 0.0), (1.0, 2.0), (3.0, 6.0)), 9.0),
        (((2.0, -1.0), (3.0, 1.0)), 0.0),
    ],
)
def test_trapezoidal_integral_known_values(points, expected):
    assert trapezoidal_integral(points) == pytest.approx(expected)


@pytest.mark.parametrize(
    "points",
    [
        ((0.0, 1.0),),
        ((1.0, 1.0), (0.0, 2.0)),
        ((0.0, 1.0), (0.0, 2.0)),
        ((0.0, math.nan), (1.0, 2.0)),
        ((0.0, 1.0), (math.inf, 2.0)),
    ],
)
def test_trapezoidal_integral_rejects_invalid_points(points):
    with pytest.raises(MathValidationError):
        trapezoidal_integral(points)


def test_alpha_zero_has_analytic_result():
    # Integral = (10 * 2 / 5) * 2 = 8; prefactor = 300 * .8 = 240.
    result = compute_delta_v(params(), samples())
    assert result.integral_value == pytest.approx(8.0)
    assert result.prefactor == pytest.approx(240.0)
    assert result.delta_v == pytest.approx(1920.0)
    assert result.formula_version == FORMULA_VERSION


def test_varying_trajectory_matches_manual_trapezoids_and_is_deterministic():
    trajectory = (
        DeltaVSample(0.0, 2.0, 0.5, -1.0, 4.0),
        DeltaVSample(1.0, 4.0, 1.0, 2.0, 2.0),
        DeltaVSample(3.0, 8.0, 1.5, 3.0, 4.0),
    )
    p = params(alpha=0.02, initial_velocity=4.0)
    first = compute_delta_v(p, trajectory)
    assert first == compute_delta_v(p, trajectory)
    factor = math.exp(2 * p.alpha * p.initial_velocity)
    f = [
        s.input_power
        * math.exp(-p.alpha * s.workload**2)
        * factor
        * s.velocity
        / s.mass
        for s in trajectory
    ]
    expected_integral = 0.5 * (f[0] + f[1]) + 0.5 * (f[1] + f[2]) * 2
    expected = (
        p.specific_impulse
        * p.efficiency
        * math.exp(-p.alpha * p.initial_velocity**2)
        * expected_integral
    )
    assert first.delta_v == pytest.approx(expected)


@pytest.mark.parametrize("alpha", [1e-15, 100.0])
def test_tiny_and_large_finite_damping(alpha):
    result = compute_delta_v(params(alpha=alpha, initial_velocity=0.0), samples())
    assert math.isfinite(result.delta_v)


def test_zero_efficiency_and_signed_velocity_are_permitted():
    assert compute_delta_v(params(efficiency=0.0), samples()).delta_v == 0.0
    signed = (DeltaVSample(0, 1, 0, -2, 1), DeltaVSample(1, 1, 0, -2, 1))
    assert compute_delta_v(params(), signed).delta_v < 0


@pytest.mark.parametrize(
    "changes",
    [
        {"specific_impulse": 0.0},
        {"specific_impulse": -1.0},
        {"efficiency": -0.1},
        {"efficiency": 1.1},
        {"alpha": -0.1},
        {"initial_velocity": math.nan},
        {"specific_impulse": math.inf},
        {"efficiency": True},
    ],
)
def test_invalid_parameters_are_rejected(changes):
    with pytest.raises(MathValidationError):
        compute_delta_v(params(**changes), samples())


@pytest.mark.parametrize(
    "bad_sample",
    [
        DeltaVSample(0, 1, 0, 1, 0),
        DeltaVSample(0, 1, 0, 1, -1),
        DeltaVSample(0, -1, 0, 1, 1),
        DeltaVSample(0, 1, math.nan, 1, 1),
        DeltaVSample(0, 1, 0, math.inf, 1),
        DeltaVSample(True, 1, 0, 1, 1),
    ],
)
def test_invalid_sample_fields_are_rejected(bad_sample):
    with pytest.raises(MathValidationError):
        compute_delta_v(params(), (bad_sample, DeltaVSample(1, 1, 0, 1, 1)))


@pytest.mark.parametrize(
    "series", [(), (DeltaVSample(0, 1, 0, 1, 1),), (object(), object())]
)
def test_malformed_or_short_series_are_rejected(series):
    with pytest.raises(MathValidationError):
        compute_delta_v(params(), series)


@pytest.mark.parametrize("second_time", [0.0, -1.0])
def test_sample_times_must_strictly_increase(second_time):
    series = (DeltaVSample(0, 1, 0, 1, 1), DeltaVSample(second_time, 1, 0, 1, 1))
    with pytest.raises(MathValidationError):
        compute_delta_v(params(), series)


def test_exponential_overflow_is_explicit():
    with pytest.raises(MathNumericalError, match="overflow"):
        compute_delta_v(params(alpha=400.0, initial_velocity=1.0), samples())


def test_nonfinite_final_result_is_explicit():
    series = (DeltaVSample(0, 2, 0, 1, 1), DeltaVSample(1, 2, 0, 1, 1))
    with pytest.raises(MathNumericalError, match="delta_v"):
        compute_delta_v(params(specific_impulse=1e308, efficiency=1.0), series)


def test_evidence_is_immutable_compact_and_context_is_snapshotted():
    result = compute_delta_v(params(), samples())
    evidence = build_delta_v_evidence(result)
    with pytest.raises(FrozenInstanceError):
        evidence.delta_v = 1  # type: ignore[misc]
    original = {"nested": {"owner": "caller"}}
    attached = attach_delta_v_evidence(original, result)
    original["nested"]["owner"] = "changed"
    assert attached["nested"]["owner"] == "caller"
    assert attached["ailee_delta_v"]["delta_v"] == result.delta_v
    assert "samples" not in attached["ailee_delta_v"]


def test_evidence_enters_audit_without_changing_trust_decision():
    config = AileeConfig(enable_consensus=False)
    plain = AileeTrustPipeline(config).process(4.0, raw_confidence=0.99, timestamp=10.0)
    context = attach_delta_v_evidence({}, compute_delta_v(params(), samples()))
    enriched = AileeTrustPipeline(config).process(
        4.0, raw_confidence=0.99, timestamp=10.0, context=context
    )
    assert (
        plain.value,
        plain.safety_status,
        plain.grace_status,
        plain.consensus_status,
        plain.used_fallback,
        plain.confidence_score,
    ) == (
        enriched.value,
        enriched.safety_status,
        enriched.grace_status,
        enriched.consensus_status,
        enriched.used_fallback,
        enriched.confidence_score,
    )
    assert (
        enriched.metadata["context"]["ailee_delta_v"]["formula_version"]
        == FORMULA_VERSION
    )
