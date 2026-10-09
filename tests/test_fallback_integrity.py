"""Fallback envelope, extreme arithmetic, routing, and atomic failure regressions."""

from dataclasses import replace
import math
import sys

import pytest

from ailee import (
    AileeConfig,
    AileeTrustPipeline,
    ConsensusStatus,
    GraceStatus,
    SafetyStatus,
)


def assert_finite_evidence(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite_evidence(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            assert_finite_evidence(child)


def snapshot(pipe):
    return (pipe.history[:], pipe.last_good_value, pipe.last_result)


def assert_unchanged(pipe, before):
    history, last_good, last_result = before
    assert pipe.history == history
    assert pipe.last_good_value == last_good
    assert pipe.last_result is last_result


@pytest.mark.parametrize(
    "clamps",
    [
        {"fallback_clamp_min": 2.0},
        {"fallback_clamp_max": -1.0},
        {"fallback_clamp_min": 2.0, "fallback_clamp_max": 3.0},
        {"fallback_clamp_min": -3.0, "fallback_clamp_max": -2.0},
    ],
)
def test_disjoint_fallback_and_hard_envelopes_are_rejected(clamps):
    # BEDROCK repro: hard [0, 1], fallback min 2 previously emitted/committed 2.
    with pytest.raises(ValueError):
        AileeConfig(hard_min=0.0, hard_max=1.0, **clamps)


@pytest.mark.parametrize(
    "clamps, expected",
    [
        ({"fallback_clamp_min": -0.5, "fallback_clamp_max": 1.5}, 0.5),
        ({"fallback_clamp_min": 0.75}, 0.75),
        ({"fallback_clamp_max": 0.25}, 0.25),
        ({"fallback_clamp_min": 1.0, "fallback_clamp_max": 2.0}, 1.0),
        ({"fallback_clamp_min": -1.0, "fallback_clamp_max": 0.0}, 0.0),
        ({"fallback_clamp_min": 0.25, "fallback_clamp_max": 0.75}, 0.5),
    ],
)
def test_compatible_fallback_clamps_preserve_policy(clamps, expected):
    pipe = AileeTrustPipeline(AileeConfig(hard_min=0.0, hard_max=1.0, **clamps))
    result = pipe.process(2.0, timestamp=10.0)
    assert result.used_fallback
    assert result.value == expected
    assert pipe.history == [(10.0, expected)]
    assert pipe.last_good_value is None
    assert_finite_evidence(result.metadata)


@pytest.mark.parametrize(
    "field, value",
    [
        ("fallback_clamp_min", 2.0),
        ("fallback_clamp_max", -1.0),
        ("hard_min", 2.0),
        ("hard_max", -1.0),
        ("fallback_clamp_min", math.nan),
        ("hard_max", math.inf),
        ("fallback_mode", "unknown"),
    ],
)
def test_mutated_invalid_fallback_configuration_fails_atomically(field, value):
    pipe = AileeTrustPipeline(
        AileeConfig(hard_min=0.0, hard_max=1.0, accept_threshold=0.0, enable_consensus=False)
    )
    pipe.process(0.5, timestamp=1.0)
    before = snapshot(pipe)
    setattr(pipe.cfg, field, value)
    with pytest.raises(ValueError):
        pipe.process(2.0, timestamp=2.0)
    assert_unchanged(pipe, before)


ROUTES = [
    ("hard", SafetyStatus.OUTRIGHT_REJECTED, GraceStatus.SKIPPED, ConsensusStatus.SKIPPED),
    ("safety", SafetyStatus.OUTRIGHT_REJECTED, GraceStatus.SKIPPED, ConsensusStatus.SKIPPED),
    ("grace_disabled", SafetyStatus.BORDERLINE, GraceStatus.SKIPPED, ConsensusStatus.SKIPPED),
    ("grace_fail", SafetyStatus.BORDERLINE, GraceStatus.FAIL, ConsensusStatus.SKIPPED),
    ("consensus_fail", SafetyStatus.ACCEPTED, GraceStatus.SKIPPED, ConsensusStatus.FAIL),
    ("grace_consensus_fail", SafetyStatus.BORDERLINE, GraceStatus.PASS, ConsensusStatus.FAIL),
]


@pytest.mark.parametrize("mode", ["median", "mean", "last_good"])
@pytest.mark.parametrize("route, safety, grace, consensus", ROUTES)
def test_every_fallback_route_preserves_bounds_status_and_history(
    mode, route, safety, grace, consensus
):
    settings = {"hard_min": 0.0, "hard_max": 1.0, "fallback_mode": mode}
    raw_value, peers = 0.5, None
    history = [0.25, 0.75]
    if route == "hard":
        raw_value = 2.0
    elif route == "grace_disabled":
        settings.update(borderline_low=0.5, enable_grace=False)
    elif route == "grace_fail":
        settings.update(borderline_low=0.0)
    elif route == "consensus_fail":
        settings.update(accept_threshold=0.0)
        peers = [0.9, 0.9]
    elif route == "grace_consensus_fail":
        settings.update(
            accept_threshold=1.0,
            borderline_low=0.0,
            borderline_high=1.0,
            grace_peer_delta=0.2,
            consensus_delta=0.01,
        )
        peers = [0.6, 0.6]
        history = [0.5] * 4
    pipe = AileeTrustPipeline(AileeConfig(**settings))
    pipe.history = [(float(index), value) for index, value in enumerate(history)]
    pipe.last_good_value = 0.4
    result = pipe.process(raw_value, raw_confidence=0.0, peer_values=peers, timestamp=10.0)

    expected = 0.4 if mode == "last_good" else 0.5
    assert (result.safety_status, result.grace_status, result.consensus_status) == (
        safety, grace, consensus
    )
    assert result.used_fallback
    assert result.value == expected
    assert 0.0 <= result.value <= 1.0
    assert pipe.history[-1] == (10.0, expected)
    assert pipe.last_good_value == 0.4
    assert pipe.last_result is result
    assert_finite_evidence(result.metadata)


@pytest.mark.parametrize(
    "mode, history, expected",
    [
        ("median", [1e308, 1e308], 1e308),
        ("median", [-1e308, -1e308], -1e308),
        ("mean", [1e308, 1e308], 1e308),
        ("mean", [-1e308, -1e308], -1e308),
        ("median", [-1e308, 1e308], 0.0),
        ("mean", [1e308, 1e308, -1e308, -1e308], 0.0),
        ("mean", [1e308, 1e308, -1e308, -1e308, 1.0], 0.2),
        ("mean", [-1e308, -1e308, 1e308, 1e308, -1.0], -0.2),
        ("median", [sys.float_info.max, sys.float_info.max], sys.float_info.max),
        ("mean", [-sys.float_info.max, -sys.float_info.max], -sys.float_info.max),
    ],
)
def test_extreme_finite_history_fallback_calculates_and_commits_finite_value(
    mode, history, expected
):
    # A one-sided hard boundary forces rejection without involving confidence
    # variance/GRACE arithmetic, so this directly isolates fallback arithmetic.
    bounds = {"hard_min": min(expected, 0.0)} if expected >= 0 else {"hard_max": 0.0}
    raw_value = -1.0 if expected >= 0 else 1.0
    pipe = AileeTrustPipeline(AileeConfig(fallback_mode=mode, **bounds))
    pipe.history = [(float(index), value) for index, value in enumerate(history)]
    result = pipe.process(raw_value, timestamp=20.0)
    assert result.used_fallback
    assert math.isfinite(result.value)
    assert result.value == expected
    assert pipe.history[-1] == (20.0, expected)
    assert pipe.last_good_value is None
    assert_finite_evidence(result.metadata)


@pytest.mark.parametrize("mode", ["median", "mean"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("consensus_failure", [False, True])
def test_extreme_history_fallback_after_safety_or_consensus_remains_finite(
    mode, sign, consensus_failure
):
    pipe = AileeTrustPipeline(
        AileeConfig(fallback_mode=mode, accept_threshold=0.0 if consensus_failure else 1.0)
    )
    pipe.history = [(0.0, sign * 1e308), (1.0, sign * 1e308)]
    result = pipe.process(0.0, peer_values=[1.0, 1.0] if consensus_failure else None, timestamp=2.0)
    assert result.used_fallback
    assert result.value == sign * 1e308
    assert result.consensus_status == (
        ConsensusStatus.FAIL if consensus_failure else ConsensusStatus.SKIPPED
    )
    assert pipe.history[-1] == (2.0, sign * 1e308)
    assert_finite_evidence(result.metadata)


@pytest.mark.parametrize(
    "minimum, maximum, expected",
    [
        (1e308, 1.4e308, 1.2e308),
        (-1.4e308, -1e308, -1.2e308),
        (-sys.float_info.max, sys.float_info.max, 0.0),
        (sys.float_info.max, sys.float_info.max, sys.float_info.max),
    ],
)
def test_empty_history_extreme_hard_bounds_use_stable_midpoint(minimum, maximum, expected):
    pipe = AileeTrustPipeline(AileeConfig(hard_min=minimum, hard_max=maximum))
    # Check the calculation as well as output: clipping infinity to a finite
    # endpoint conceals the bug and changes the intended midpoint policy.
    assert pipe._fallback_value() == expected
    result = pipe.process(0.0, timestamp=10.0)
    assert result.used_fallback
    assert result.value == expected
    assert minimum <= result.value <= maximum
    assert pipe.history == [(10.0, expected)]
    assert_finite_evidence(result.metadata)


@pytest.mark.parametrize("value", [-1e308, 1e308])
@pytest.mark.parametrize("has_last_good", [False, True])
def test_last_good_extreme_fallback_keeps_existing_policy(value, has_last_good):
    settings = {"hard_min": 0.0} if value > 0 else {"hard_max": 0.0}
    pipe = AileeTrustPipeline(AileeConfig(fallback_mode="last_good", **settings))
    pipe.history = [(0.0, value), (1.0, value)]
    if has_last_good:
        pipe.last_good_value = value
    result = pipe.process(-1.0 if value > 0 else 1.0, timestamp=2.0)
    assert result.used_fallback
    assert result.value == value
    assert pipe.last_good_value == (value if has_last_good else None)
    assert pipe.history[-1] == (2.0, value)
    assert_finite_evidence(result.metadata)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("source", ["median", "mean", "last_good"])
def test_invalid_fallback_state_fails_without_partial_commit(bad, source):
    pipe = AileeTrustPipeline(AileeConfig(hard_min=0.0, hard_max=1.0, fallback_mode=source))
    pipe.history = [(0.0, 0.5), (1.0, 0.5)]
    pipe.last_good_value = 0.5
    if source == "last_good":
        pipe.last_good_value = bad
    else:
        pipe.history.append((2.0, bad))
    before = snapshot(pipe)
    with pytest.raises(ValueError):
        pipe.process(2.0, timestamp=3.0)
    # NaN does not equal itself, so compare retained object identities for the
    # corrupted pre-existing state rather than interpreting it as valid data.
    assert pipe.history == before[0]
    assert pipe.last_good_value is before[1]
    assert pipe.last_result is before[2]


def test_nonfinite_derived_forecast_cannot_enter_fallback_audit_or_commit():
    pipe = AileeTrustPipeline(AileeConfig(history_window=1, borderline_low=0.4))
    pipe.history = [(0.0, 7e307), (1.0, 1.7e308), (2.0, 1.7e308)]
    pipe.last_good_value = 1.7e308
    before = snapshot(pipe)
    # The forecast overflows, GRACE fails for insufficient evidence, and the
    # fallback candidate itself is finite. Publishing an infinite forecast in
    # the resulting audit record still violates fallback integrity.
    with pytest.raises(ValueError):
        pipe.process(1.7e308, timestamp=3.0)
    assert_unchanged(pipe, before)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.0, 2.0])
def test_commit_guard_rejects_invalid_final_value_before_any_state_mutation(value):
    pipe = AileeTrustPipeline(
        AileeConfig(hard_min=0.0, hard_max=1.0, accept_threshold=0.0, enable_consensus=False)
    )
    previous = pipe.process(0.5, timestamp=1.0)
    before = snapshot(pipe)
    invalid_result = replace(previous, value=value, used_fallback=True)
    with pytest.raises(ValueError):
        pipe._commit_result(2.0, invalid_result, accepted=False)
    assert_unchanged(pipe, before)


def test_caller_context_is_preserved_without_treating_it_as_derived_audit():
    pipe = AileeTrustPipeline(AileeConfig(hard_min=0.0, hard_max=1.0))
    context = {"upstream_note": math.inf, "units": "score"}
    result = pipe.process(2.0, context=context, timestamp=1.0)
    assert result.value == 0.5
    assert result.metadata["context"] == context
    assert pipe.history == [(1.0, 0.5)]


@pytest.mark.parametrize("mode", ["median", "mean", "last_good"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_tightened_hard_envelope_bounds_existing_fallback_state(mode, sign):
    pipe = AileeTrustPipeline(
        AileeConfig(fallback_mode=mode, hard_min=-2.0, hard_max=2.0)
    )
    pipe.history = [(0.0, sign * 1.5), (1.0, sign * 1.5)]
    pipe.last_good_value = sign * 1.5
    pipe.cfg.hard_min, pipe.cfg.hard_max = -1.0, 1.0
    result = pipe.process(sign * 2.0, timestamp=2.0)
    assert result.used_fallback
    assert result.value == sign
    assert pipe.history[-1] == (2.0, sign)
    assert pipe.last_good_value == sign * 1.5
    assert_finite_evidence(result.metadata)


def test_extreme_integer_hard_bounds_keep_finite_midpoint_policy():
    pipe = AileeTrustPipeline(
        AileeConfig(hard_min=10**308, hard_max=14 * 10**307)
    )
    result = pipe.process(0.0, timestamp=1.0)
    assert result.value == 1.2e308
    assert pipe.history == [(1.0, 1.2e308)]
    assert_finite_evidence(result.metadata)


def test_extreme_confidence_arithmetic_failure_preserves_state():
    pipe = AileeTrustPipeline(AileeConfig())
    previous = pipe.process(0.0, timestamp=0.0)
    pipe.history = [(1.0, -1e308), (2.0, 1e308)]
    pipe.last_good_value = 0.0
    before = snapshot(pipe)
    with pytest.raises(OverflowError):
        pipe.process(0.0, timestamp=3.0)
    assert_unchanged(pipe, before)
    assert pipe.last_result is previous
