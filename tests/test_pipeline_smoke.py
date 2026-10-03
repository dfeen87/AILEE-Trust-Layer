# Licensed under the PolyForm Noncommercial License 1.0.0
"""Behavioral and invariant tests for the AILEE trust pipeline."""
import math

import pytest
from ailee import AileeTrustPipeline, AileeConfig


def test_pipeline_accepts_high_confidence():
    # With no history or peers, the composite score is ~0.57 (neutral defaults blended
    # with raw_confidence at 15%). Use a threshold compatible with cold-start behavior.
    cfg = AileeConfig(accept_threshold=0.50, borderline_low=0.30, borderline_high=0.50)
    pipe = AileeTrustPipeline(cfg)
    result = pipe.process(raw_value=10.0, raw_confidence=0.95)
    assert result.safety_status.value == "ACCEPTED"
    assert not result.used_fallback


def test_pipeline_rejects_low_confidence():
    pipe = AileeTrustPipeline(AileeConfig())
    result = pipe.process(raw_value=10.0, raw_confidence=0.30)
    assert result.used_fallback or result.safety_status.value != "ACCEPTED"


def test_hard_envelope_rejection():
    cfg = AileeConfig(hard_min=0.0, hard_max=100.0)
    pipe = AileeTrustPipeline(cfg)
    result = pipe.process(raw_value=999.0, raw_confidence=0.99)
    assert result.safety_status.value == "OUTRIGHT_REJECTED"
    assert result.used_fallback


def test_config_validation_rejects_invalid_fallback():
    with pytest.raises(ValueError, match="Invalid fallback_mode"):
        AileeConfig(fallback_mode="invalid")


def test_config_validation_rejects_invalid_weights():
    with pytest.raises(ValueError, match="weights must sum"):
        AileeConfig(w_stability=0.9, w_agreement=0.9, w_likelihood=0.9)


@pytest.mark.parametrize(
    "changes",
    [
        {"accept_threshold": math.nan},
        {"borderline_low": -0.01},
        {"borderline_high": 1.01},
        {"history_window": 0},
        {"forecast_window": -1},
        {"consensus_quorum": 0},
        {"grace_max_abs_z": 0.0},
        {"grace_forecast_epsilon": -0.01},
        {"grace_peer_delta": -0.01},
        {"consensus_delta": -0.01},
        {"consensus_pass_ratio": 1.01},
        {"hard_min": 2.0, "hard_max": 1.0},
        {"fallback_clamp_min": 2.0, "fallback_clamp_max": 1.0},
    ],
)
def test_config_rejects_unsafe_numeric_domains(changes):
    with pytest.raises((TypeError, ValueError)):
        AileeConfig(**changes)


@pytest.mark.parametrize(
    "arguments",
    [
        {"raw_value": math.nan},
        {"raw_value": math.inf},
        {"raw_value": 1.0, "raw_confidence": math.nan},
        {"raw_value": 1.0, "peer_values": [1.0, math.inf]},
        {"raw_value": 1.0, "timestamp": -math.inf},
    ],
)
def test_malformed_runtime_numbers_fail_before_state_mutation(arguments):
    pipe = AileeTrustPipeline(AileeConfig())
    original_state = (pipe.get_history_values(), pipe.last_good_value, pipe.last_result)

    with pytest.raises(ValueError, match="finite"):
        pipe.process(**arguments)

    assert (pipe.get_history_values(), pipe.last_good_value, pipe.last_result) == original_state
