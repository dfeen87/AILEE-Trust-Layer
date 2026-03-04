"""Minimal smoke test for the AILEE trust pipeline."""
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
