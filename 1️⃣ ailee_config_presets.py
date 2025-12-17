"""
AILEE Configuration Presets
Drop-in configs for common AI domains.
"""

from ailee_trust_pipeline_v1 import AileeConfig

LLM_SCORING = AileeConfig(
    accept_threshold=0.92,
    borderline_low=0.75,
    borderline_high=0.92,
    consensus_quorum=3,
    consensus_delta=0.15,
    grace_peer_delta=0.20,
    fallback_mode="median",
)

SENSOR_FUSION = AileeConfig(
    accept_threshold=0.90,
    borderline_low=0.70,
    borderline_high=0.90,
    consensus_quorum=4,
    consensus_delta=1.5,
    grace_peer_delta=2.0,
    hard_min=0.0,
    hard_max=100.0,
)

FINANCIAL_SIGNAL = AileeConfig(
    accept_threshold=0.95,
    borderline_low=0.85,
    borderline_high=0.95,
    consensus_quorum=5,
    consensus_delta=0.05,
    fallback_mode="last_good",
)
