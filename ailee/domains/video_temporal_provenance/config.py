# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""
Configuration structures for Video Temporal Provenance domain.
"""

from dataclasses import dataclass

@dataclass
class VideoTemporalConfig:
    min_trust_threshold: float = 75.0
    strict_mode: bool = False
    enable_watermark_verification: bool = True
    max_allowed_anomalies: int = 1
    flow_stability_threshold: float = 0.5
