# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""
Policy structures for Video Temporal Provenance domain.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class VideoTemporalPolicy:
    min_overall_trust: float = 70.0
    reject_synthetic_transitions: bool = False
    required_watermark_key: bytes = field(default_factory=bytes)
