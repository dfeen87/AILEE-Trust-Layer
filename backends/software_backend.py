from typing import Any, Dict, Optional, Sequence

from ailee_trust_pipeline_v1 import (
    AileeConfig,
    AileeTrustPipeline,
    DecisionResult,
)

from .base import BackendCapabilities


class SoftwareBackend:
    name = "software"

    def __init__(self, config: AileeConfig):
        self.pipeline = AileeTrustPipeline(config)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_safety_layer=True,
            supports_grace_layer=True,
            supports_consensus_layer=True,
            supports_fallback_layer=True,
            emits_hardware_metadata=False,
        )

    def process(
        self,
        raw_value: float,
        raw_confidence: Optional[float] = None,
        peer_values: Optional[Sequence[float]] = None,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResult:
        return self.pipeline.process(
            raw_value=raw_value,
            raw_confidence=raw_confidence,
            peer_values=peer_values,
            timestamp=timestamp,
            context=context,
        )
