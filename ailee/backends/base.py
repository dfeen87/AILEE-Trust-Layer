from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

from ..ailee_trust_pipeline_v1 import DecisionResult


@dataclass(frozen=True)
class BackendCapabilities:
    supports_safety_layer: bool = True
    supports_grace_layer: bool = False
    supports_consensus_layer: bool = False
    supports_fallback_layer: bool = False
    emits_hardware_metadata: bool = False


class AileeBackend(Protocol):
    name: str

    def capabilities(self) -> BackendCapabilities:
        ...

    def process(
        self,
        raw_value: float,
        raw_confidence: Optional[float] = None,
        peer_values: Optional[Sequence[float]] = None,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResult:
        ...
