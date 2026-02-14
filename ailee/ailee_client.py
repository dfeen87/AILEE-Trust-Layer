from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

from .ailee_trust_pipeline_v1 import (
    AileeConfig,
    DecisionResult,
)

from .backends import SoftwareBackend, FeenBackend


class AileeClient:
    """
    Unified entrypoint for AILEE trust validation.

    Backend selection order:
      1) Explicit backend argument
      2) Environment variable AILEE_BACKEND
      3) Automatic (FEEN if available, else software)
    """

    def __init__(
        self,
        config: AileeConfig,
        backend: Optional[str] = None,
        enable_feen_fallback: bool = True,
    ):
        self.config = config

        backend_name = (
            backend
            or os.getenv("AILEE_BACKEND", "").strip().lower()
            or "auto"
        )

        self.backend_name = backend_name
        self.backend = self._select_backend(
            backend_name,
            enable_feen_fallback=enable_feen_fallback,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        raw_value: float,
        raw_confidence: Optional[float] = None,
        peer_values: Optional[Sequence[float]] = None,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResult:
        """
        Validate a value through the selected AILEE backend.
        """
        return self.backend.process(
            raw_value=raw_value,
            raw_confidence=raw_confidence,
            peer_values=peer_values,
            timestamp=timestamp,
            context=context,
        )

    def backend_capabilities(self) -> Dict[str, bool]:
        """
        Introspect backend capabilities for diagnostics or logging.
        """
        caps = self.backend.capabilities()
        return {
            "safety": caps.supports_safety_layer,
            "grace": caps.supports_grace_layer,
            "consensus": caps.supports_consensus_layer,
            "fallback": caps.supports_fallback_layer,
            "hardware_metadata": caps.emits_hardware_metadata,
        }

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _select_backend(self, name: str, enable_feen_fallback: bool):
        if name == "software":
            return SoftwareBackend(self.config)

        if name == "feen":
            return FeenBackend(
                self.config,
                fallback_to_software=enable_feen_fallback,
            )

        if name == "auto":
            try:
                return FeenBackend(
                    self.config,
                    fallback_to_software=True,
                )
            except (ImportError, RuntimeError, ConnectionError):
                return SoftwareBackend(self.config)

        raise ValueError(
            f"Unknown backend '{name}'. "
            "Valid options: 'software', 'feen', 'auto'."
        )
