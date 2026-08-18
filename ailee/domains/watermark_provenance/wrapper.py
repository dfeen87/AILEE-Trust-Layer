# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
AILEE Integration Wrapper — Watermark-Provenance-Governance Domain
Provides legacy controller interception and trust enforcement for watermark governance workflows.
"""

from typing import Any, Dict, List, Optional, Callable
from .watermark_provenance import (
    WatermarkProvenanceGovernor,
    WatermarkProvenancePolicy,
    WatermarkProvenanceSignals,
    WatermarkProvenanceDecision,
    WatermarkProvenanceTrustLevel,
    WatermarkProvenanceControlDomain,
    WatermarkProvenanceControlAction,
    create_watermark_provenance_governor,
)


class WatermarkProvenanceWrapper:
    """
    Interception wrapper for legacy AI or watermark classification pipelines.
    Guarantees AILEE governance policies and non-binary provenance qualifiers are applied.
    """

    def __init__(
        self,
        legacy_fn: Callable[..., Any],
        governor: Optional[WatermarkProvenanceGovernor] = None,
        shadow_mode: bool = False,
        policy_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.legacy_fn = legacy_fn
        self.shadow_mode = shadow_mode
        if governor:
            self.governor = governor
        else:
            overrides = policy_overrides or {}
            self.governor = create_watermark_provenance_governor(**overrides)

    def execute(
        self,
        signals: WatermarkProvenanceSignals,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        decision = self.governor.evaluate(signals)

        if self.shadow_mode:
            legacy_result = self.legacy_fn(*args, **kwargs)
            return {
                "result": legacy_result,
                "shadow_decision": decision,
                "shadow_mode": True,
            }

        if not decision.actionable:
            return {
                "blocked": True,
                "reason": "AILEE Governance Policy: Action blocked due to insufficient trust or uncorroborated high-stakes context.",
                "decision": decision,
                "qualifiers": [q.value for q in decision.qualifiers],
                "provenance_confidence": decision.provenance_confidence,
            }

        legacy_result = self.legacy_fn(*args, **kwargs)
        return {
            "blocked": False,
            "result": legacy_result,
            "decision": decision,
            "qualifiers": [q.value for q in decision.qualifiers],
            "provenance_confidence": decision.provenance_confidence,
        }


def create_watermark_provenance_wrapper(
    legacy_fn: Callable[..., Any],
    governor: Optional[WatermarkProvenanceGovernor] = None,
    shadow_mode: bool = False,
    **policy_overrides: Any
) -> WatermarkProvenanceWrapper:
    return WatermarkProvenanceWrapper(
        legacy_fn=legacy_fn,
        governor=governor,
        shadow_mode=shadow_mode,
        policy_overrides=policy_overrides if policy_overrides else None,
    )
