from typing import Any, Dict, Optional, Sequence, Tuple
import statistics

from ailee_trust_pipeline_v1 import (
    AileeConfig,
    AileeTrustPipeline,
    DecisionResult,
    SafetyStatus,
    GraceStatus,
    ConsensusStatus,
)

from .base import BackendCapabilities


class FeenBackend:
    name = "feen"

    def __init__(self, config: AileeConfig, fallback_to_software: bool = True):
        self.cfg = config
        self.software_fallback = (
            AileeTrustPipeline(config) if fallback_to_software else None
        )
        self._pyfeen = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_safety_layer=True,
            supports_grace_layer=False,     # v1: grace stays canonical in software
            supports_consensus_layer=True,
            supports_fallback_layer=False,
            emits_hardware_metadata=True,
        )

    def process(
        self,
        raw_value: float,
        raw_confidence: Optional[float] = None,
        peer_values: Optional[Sequence[float]] = None,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResult:
        peers = list(peer_values or [])
        ctx = dict(context or {})

        try:
            confidence, conf_meta = self._feen_confidence(raw_value, peers)
            safety, safety_reasons = self._safety_decision(confidence)

            consensus = ConsensusStatus.SKIPPED
            consensus_meta = {}
            consensus_reasons = []

            if len(peers) >= self.cfg.consensus_quorum:
                consensus, consensus_meta, consensus_reasons = self._consensus(
                    raw_value, peers
                )

            reasons = safety_reasons + consensus_reasons

            if safety == SafetyStatus.ACCEPTED and consensus != ConsensusStatus.FAIL:
                return DecisionResult(
                    value=raw_value,
                    safety_status=safety,
                    grace_status=GraceStatus.SKIPPED,
                    consensus_status=consensus,
                    used_fallback=False,
                    confidence_score=confidence,
                    reasons=reasons,
                    metadata={
                        "backend": "feen",
                        "context": ctx,
                        "feen": {
                            "confidence": conf_meta,
                            "consensus": consensus_meta,
                        },
                    },
                )

            return self._fallback(
                "FEEN decision requires software mediation.",
                raw_value,
                raw_confidence,
                peers,
                timestamp,
                ctx,
            )

        except Exception as e:
            return self._fallback(
                f"FEEN backend error: {e}",
                raw_value,
                raw_confidence,
                peers,
                timestamp,
                ctx,
            )

    # ---------------- FEEN primitives ----------------

    def _feen_confidence(self, raw_value: float, peers: Sequence[float]) -> Tuple[float, Dict[str, Any]]:
        delta = self.cfg.grace_peer_delta
        if not peers:
            agreement = 0.5
        else:
            agreement = sum(abs(p - raw_value) <= delta for p in peers) / len(peers)

        score = max(0.0, min(1.0, agreement))
        return score, {"agreement": agreement, "delta": delta}

    def _consensus(self, raw_value: float, peers: Sequence[float]):
        mean = statistics.fmean(peers)
        within = sum(abs(p - mean) <= self.cfg.consensus_delta for p in peers)
        ratio = within / len(peers)
        raw_ok = abs(raw_value - mean) <= self.cfg.consensus_delta

        meta = {"mean": mean, "ratio": ratio, "raw_ok": raw_ok}

        if raw_ok and ratio >= self.cfg.consensus_pass_ratio:
            return ConsensusStatus.PASS, meta, ["Consensus PASS"]
        return ConsensusStatus.FAIL, meta, ["Consensus FAIL"]

    def _safety_decision(self, score: float):
        if score >= self.cfg.accept_threshold:
            return SafetyStatus.ACCEPTED, ["Safety ACCEPTED"]
        if score >= self.cfg.borderline_low:
            return SafetyStatus.BORDERLINE, ["Safety BORDERLINE"]
        return SafetyStatus.OUTRIGHT_REJECTED, ["Safety REJECTED"]

    # ---------------- fallback ----------------

    def _fallback(
        self,
        reason: str,
        raw_value: float,
        raw_confidence: Optional[float],
        peers: Sequence[float],
        timestamp: Optional[float],
        context: Dict[str, Any],
    ) -> DecisionResult:
        if not self.software_fallback:
            return DecisionResult(
                value=raw_value,
                safety_status=SafetyStatus.OUTRIGHT_REJECTED,
                grace_status=GraceStatus.SKIPPED,
                consensus_status=ConsensusStatus.SKIPPED,
                used_fallback=True,
                confidence_score=0.0,
                reasons=[reason],
                metadata={"backend": "feen"},
            )

        res = self.software_fallback.process(
            raw_value, raw_confidence, peers, timestamp, context
        )
        res.metadata["backend"] = "feen"
        res.metadata["fallback_reason"] = reason
        return res
