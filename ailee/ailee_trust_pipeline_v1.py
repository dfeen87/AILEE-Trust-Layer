# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""
AILEE Trust Pipeline — v7.0.0
Single-file reference implementation for AI developers.

Implements a layered trust/validation pipeline:

NOTE:
This pipeline begins at the trust boundary. The upstream model stage
(raw data generation) is intentionally external and supplies
`raw_value` and optional `raw_confidence` into this pipeline.

1) Safety Layer      (confidence scoring + hard thresholds)
2) Grace Layer (2A)  (borderline mediation using context + trend + forecast checks)
3) Consensus Layer   (peer agreement + quorum)
4) Fallback          (rolling median/mean + last-known-good stability)
5) Final Output      (one trusted value + full audit metadata)

Design goals:
- Simple to clone and integrate
- Pluggable model + peer adapters
- Deterministic (no randomness)
- Auditable decisions (structured metadata, reasons, metrics)
- Safe defaults for real-time systems

Usage (minimal):

    pipeline = AileeTrustPipeline(AileeConfig())
    result = pipeline.process(
        raw_value=model_output,
        raw_confidence=model_confidence,
        peer_values=[...],  # optional
        timestamp=time.time(),
        context={"feature": "variable_x", "units": "ms"}
    )
    if result.used_fallback:
        ...
    value = result.value

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
import statistics
import time
import math


# -----------------------------
# Enums / Results
# -----------------------------

class SafetyStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    BORDERLINE = "BORDERLINE"
    OUTRIGHT_REJECTED = "OUTRIGHT_REJECTED"


class GraceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class ConsensusStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class DecisionResult:
    value: float
    safety_status: SafetyStatus
    grace_status: GraceStatus
    consensus_status: ConsensusStatus
    used_fallback: bool
    confidence_score: float
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class AileeConfig:
    # Safety confidence thresholds
    accept_threshold: float = 0.90
    borderline_low: float = 0.70
    borderline_high: float = 0.90

    # Confidence weights (stability, agreement, likelihood)
    w_stability: float = 0.45
    w_agreement: float = 0.30
    w_likelihood: float = 0.25

    # Historical window sizes
    history_window: int = 60          # for stability/variance, fallback median/mean
    forecast_window: int = 12         # for local forecast / trend checks

    # Grace checks (borderline mediation)
    grace_max_abs_z: float = 3.0      # plausibility band for deviation checks
    grace_forecast_epsilon: float = 0.15  # relative deviation allowed vs forecast
    grace_min_peer_agreement_ratio: float = 0.60  # fraction of peers within delta
    grace_peer_delta: float = 0.10    # absolute delta for peer agreement (domain-specific)
    agreement_delta: Optional[float] = None  # override for confidence agreement (defaults to grace_peer_delta)

    # Consensus checks
    consensus_quorum: int = 2         # minimum peers required to attempt consensus
    consensus_delta: float = 0.10     # absolute deviation allowed vs peer mean
    consensus_pass_ratio: float = 0.60 # fraction of peers within delta for pass

    # Fallback behavior
    fallback_mode: str = "median"     # "median" or "mean" or "last_good"
    fallback_clamp_min: Optional[float] = None
    fallback_clamp_max: Optional[float] = None

    # Optional domain hard constraints (safety envelope)
    hard_min: Optional[float] = None
    hard_max: Optional[float] = None

    # Operational controls
    enable_grace: bool = True
    enable_consensus: bool = True
    enable_audit_metadata: bool = True

    def __post_init__(self):
        self._validate_fallback_config()
        unit_interval_fields = (
            "accept_threshold",
            "borderline_low",
            "borderline_high",
            "w_stability",
            "w_agreement",
            "w_likelihood",
            "grace_min_peer_agreement_ratio",
            "consensus_pass_ratio",
        )
        for name in unit_interval_fields:
            value = _finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0]")
        if self.borderline_low > self.borderline_high:
            raise ValueError("borderline_low must be <= borderline_high")
        weight_sum = self.w_stability + self.w_agreement + self.w_likelihood
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Confidence weights must sum to ~1.0 (got {weight_sum:.3f})")
        for name in ("history_window", "forecast_window", "consensus_quorum"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if _finite_float(self.grace_max_abs_z, "grace_max_abs_z") <= 0.0:
            raise ValueError("grace_max_abs_z must be > 0.0")
        for name in (
            "grace_forecast_epsilon",
            "grace_peer_delta",
            "consensus_delta",
        ):
            if _finite_float(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be >= 0.0")
        if self.agreement_delta is not None and _finite_float(
            self.agreement_delta, "agreement_delta"
        ) < 0.0:
            raise ValueError("agreement_delta must be >= 0.0")
        for name in ("enable_grace", "enable_consensus", "enable_audit_metadata"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")

    def _validate_fallback_config(self) -> None:
        # Configuration remains mutable; process rechecks this safety boundary.
        valid_modes = frozenset({"median", "mean", "last_good"})
        if self.fallback_mode not in valid_modes:
            raise ValueError(
                f"Invalid fallback_mode '{self.fallback_mode}'. "
                f"Must be one of: {sorted(valid_modes)}"
            )
        for name in (
            "fallback_clamp_min",
            "fallback_clamp_max",
            "hard_min",
            "hard_max",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite_float(value, name)
        if (
            self.hard_min is not None
            and self.hard_max is not None
            and self.hard_min > self.hard_max
        ):
            raise ValueError("hard_min must be <= hard_max")
        if (
            self.fallback_clamp_min is not None
            and self.fallback_clamp_max is not None
            and self.fallback_clamp_min > self.fallback_clamp_max
        ):
            raise ValueError("fallback_clamp_min must be <= fallback_clamp_max")
        if (
            self.fallback_clamp_min is not None
            and self.hard_max is not None
            and self.fallback_clamp_min > self.hard_max
        ) or (
            self.fallback_clamp_max is not None
            and self.hard_min is not None
            and self.fallback_clamp_max < self.hard_min
        ):
            raise ValueError("fallback clamps must overlap the hard safety envelope")


# -----------------------------
# Helpers
# -----------------------------

def _finite_float(value: object, name: str) -> float:
    """Convert a numeric boundary value without admitting bool/NaN/infinity."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number

def _clamp(x: float, lo: Optional[float], hi: Optional[float]) -> float:
    if lo is not None and x < lo:
        return lo
    if hi is not None and x > hi:
        return hi
    return x


def _safe_midpoint(left: float, right: float) -> float:
    total = left + right
    if math.isfinite(total):
        return total / 2.0
    # Finite endpoints can overflow in their sum while the midpoint is finite.
    return left / 2.0 + right / 2.0


def _validate_audit_numbers(value: Any, name: str) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite_float(value, name)
    elif isinstance(value, dict):
        for key, child in value.items():
            _validate_audit_numbers(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_audit_numbers(child, f"{name}[{index}]")


def _safe_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return statistics.pstdev(values)  # population stdev (stable for streaming-ish usage)
    except statistics.StatisticsError:
        return 0.0


def _safe_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return statistics.pvariance(values)
    except statistics.StatisticsError:
        return 0.0


def _rolling(values: List[float], window: int) -> List[float]:
    if window <= 0:
        return values[:]
    return values[-window:]


def _relative_error(a: float, b: float, eps: float = 1e-12) -> float:
    denom = max(abs(b), eps)
    return abs(a - b) / denom


# -----------------------------
# Core Pipeline
# -----------------------------

class AileeTrustPipeline:
    """
    AILEE pipeline state is stored per-instance.
    For multiple variables, create one pipeline instance per signal/feature
    (or wrap this in a dict keyed by variable name).
    """

    def __init__(self, config: AileeConfig):
        self.cfg = config
        self.history: List[Tuple[float, float]] = []  # [(timestamp, value)]
        self.last_good_value: Optional[float] = None
        self.last_result: Optional[DecisionResult] = None

    # -------------------------
    # Public API
    # -------------------------

    def process(
        self,
        raw_value: float,
        raw_confidence: Optional[float] = None,
        peer_values: Optional[Sequence[float]] = None,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResult:
        # Validate and snapshot every untrusted numeric input before reading or
        # mutating pipeline state. This keeps malformed calls atomic and keeps
        # NaN/infinity out of confidence, consensus, history, and audit data.
        raw_value = _finite_float(raw_value, "raw_value")
        if raw_confidence is not None:
            raw_confidence = _finite_float(raw_confidence, "raw_confidence")
        ts = _finite_float(
            timestamp if timestamp is not None else time.time(), "timestamp"
        )
        peers = [
            _finite_float(value, f"peer_values[{index}]")
            for index, value in enumerate(
                peer_values if peer_values is not None else ()
            )
        ]
        ctx = dict(context or {})
        self.cfg._validate_fallback_config()
        history_cache: Dict[int, List[float]] = {}

        reasons: List[str] = []
        meta: Dict[str, Any] = {"timestamp": ts, "context": ctx} if self.cfg.enable_audit_metadata else {}

        # 0) Hard envelope check (domain safety)
        if not self._within_hard_bounds(raw_value):
            reasons.append("Hard envelope violation: raw_value outside [hard_min, hard_max].")
            value = self._fallback_value(history_cache)
            result = self._final_result(
                value=value,
                safety_status=SafetyStatus.OUTRIGHT_REJECTED,
                grace_status=GraceStatus.SKIPPED,
                consensus_status=ConsensusStatus.SKIPPED,
                used_fallback=True,
                confidence_score=0.0,
                reasons=reasons,
                meta=meta,
                raw_value=raw_value,
                raw_confidence=raw_confidence,
                peers=peers,
            )
            self._commit_result(ts, result, accepted=False)
            return result

        # 1) SAFETY LAYER: compute confidence score (preferred: internal score; fallback: raw_confidence)
        confidence_score = self._confidence_score(raw_value, peers, raw_confidence, meta, history_cache)

        safety_status = self._safety_decision(confidence_score, reasons)
        grace_status = GraceStatus.SKIPPED
        consensus_status = ConsensusStatus.SKIPPED

        # 2) ROUTING
        if safety_status == SafetyStatus.ACCEPTED:
            # Try consensus (optional)
            if self.cfg.enable_consensus:
                consensus_status = self._consensus_check(raw_value, peers, reasons, meta)
                if consensus_status == ConsensusStatus.PASS:
                    value = raw_value
                    result = self._final_result(value, safety_status, grace_status, consensus_status,
                                                used_fallback=False, confidence_score=confidence_score,
                                                reasons=reasons, meta=meta,
                                                raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
                    self._commit_result(ts, result, accepted=True)
                    return result
                if consensus_status == ConsensusStatus.SKIPPED:
                    reasons.append("Consensus SKIPPED after ACCEPTED -> accept raw_value.")
                    value = raw_value
                    result = self._final_result(value, safety_status, grace_status, consensus_status,
                                                used_fallback=False, confidence_score=confidence_score,
                                                reasons=reasons, meta=meta,
                                                raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
                    self._commit_result(ts, result, accepted=True)
                    return result
                # consensus fail -> fallback
                reasons.append("Consensus FAIL after ACCEPTED -> fallback.")
                value = self._fallback_value(history_cache)
                result = self._final_result(value, safety_status, grace_status, consensus_status,
                                            used_fallback=True, confidence_score=confidence_score,
                                            reasons=reasons, meta=meta,
                                            raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
                self._commit_result(ts, result, accepted=False)
                return result

            # Consensus disabled
            value = raw_value
            result = self._final_result(value, safety_status, grace_status, consensus_status,
                                        used_fallback=False, confidence_score=confidence_score,
                                        reasons=reasons, meta=meta,
                                        raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
            self._commit_result(ts, result, accepted=True)
            return result

        if safety_status == SafetyStatus.BORDERLINE and self.cfg.enable_grace:
            # 2A) GRACE LAYER
            grace_status = self._grace_check(raw_value, peers, reasons, meta, history_cache)

            if grace_status == GraceStatus.PASS:
                # Proceed to consensus (optional)
                if self.cfg.enable_consensus:
                    consensus_status = self._consensus_check(raw_value, peers, reasons, meta)
                    if consensus_status == ConsensusStatus.PASS:
                        value = raw_value
                        result = self._final_result(value, safety_status, grace_status, consensus_status,
                                                    used_fallback=False, confidence_score=confidence_score,
                                                    reasons=reasons, meta=meta,
                                                    raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
                        self._commit_result(ts, result, accepted=True)
                        return result
                    if consensus_status == ConsensusStatus.SKIPPED:
                        reasons.append("Consensus SKIPPED after GRACE PASS -> accept raw_value.")
                        value = raw_value
                        result = self._final_result(value, safety_status, grace_status, consensus_status,
                                                    used_fallback=False, confidence_score=confidence_score,
                                                    reasons=reasons, meta=meta,
                                                    raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
                        self._commit_result(ts, result, accepted=True)
                        return result
                    reasons.append("Consensus FAIL after GRACE PASS -> fallback.")
                    value = self._fallback_value(history_cache)
                    result = self._final_result(value, safety_status, grace_status, consensus_status,
                                                used_fallback=True, confidence_score=confidence_score,
                                                reasons=reasons, meta=meta,
                                                raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
                    self._commit_result(ts, result, accepted=False)
                    return result

                # Consensus disabled
                value = raw_value
                result = self._final_result(value, safety_status, grace_status, consensus_status,
                                            used_fallback=False, confidence_score=confidence_score,
                                            reasons=reasons, meta=meta,
                                            raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
                self._commit_result(ts, result, accepted=True)
                return result

            # Grace FAIL -> fallback
            reasons.append("Grace FAIL -> fallback.")
            value = self._fallback_value(history_cache)
            result = self._final_result(value, safety_status, grace_status, consensus_status,
                                        used_fallback=True, confidence_score=confidence_score,
                                        reasons=reasons, meta=meta,
                                        raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
            self._commit_result(ts, result, accepted=False)
            return result

        # OUTRIGHT REJECTED or borderline with grace disabled -> fallback
        reasons.append(f"Safety status {safety_status.value} -> fallback.")
        value = self._fallback_value(history_cache)
        result = self._final_result(value, safety_status, grace_status, consensus_status,
                                    used_fallback=True, confidence_score=confidence_score,
                                    reasons=reasons, meta=meta,
                                    raw_value=raw_value, raw_confidence=raw_confidence, peers=peers)
        self._commit_result(ts, result, accepted=False)
        return result

    # -------------------------
    # Layers
    # -------------------------

    def _within_hard_bounds(self, value: float) -> bool:
        if self.cfg.hard_min is not None and value < self.cfg.hard_min:
            return False
        if self.cfg.hard_max is not None and value > self.cfg.hard_max:
            return False
        return True

    def _safety_decision(self, confidence_score: float, reasons: List[str]) -> SafetyStatus:
        accept_threshold = self.cfg.accept_threshold
        borderline_high = min(self.cfg.borderline_high, accept_threshold)
        borderline_low = min(self.cfg.borderline_low, borderline_high)
        if borderline_high != self.cfg.borderline_high or borderline_low != self.cfg.borderline_low:
            reasons.append("Safety: thresholds normalized to maintain ordering.")
        if confidence_score >= accept_threshold:
            reasons.append(f"Safety: ACCEPTED (confidence_score={confidence_score:.3f} >= {accept_threshold:.2f}).")
            return SafetyStatus.ACCEPTED
        if borderline_low <= confidence_score < borderline_high:
            reasons.append(f"Safety: BORDERLINE ({borderline_low:.2f} <= {confidence_score:.3f} < {borderline_high:.2f}).")
            return SafetyStatus.BORDERLINE
        reasons.append(f"Safety: OUTRIGHT_REJECTED (confidence_score={confidence_score:.3f} < {borderline_low:.2f}).")
        return SafetyStatus.OUTRIGHT_REJECTED

    def _confidence_score(
        self,
        raw_value: float,
        peers: Sequence[float],
        raw_confidence: Optional[float],
        meta: Dict[str, Any],
        history_cache: Optional[Dict[int, List[float]]] = None,
    ) -> float:
        """
        Confidence Score = w_stability * Stability + w_agreement * Agreement + w_likelihood * Likelihood

        Stability: inverse-variance of recent history (0..1)
        Agreement: fraction peers close to raw_value (0..1)
        Likelihood: plausibility within +/- (grace_max_abs_z * sigma) of history mean (0..1)

        If raw_confidence is provided, we blend it as an extra signal (lightly),
        but the pipeline remains functional without it.
        """
        hist_vals = self._history_values(self.cfg.history_window, history_cache)
        variance = _safe_variance(hist_vals)
        if len(hist_vals) < 2:
            stability = 0.5  # neutral when insufficient history exists
        else:
            stability = 1.0 / (1.0 + variance)  # 0..1-ish

        agreement_delta = self.cfg.agreement_delta if self.cfg.agreement_delta is not None else self.cfg.grace_peer_delta
        agreement = self._agreement_score(raw_value, peers, delta=agreement_delta)
        likelihood = self._likelihood_score(raw_value, hist_vals, max_abs_z=self.cfg.grace_max_abs_z)

        score = (
            self.cfg.w_stability * stability +
            self.cfg.w_agreement * agreement +
            self.cfg.w_likelihood * likelihood
        )

        # Optional blending of model-provided confidence (kept bounded + not dominating)
        if raw_confidence is not None:
            rc = _clamp(float(raw_confidence), 0.0, 1.0)
            # Blend 15% external confidence into internal confidence score
            score = 0.85 * score + 0.15 * rc

        score = _clamp(score, 0.0, 1.0)

        if self.cfg.enable_audit_metadata:
            meta["confidence_components"] = {
                "stability": stability,
                "variance": variance,
                "agreement": agreement,
                "likelihood": likelihood,
                "raw_confidence_used": raw_confidence is not None,
            }
            meta["confidence_score"] = score

        return score

    def _agreement_score(self, raw_value: float, peers: Sequence[float], delta: float) -> float:
        if not peers:
            return 0.5  # neutral when no peers exist
        within = sum(1 for p in peers if abs(p - raw_value) <= delta)
        return within / max(1, len(peers))

    def _likelihood_score(self, raw_value: float, hist_vals: Sequence[float], max_abs_z: float) -> float:
        if len(hist_vals) < 4:
            return 0.5  # neutral prior when little history exists
        mu = statistics.fmean(hist_vals)
        sigma = _safe_stdev(hist_vals)
        if sigma <= 1e-12:
            # If no variation historically, any deviation is suspicious
            return 1.0 if abs(raw_value - mu) <= 1e-12 else 0.2
        z = (raw_value - mu) / sigma
        # Map |z| <= max_abs_z to (near) 1.0; beyond to 0.0 smoothly
        az = abs(z)
        if az >= max_abs_z:
            return 0.0
        return max(0.0, 1.0 - (az / max_abs_z))

    def _grace_check(
        self,
        raw_value: float,
        peers: Sequence[float],
        reasons: List[str],
        meta: Dict[str, Any],
        history_cache: Optional[Dict[int, List[float]]] = None,
    ) -> GraceStatus:
        """
        Grace Layer: contextual salvage of borderline data.
        Uses three deterministic checks:
        1) Trend plausibility (velocity/acceleration bounded)
        2) Forecast proximity (raw_value near local forecast)
        3) Early peer-context agreement
        """
        hist_vals = self._history_values(self.cfg.forecast_window, history_cache)

        passed_checks: List[str] = []
        available_checks: List[str] = []

        # (1) Trend plausibility
        trend_ok: Optional[bool] = None
        if len(hist_vals) >= 4:
            trend_ok = self._trend_check(raw_value, hist_vals)
            available_checks.append("trend_ok")
            if trend_ok:
                passed_checks.append("trend_ok")

        # (2) Forecast proximity
        forecast_ok: Optional[bool] = None
        forecast_val: Optional[float] = None
        if len(hist_vals) >= 3:
            forecast_ok, forecast_val = self._forecast_check(raw_value, hist_vals)
            available_checks.append("forecast_ok")
            if forecast_ok:
                passed_checks.append("forecast_ok")

        # (3) Peer agreement ratio
        peer_ok: Optional[bool] = None
        if peers:
            peer_ok = self._peer_context_check(raw_value, peers)
            available_checks.append("peer_ok")
            if peer_ok:
                passed_checks.append("peer_ok")

        if len(available_checks) < 2:
            reasons.append("Grace: FAIL (insufficient evidence for evaluation).")
            if self.cfg.enable_audit_metadata:
                meta["grace"] = {"passed_checks": passed_checks, "forecast": forecast_val}
            return GraceStatus.FAIL

        # Grace decision rule:
        # - PASS if at least 2 of 3 checks pass
        # - FAIL otherwise
        if len(passed_checks) >= 2:
            reasons.append(f"Grace: PASS ({passed_checks}).")
            if self.cfg.enable_audit_metadata:
                meta["grace"] = {"passed_checks": passed_checks, "forecast": forecast_val}
            return GraceStatus.PASS

        reasons.append(f"Grace: FAIL (passed={passed_checks}).")
        if self.cfg.enable_audit_metadata:
            meta["grace"] = {"passed_checks": passed_checks, "forecast": forecast_val}
        return GraceStatus.FAIL

    def _trend_check(self, raw_value: float, hist_vals: Sequence[float]) -> bool:
        # Need enough history to estimate velocity/acceleration
        if len(hist_vals) < 4:
            return True  # avoid over-penalizing early-stage systems

        # Compute simple velocity (delta) and acceleration (delta of delta)
        v1 = hist_vals[-1] - hist_vals[-2]
        v2 = hist_vals[-2] - hist_vals[-3]
        a = v1 - v2

        # Compare candidate step
        candidate_v = raw_value - hist_vals[-1]
        # Conservative plausibility: candidate velocity within a modest band of recent velocity
        # and candidate acceleration not exploding relative to recent acceleration scale.
        vel_band = max(1e-6, 3.0 * abs(v1) + 1e-6)
        acc_band = max(1e-6, 3.0 * abs(a) + 1e-6)

        return (abs(candidate_v) <= vel_band) and (abs(candidate_v - v1) <= acc_band)

    def _forecast_check(self, raw_value: float, hist_vals: Sequence[float]) -> Tuple[bool, Optional[float]]:
        if len(hist_vals) < 3:
            return True, None

        # Very lightweight forecast: last value + mean velocity
        velocities = [hist_vals[i] - hist_vals[i - 1] for i in range(1, len(hist_vals))]
        mean_v = statistics.fmean(velocities)
        forecast = hist_vals[-1] + mean_v

        # Accept if relative deviation <= epsilon OR absolute deviation <= peer_delta as a backup
        rel = _relative_error(raw_value, forecast)
        abs_ok = abs(raw_value - forecast) <= self.cfg.grace_peer_delta
        return (rel <= self.cfg.grace_forecast_epsilon) or abs_ok, forecast

    def _peer_context_check(self, raw_value: float, peers: Sequence[float]) -> bool:
        if not peers:
            return True  # no peers -> don't block grace
        within = sum(1 for p in peers if abs(p - raw_value) <= self.cfg.grace_peer_delta)
        ratio = within / max(1, len(peers))
        return ratio >= self.cfg.grace_min_peer_agreement_ratio

    def _consensus_check(self, raw_value: float, peers: Sequence[float], reasons: List[str], meta: Dict[str, Any]) -> ConsensusStatus:
        if len(peers) < self.cfg.consensus_quorum:
            reasons.append("Consensus: SKIPPED (insufficient peers).")
            return ConsensusStatus.SKIPPED

        peer_mean = statistics.fmean(peers)
        within = sum(1 for p in peers if abs(p - peer_mean) <= self.cfg.consensus_delta)
        ratio = within / max(1, len(peers))

        # Additionally require the raw_value to be within delta of peer_mean
        raw_ok = abs(raw_value - peer_mean) <= self.cfg.consensus_delta

        if self.cfg.enable_audit_metadata:
            meta["consensus"] = {
                "peer_mean": peer_mean,
                "within_ratio": ratio,
                "raw_within_peer_mean": raw_ok,
                "delta": self.cfg.consensus_delta,
                "quorum": self.cfg.consensus_quorum,
            }

        if raw_ok and ratio >= self.cfg.consensus_pass_ratio:
            reasons.append(f"Consensus: PASS (ratio={ratio:.2f}, raw_ok={raw_ok}).")
            return ConsensusStatus.PASS

        reasons.append(f"Consensus: FAIL (ratio={ratio:.2f}, raw_ok={raw_ok}).")
        return ConsensusStatus.FAIL

    # -------------------------
    # Fallback
    # -------------------------

    def _fallback_value(self, history_cache: Optional[Dict[int, List[float]]] = None) -> float:
        hist_vals = self._history_values(self.cfg.history_window, history_cache)

        if self.cfg.fallback_mode == "last_good" and self.last_good_value is not None:
            value = _finite_float(self.last_good_value, "last_good_value")
        elif not hist_vals:
            if self.cfg.hard_min is not None and self.cfg.hard_max is not None:
                value = _safe_midpoint(
                    _finite_float(self.cfg.hard_min, "hard_min"),
                    _finite_float(self.cfg.hard_max, "hard_max"),
                )
            elif self.cfg.hard_min is not None:
                value = self.cfg.hard_min
            elif self.cfg.hard_max is not None:
                value = self.cfg.hard_max
            else:
                value = 0.0
        else:
            hist_vals = [
                _finite_float(value, f"fallback history[{index}]")
                for index, value in enumerate(hist_vals)
            ]
            if self.cfg.fallback_mode == "mean":
                try:
                    value = statistics.fmean(hist_vals)
                except OverflowError:
                    # Exact standard-library accumulation avoids intermediate
                    # overflow and retains cancellation across finite extremes.
                    value = statistics.mean(hist_vals)
            else:
                # last_good without an accepted value retains median policy.
                ordered = sorted(hist_vals)
                middle = len(ordered) // 2
                value = ordered[middle] if len(ordered) % 2 else _safe_midpoint(
                    ordered[middle - 1], ordered[middle]
                )

        # Reject invalid calculations before clamping can conceal them. Hard
        # bounds are authoritative after legitimate fallback-policy clamps.
        value = _finite_float(value, "fallback value")
        value = _clamp(value, self.cfg.fallback_clamp_min, self.cfg.fallback_clamp_max)
        value = _clamp(value, self.cfg.hard_min, self.cfg.hard_max)
        return self._validate_result_value(value)

    def _validate_result_value(self, value: float) -> float:
        value = _finite_float(value, "result value")
        if not self._within_hard_bounds(value):
            raise ValueError("result value must be within the hard safety envelope")
        return value

    # -------------------------
    # Commit / Metadata
    # -------------------------

    def _final_result(
        self,
        value: float,
        safety_status: SafetyStatus,
        grace_status: GraceStatus,
        consensus_status: ConsensusStatus,
        used_fallback: bool,
        confidence_score: float,
        reasons: List[str],
        meta: Dict[str, Any],
        raw_value: float,
        raw_confidence: Optional[float],
        peers: Sequence[float],
    ) -> DecisionResult:
        value = self._validate_result_value(value)
        confidence_score = _finite_float(confidence_score, "confidence_score")
        md = self._finalize_meta(meta, raw_value, raw_confidence, peers)
        return DecisionResult(
            value=value,
            safety_status=safety_status,
            grace_status=grace_status,
            consensus_status=consensus_status,
            used_fallback=used_fallback,
            confidence_score=confidence_score,
            reasons=reasons[:],
            metadata=md,
        )

    def _finalize_meta(
        self,
        meta: Dict[str, Any],
        raw_value: float,
        raw_confidence: Optional[float],
        peers: Sequence[float],
    ) -> Dict[str, Any]:
        if not self.cfg.enable_audit_metadata:
            return {}
        meta = dict(meta)
        meta["input"] = {
            "raw_value": raw_value,
            "raw_confidence": raw_confidence,
            "peer_count": len(peers),
            "peer_values_sample": list(peers[:10]),
        }
        meta["state"] = {
            "history_len": len(self.history),
            "last_good_value": self.last_good_value,
        }
        # Context belongs to the caller and may contain arbitrary application
        # metadata. Every numeric field produced by the pipeline must be finite.
        for name, value in meta.items():
            if name != "context":
                _validate_audit_numbers(value, f"metadata.{name}")
        return meta

    def _commit_result(self, ts: float, result: DecisionResult, accepted: bool) -> None:
        value = self._validate_result_value(result.value)
        # Store final result and update history with the chosen value (not raw_value),
        # because downstream systems should learn from the trusted stream.
        self.last_result = result
        self.history.append((ts, value))
        self.history = _rolling(self.history, self.cfg.history_window * 4)  # keep a longer internal buffer

        if accepted and not result.used_fallback:
            self.last_good_value = value

    # -------------------------
    # Convenience: diagnostics
    # -------------------------

    def _history_values(
        self,
        window: int,
        history_cache: Optional[Dict[int, List[float]]] = None,
    ) -> List[float]:
        if history_cache is None:
            return [v for _, v in _rolling(self.history, window)]
        if window not in history_cache:
            history_cache[window] = [v for _, v in _rolling(self.history, window)]
        return history_cache[window]

    def get_last_result(self) -> Optional[DecisionResult]:
        return self.last_result

    def get_history_values(self, window: Optional[int] = None) -> List[float]:
        vals = [v for _, v in self.history]
        return _rolling(vals, window or self.cfg.history_window)


# -----------------------------
# Demo / Quick Self-Test
# -----------------------------
if __name__ == "__main__":
    cfg = AileeConfig(
        hard_min=0.0,
        hard_max=100.0,
        consensus_quorum=3,
        consensus_delta=2.5,
        grace_peer_delta=3.0,
        fallback_mode="median",
    )
    pipeline = AileeTrustPipeline(cfg)

    # Simulate a stream
    stream = [10, 10.2, 10.1, 10.15, 10.3, 18.0, 10.25, 10.2, 10.1, 200.0, 10.2]
    confidences = [0.95, 0.92, 0.93, 0.90, 0.88, 0.72, 0.91, 0.93, 0.94, 0.10, 0.96]

    for i, (x, c) in enumerate(zip(stream, confidences)):
        peers = [x + 0.3, x - 0.2, x + 0.1]  # pretend we have 3 peers near the raw value
        res = pipeline.process(
            raw_value=float(x),
            raw_confidence=float(c),
            peer_values=peers,
            timestamp=time.time(),
            context={"step": i, "feature": "variable_x"},
        )
        print(
            f"[{i}] raw={x:.3f} conf={c:.2f} -> value={res.value:.3f} | "
            f"safety={res.safety_status.value} grace={res.grace_status.value} "
            f"consensus={res.consensus_status.value} fallback={res.used_fallback}"
        )
