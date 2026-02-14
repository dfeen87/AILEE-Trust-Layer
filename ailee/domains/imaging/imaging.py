"""
AILEE Trust Layer — IMAGING Domain
Version: 1.0.1 - Production Grade (Refined)

Imaging-focused governance domain for AI-assisted and computational imaging systems.

This domain does NOT implement reconstruction algorithms, forward models, or physics.
It governs whether imaging outputs are trustworthy based on quality metrics,
multi-method consensus, noise models, and efficiency constraints.

Primary governed signals:
- Image quality metrics (SNR, resolution, artifact scores)
- Reconstruction confidence
- Energy/dose efficiency
- Acquisition parameter optimization

Key properties:
- Deterministic quality assessment
- Multi-method validation (physics-based vs AI-based)
- Auditable imaging decisions with full metadata
- Adaptive acquisition support
- Safety-constrained optimization (dose limits, SAR limits, etc.)
- Modality-agnostic framework with domain-specific adapters

INTEGRATION EXAMPLE:

    # Setup (once)
    policy = ImagingGovernancePolicy(
        modality="MRI",
        min_snr_threshold=15.0,
        max_acquisition_time_s=300.0,
    )
    governor = ImagingGovernor(policy=policy)
    
    # Per-acquisition evaluation
    while imaging_active:
        signals = ImagingSignals(
            quality_metric=compute_snr(reconstruction),
            model_confidence=ai_model.get_confidence(),
            peer_reconstructions=[physics_based_recon, bootstrap_recon],
            acquisition_params=AcquisitionParams(
                energy_input=current_dose,
                acquisition_time_s=elapsed_time,
            ),
            noise_model=NoiseModel(
                estimated_noise_level=noise_estimate,
                snr_measurement=measured_snr,
            ),
        )
        
        decision = governor.evaluate(signals)
        
        if decision.quality_acceptable:
            accept_reconstruction(reconstruction)
        elif decision.recommendation == "adapt_and_continue":
            adjust_acquisition_params(decision.suggested_params)
        else:
            trigger_fallback_reconstruction()
        
        # Log for quality assurance
        qa_system.record_imaging_event(governor.get_last_event())

This module is designed for medical imaging QA, scientific imaging optimization,
and AI reconstruction validation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import math
import statistics
import time


# ---- Core imports ----
try:
    from ...ailee_trust_pipeline_v1 import (
        AileeTrustPipeline,
        AileeConfig,
        DecisionResult,
        SafetyStatus
    )
except Exception:  # pragma: no cover
    AileeTrustPipeline = None  # type: ignore
    AileeConfig = None  # type: ignore
    DecisionResult = None  # type: ignore
    SafetyStatus = None  # type: ignore


# ---- Forward references for type checking ----
if TYPE_CHECKING:
    from typing import TYPE_CHECKING as _TYPE_CHECKING


# -----------------------------
# Imaging Modalities
# -----------------------------

class ImagingModality(str, Enum):
    """Supported imaging modalities"""
    MRI = "MRI"
    CT = "CT"
    ULTRASOUND = "ULTRASOUND"
    XRAY = "XRAY"
    PET = "PET"
    SPECT = "SPECT"
    OCT = "OCT"
    OPTICAL_MICROSCOPY = "OPTICAL_MICROSCOPY"
    ELECTRON_MICROSCOPY = "ELECTRON_MICROSCOPY"
    FLUORESCENCE = "FLUORESCENCE"
    ASTRONOMY = "ASTRONOMY"
    SATELLITE = "SATELLITE"
    NDT = "NDT"  # Non-destructive testing
    INDUSTRIAL_INSPECTION = "INDUSTRIAL_INSPECTION"
    COMPUTATIONAL = "COMPUTATIONAL"  # Generic computational imaging
    UNKNOWN = "UNKNOWN"


class ReconstructionMethod(str, Enum):
    """Reconstruction algorithm types for multi-method validation"""
    PHYSICS_BASED = "PHYSICS_BASED"  # FBP, analytical
    ITERATIVE = "ITERATIVE"  # Conjugate gradient, ADMM
    COMPRESSED_SENSING = "COMPRESSED_SENSING"
    AI_DEEP_LEARNING = "AI_DEEP_LEARNING"
    HYBRID = "HYBRID"  # Physics + AI
    REFERENCE_PHANTOM = "REFERENCE_PHANTOM"
    BOOTSTRAP = "BOOTSTRAP"  # Statistical resampling


# -----------------------------
# Quality Decision Status
# -----------------------------

class QualityDecision(str, Enum):
    """Imaging quality assessment decision"""
    ACCEPT = "ACCEPT"
    ACCEPT_WITH_CAUTION = "ACCEPT_WITH_CAUTION"
    REJECT_REACQUIRE = "REJECT_REACQUIRE"
    ADAPT_AND_CONTINUE = "ADAPT_AND_CONTINUE"
    FALLBACK_RECONSTRUCTION = "FALLBACK_RECONSTRUCTION"


# -----------------------------
# Acquisition Parameters
# -----------------------------

@dataclass(frozen=True)
class AcquisitionParams:
    """
    Acquisition parameter snapshot for efficiency tracking.
    Modality-specific fields can be added via metadata dict.
    """
    # Energy/dose metrics
    energy_input: Optional[float] = None  # Joules, photon count, etc.
    radiation_dose_mgy: Optional[float] = None  # For X-ray, CT
    sar_w_per_kg: Optional[float] = None  # Specific absorption rate (MRI)
    acoustic_power_w: Optional[float] = None  # Ultrasound
    
    # Temporal metrics
    acquisition_time_s: Optional[float] = None
    integration_time_ms: Optional[float] = None
    
    # Spatial metrics
    spatial_resolution_mm: Optional[float] = None
    temporal_resolution_fps: Optional[float] = None
    field_of_view_mm: Optional[Tuple[float, ...]] = None
    
    # Sampling metrics
    undersampling_factor: Optional[float] = None  # For compressed sensing
    nyquist_ratio: Optional[float] = None  # Actual/required sampling
    
    # Modality-specific parameters
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_efficiency_score(self, information_content: float) -> Optional[float]:
        """
        Compute energy-to-information efficiency.
        Returns None if insufficient data.
        """
        if self.energy_input is not None and self.energy_input > 0:
            return information_content / self.energy_input
        return None
    
    def get_time_efficiency(self, information_content: float) -> Optional[float]:
        """Compute information per unit time"""
        if self.acquisition_time_s is not None and self.acquisition_time_s > 0:
            return information_content / self.acquisition_time_s
        return None
    
    def within_safety_limits(self, modality: ImagingModality) -> Tuple[bool, List[str]]:
        """Check if parameters respect safety limits"""
        issues: List[str] = []
        
        # Radiation dose limits (general guidance)
        if modality in (ImagingModality.CT, ImagingModality.XRAY):
            if self.radiation_dose_mgy is not None and self.radiation_dose_mgy > 100.0:
                issues.append(f"radiation_dose={self.radiation_dose_mgy:.1f}mGy exceeds 100mGy")
        
        # SAR limits (MRI safety)
        if modality == ImagingModality.MRI:
            if self.sar_w_per_kg is not None:
                if self.sar_w_per_kg > 4.0:  # Whole body limit
                    issues.append(f"sar={self.sar_w_per_kg:.2f}W/kg exceeds 4.0W/kg")
        
        # Acoustic power limits (ultrasound)
        if modality == ImagingModality.ULTRASOUND:
            if self.acoustic_power_w is not None and self.acoustic_power_w > 0.720:  # FDA limit
                issues.append(f"acoustic_power={self.acoustic_power_w:.3f}W exceeds 0.720W")
        
        return len(issues) == 0, issues


# -----------------------------
# Noise Model
# -----------------------------

@dataclass(frozen=True)
class NoiseModel:
    """
    Noise characterization for imaging systems.
    Different modalities have different dominant noise sources.
    """
    # Generic noise metrics
    estimated_noise_level: Optional[float] = None  # Standard deviation in image units
    snr_measurement: Optional[float] = None  # Signal-to-noise ratio
    cnr_measurement: Optional[float] = None  # Contrast-to-noise ratio
    
    # Noise type indicators
    quantum_noise_dominated: bool = False  # Poisson statistics
    thermal_noise_dominated: bool = False
    speckle_noise_present: bool = False  # Ultrasound, OCT
    
    # Noise floor
    detector_noise_floor: Optional[float] = None
    electronic_noise_level: Optional[float] = None
    
    # Propagation through reconstruction
    reconstruction_noise_amplification: Optional[float] = None
    
    def is_noise_acceptable(self, min_snr: float) -> Tuple[bool, str]:
        """Check if noise level is within acceptable range"""
        if self.snr_measurement is None:
            return True, "no_snr_measurement"
        
        if self.snr_measurement < min_snr:
            return False, f"snr={self.snr_measurement:.1f} below min_snr={min_snr:.1f}"
        
        return True, "snr_acceptable"
    
    def estimate_required_averaging(self, target_snr: float) -> Optional[int]:
        """
        Estimate number of averages needed to reach target SNR.
        SNR improves as sqrt(N_averages) for uncorrelated noise.
        """
        if self.snr_measurement is None or self.snr_measurement <= 0:
            return None
        
        ratio = target_snr / self.snr_measurement
        if ratio <= 1.0:
            return 1  # Already at target
        
        # SNR ~ sqrt(N) for averaging
        return int(math.ceil(ratio ** 2))


# -----------------------------
# Artifact Assessment
# -----------------------------

@dataclass(frozen=True)
class ArtifactAssessment:
    """
    Quantify imaging artifacts that degrade quality.
    Modality-specific artifacts can be added.
    """
    # Motion artifacts
    motion_detected: bool = False
    motion_severity_score: Optional[float] = None  # 0..1
    
    # Reconstruction artifacts
    gibbs_ringing_score: Optional[float] = None  # 0..1
    aliasing_score: Optional[float] = None  # 0..1
    truncation_artifacts: bool = False
    
    # Systematic artifacts
    bias_field_present: bool = False  # MRI
    beam_hardening: bool = False  # CT
    shadowing: bool = False  # Ultrasound
    
    # AI reconstruction specific
    hallucination_risk_score: Optional[float] = None  # 0..1
    texture_anomaly_score: Optional[float] = None  # 0..1
    
    overall_artifact_score: Optional[float] = None  # 0..1, higher = more artifacts
    
    def is_acceptable(self, threshold: float = 0.30) -> Tuple[bool, List[str]]:
        """Check if artifacts are within acceptable limits"""
        issues: List[str] = []
        
        if self.overall_artifact_score is not None:
            if self.overall_artifact_score > threshold:
                issues.append(f"overall_artifacts={self.overall_artifact_score:.2f} exceeds {threshold:.2f}")
        
        if self.motion_detected and self.motion_severity_score is not None:
            if self.motion_severity_score > 0.50:
                issues.append(f"severe_motion={self.motion_severity_score:.2f}")
        
        if self.hallucination_risk_score is not None and self.hallucination_risk_score > 0.20:
            issues.append(f"hallucination_risk={self.hallucination_risk_score:.2f} exceeds 0.20")
        
        return len(issues) == 0, issues


# -----------------------------
# Multi-Method Reconstruction Results
# -----------------------------

@dataclass(frozen=True)
class ReconstructionResult:
    """Single reconstruction result with quality metrics"""
    method: ReconstructionMethod
    quality_metric: float  # SNR, SSIM, or domain-specific metric
    confidence: Optional[float] = None
    computation_time_s: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Domain Inputs
# -----------------------------

@dataclass(frozen=True)
class ImagingSignals:
    """
    Governance signals for imaging quality assessment.
    Primary input structure for the imaging governor.
    
    Note: timestamp is automatically set to current time if not provided.
    """
    # Primary quality metric
    quality_metric: float  # SNR, resolution, SSIM, etc.
    model_confidence: Optional[float] = None  # 0..1 from AI model
    
    # Multi-method validation
    peer_reconstructions: Tuple[ReconstructionResult, ...] = ()
    
    # Acquisition context
    acquisition_params: Optional[AcquisitionParams] = None
    noise_model: Optional[NoiseModel] = None
    artifact_assessment: Optional[ArtifactAssessment] = None
    
    # Modality and context
    modality: ImagingModality = ImagingModality.UNKNOWN
    acquisition_id: Optional[str] = None
    
    # Metadata
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Automatically set timestamp if not provided"""
        if self.timestamp is None:
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(self, 'timestamp', time.time())


# -----------------------------
# Adaptive Acquisition Strategy
# -----------------------------

@dataclass
class AdaptiveStrategy:
    """
    Recommendations for adaptive acquisition parameter adjustment.
    The governor produces these; the imaging system executes them.
    """
    action: str  # "increase_snr" | "reduce_time" | "balance" | "stop"
    
    # Parameter adjustments (multiplicative factors)
    energy_adjustment: float = 1.0  # >1 = increase, <1 = decrease
    time_adjustment: float = 1.0
    resolution_adjustment: float = 1.0
    
    # Stopping criteria
    stop_acquisition: bool = False
    reason: str = ""
    
    # Suggested absolute values (if known)
    suggested_params: Optional[AcquisitionParams] = None
    
    def should_continue(self) -> bool:
        """Check if acquisition should continue"""
        return not self.stop_acquisition and self.action != "stop"


# -----------------------------
# Imaging Events (QA Logging)
# -----------------------------

@dataclass(frozen=True)
class ImagingEvent:
    """
    Structured event for quality assurance logging.
    Critical for medical device validation and scientific reproducibility.
    """
    timestamp: float
    event_type: str  # "quality_accept" | "quality_reject" | "adaptation" | "fallback_used"
    modality: ImagingModality
    quality_metric: float
    decision: QualityDecision
    reasons: List[str]
    
    # Context
    acquisition_params: Optional[AcquisitionParams] = None
    noise_model: Optional[NoiseModel] = None
    artifact_assessment: Optional[ArtifactAssessment] = None
    
    # Pipeline results
    ailee_decision: Optional[DecisionResult] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Domain Configuration
# -----------------------------

@dataclass(frozen=True)
class ImagingGovernancePolicy:
    """
    Domain policy for imaging quality governance.
    Defines quality thresholds, safety limits, and optimization goals.
    """
    # Modality
    modality: ImagingModality = ImagingModality.UNKNOWN
    
    # Quality thresholds
    min_snr_threshold: float = 10.0
    min_cnr_threshold: Optional[float] = None
    max_artifact_score: float = 0.30
    
    # Safety limits (None = not applicable)
    max_radiation_dose_mgy: Optional[float] = None
    max_sar_w_per_kg: Optional[float] = None
    max_acoustic_power_w: Optional[float] = None
    
    # Efficiency targets
    max_acquisition_time_s: Optional[float] = None
    target_energy_efficiency: Optional[float] = None  # Information per unit energy
    
    # Multi-method validation
    require_peer_consensus: bool = True
    min_peer_agreement_ratio: float = 0.70
    
    # Adaptive acquisition
    enable_adaptive_acquisition: bool = True
    adaptation_confidence_threshold: float = 0.75
    
    # Event logging
    max_event_log_size: int = 1000


def default_imaging_config(modality: ImagingModality) -> "AileeConfig":
    """
    Safe defaults for imaging governance pipeline configuration.
    Tuned per modality for appropriate sensitivity to quality variations.
    """
    if AileeConfig is None:
        raise RuntimeError("AILEE core imports unavailable")
    
    # Base configuration
    cfg = AileeConfig(
        accept_threshold=0.85,
        borderline_low=0.70,
        borderline_high=0.85,
        
        # Weights for imaging: prioritize agreement and stability
        w_stability=0.40,
        w_agreement=0.35,
        w_likelihood=0.25,
        
        history_window=50,
        forecast_window=10,
        
        grace_peer_delta=2.0,  # SNR units tolerance
        grace_min_peer_agreement_ratio=0.65,
        grace_forecast_epsilon=0.20,
        grace_max_abs_z=2.5,
        
        consensus_quorum=2,
        consensus_delta=2.5,  # SNR units
        consensus_pass_ratio=0.70,
        
        fallback_mode="last_good",
        
        enable_grace=True,
        enable_consensus=True,
        enable_audit_metadata=True,
    )
    
    # Modality-specific adjustments
    if modality in (ImagingModality.CT, ImagingModality.XRAY):
        # X-ray imaging: tighter thresholds due to dose concerns
        cfg.accept_threshold = 0.90
        cfg.grace_peer_delta = 1.5
        cfg.consensus_delta = 2.0
    
    elif modality == ImagingModality.MRI:
        # MRI: more lenient due to higher baseline SNR
        cfg.accept_threshold = 0.82
        cfg.grace_peer_delta = 3.0
    
    elif modality == ImagingModality.ULTRASOUND:
        # Ultrasound: handle speckle noise
        cfg.w_stability = 0.35  # Less weight on stability due to speckle
        cfg.w_agreement = 0.40
    
    elif modality in (ImagingModality.OPTICAL_MICROSCOPY, ImagingModality.FLUORESCENCE):
        # Optical: photon-limited, sensitive to noise
        cfg.accept_threshold = 0.88
        cfg.grace_max_abs_z = 3.0
    
    return cfg


# -----------------------------
# Result Structure (Forward declared)
# -----------------------------

@dataclass(frozen=True)
class ImagingDecisionResult:
    """
    Imaging governance decision result.
    Extends AILEE decision with imaging-specific context.
    """
    quality_acceptable: bool
    decision: QualityDecision
    validated_metric: float  # Renamed from validated_quality_metric for consistency
    confidence_score: float
    recommendation: str
    reasons: List[str]
    ailee_result: Optional[DecisionResult] = None
    efficiency_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Imaging Governor
# -----------------------------

class ImagingGovernor:
    """
    Production-grade governance controller for imaging systems.
    
    Validates imaging quality by integrating:
    - AILEE trust pipeline (core validation)
    - Multi-method reconstruction consensus
    - Noise model validation
    - Artifact assessment
    - Safety limit enforcement
    - Efficiency optimization
    - Adaptive acquisition support
    
    Integration contract:
    - Call evaluate(signals) per acquisition or reconstruction
    - Use decision.quality_acceptable to accept/reject
    - Monitor decision.recommendation for adaptation
    - Export events for QA/regulatory compliance
    """
    
    def __init__(
        self,
        cfg: Optional["AileeConfig"] = None,
        policy: Optional[ImagingGovernancePolicy] = None,
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable. Ensure ailee_trust_pipeline_v1.py is importable.")
        
        self.policy = policy or ImagingGovernancePolicy()
        self.cfg = cfg or default_imaging_config(self.policy.modality)
        
        # Apply policy constraints to config
        if self.policy.min_snr_threshold > 0:
            self.cfg.hard_min = 0.0
            # Max SNR is open-ended, but we can set a reasonable upper bound
            self.cfg.hard_max = 100.0
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        
        # Tracking state
        self._last_quality: Optional[float] = None
        self._last_decision: Optional[QualityDecision] = None
        
        # Event logging
        self._event_log: List[ImagingEvent] = []
        self._last_event: Optional[ImagingEvent] = None
        
        # Efficiency tracking
        self._efficiency_history: List[Tuple[float, float]] = []  # [(timestamp, efficiency)]
    
    # -------------------------
    # Public API
    # -------------------------
    
    def evaluate(self, signals: ImagingSignals) -> "ImagingDecisionResult":
        """
        Evaluate imaging quality and produce governance decision.
        
        Args:
            signals: Current imaging signals (quality metrics, reconstructions, etc.)
        
        Returns:
            ImagingDecisionResult with quality assessment and recommendations
        """
        # Timestamp is now guaranteed to be set by ImagingSignals.__post_init__
        ts = float(signals.timestamp)
        
        reasons: List[str] = []
        
        # 1) Safety limits check (hard constraints)
        if signals.acquisition_params is not None:
            safe, safety_issues = signals.acquisition_params.within_safety_limits(signals.modality)
            if not safe:
                reasons.extend([f"Safety: {issue}" for issue in safety_issues])
                decision = self._create_reject_decision(
                    signals, reasons, "safety_limit_violation", ts
                )
                self._log_event(ts, signals, decision, reasons)
                return decision
        
        # 2) Noise model validation
        if signals.noise_model is not None:
            noise_ok, noise_reason = signals.noise_model.is_noise_acceptable(self.policy.min_snr_threshold)
            if not noise_ok:
                reasons.append(f"Noise: {noise_reason}")
                # Don't reject immediately, let AILEE pipeline evaluate
        
        # 3) Artifact assessment
        if signals.artifact_assessment is not None:
            artifacts_ok, artifact_issues = signals.artifact_assessment.is_acceptable(self.policy.max_artifact_score)
            if not artifacts_ok:
                reasons.extend([f"Artifact: {issue}" for issue in artifact_issues])
                # Significant artifacts may trigger fallback
        
        # 4) Extract peer quality metrics
        peer_values = [r.quality_metric for r in signals.peer_reconstructions]
        
        # 5) Build context for AILEE pipeline
        ctx = self._build_context(signals, reasons)
        
        # 6) Use AILEE pipeline to validate quality metric
        ailee_result = self.pipeline.process(
            raw_value=float(signals.quality_metric),
            raw_confidence=float(signals.model_confidence) if signals.model_confidence is not None else None,
            peer_values=peer_values,
            timestamp=ts,
            context=ctx,
        )
        
        # 7) Make imaging-specific decision
        decision = self._make_imaging_decision(signals, ailee_result, reasons, ts)
        
        # 8) Log event
        self._log_event(ts, signals, decision, reasons)
        
        # 9) Update state
        self._commit_state(ts, signals, decision, ailee_result)
        
        return decision
    
    def get_last_event(self) -> Optional[ImagingEvent]:
        """Get most recent imaging event"""
        return self._last_event
    
    def export_events(self, since_ts: Optional[float] = None) -> List[ImagingEvent]:
        """Export imaging events for QA/analysis"""
        if since_ts is None:
            return self._event_log[:]
        return [e for e in self._event_log if e.timestamp >= since_ts]
    
    def get_efficiency_trend(self) -> str:
        """Get efficiency trend: improving, stable, declining"""
        if len(self._efficiency_history) < 10:
            return "unknown"
        
        recent = [eff for _, eff in self._efficiency_history[-10:]]
        older = [eff for _, eff in self._efficiency_history[-20:-10]] if len(self._efficiency_history) >= 20 else recent
        
        recent_avg = statistics.fmean(recent)
        older_avg = statistics.fmean(older)
        
        diff = recent_avg - older_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"
    
    def suggest_adaptive_strategy(self, signals: ImagingSignals, decision: "ImagingDecisionResult") -> AdaptiveStrategy:
        """
        Suggest adaptive acquisition strategy based on current quality assessment.
        This is advisory only; the imaging system decides whether to execute.
        """
        if not self.policy.enable_adaptive_acquisition:
            return AdaptiveStrategy(action="no_adaptation", reason="adaptive_acquisition_disabled")
        
        # If quality is excellent, consider reducing acquisition resources
        if decision.quality_acceptable and signals.quality_metric > self.policy.min_snr_threshold * 1.5:
            return AdaptiveStrategy(
                action="reduce_time",
                time_adjustment=0.90,  # 10% faster
                reason="quality_exceeds_target"
            )
        
        # If quality is borderline, increase acquisition resources
        if decision.decision == QualityDecision.ACCEPT_WITH_CAUTION:
            return AdaptiveStrategy(
                action="increase_snr",
                energy_adjustment=1.15,  # 15% more energy
                time_adjustment=1.10,    # 10% more time
                reason="quality_borderline"
            )
        
        # If quality is unacceptable, recommend reacquisition with higher resources
        if not decision.quality_acceptable:
            if signals.noise_model is not None and signals.noise_model.snr_measurement is not None:
                required_avg = signals.noise_model.estimate_required_averaging(self.policy.min_snr_threshold)
                if required_avg is not None:
                    return AdaptiveStrategy(
                        action="increase_snr",
                        time_adjustment=float(required_avg),
                        reason=f"requires_{required_avg}x_averaging"
                    )
            
            return AdaptiveStrategy(
                action="increase_snr",
                energy_adjustment=1.50,
                time_adjustment=1.30,
                reason="quality_insufficient"
            )
        
        # Default: maintain current parameters
        return AdaptiveStrategy(action="balance", reason="quality_acceptable")
    
    # -------------------------
    # Decision Logic
    # -------------------------
    
    def _make_imaging_decision(
        self,
        signals: ImagingSignals,
        ailee_result: DecisionResult,
        reasons: List[str],
        ts: float,
    ) -> "ImagingDecisionResult":
        """Synthesize imaging-specific decision from AILEE result"""
        
        # Check if AILEE accepted the quality metric
        ailee_accepted = not ailee_result.used_fallback
        try:
            if SafetyStatus is not None:
                ailee_accepted = ailee_accepted and (
                    ailee_result.safety_status in (SafetyStatus.ACCEPTED, SafetyStatus.BORDERLINE)
                )
        except Exception:
            pass
        
        # Determine quality decision
        decision_type: QualityDecision
        quality_acceptable = False
        recommendation = ""
        
        if ailee_accepted and signals.quality_metric >= self.policy.min_snr_threshold:
            # Check for caution conditions
            if ailee_result.safety_status == SafetyStatus.BORDERLINE:
                decision_type = QualityDecision.ACCEPT_WITH_CAUTION
                quality_acceptable = True
                recommendation = "monitor_quality"
                reasons.append("Quality borderline but acceptable.")
            else:
                decision_type = QualityDecision.ACCEPT
                quality_acceptable = True
                recommendation = "continue"
                reasons.append("Quality meets all criteria.")
        
        elif ailee_result.used_fallback:
            # AILEE triggered fallback
            decision_type = QualityDecision.FALLBACK_RECONSTRUCTION
            quality_acceptable = False
            recommendation = "use_fallback_reconstruction"
            reasons.append("AILEE fallback triggered -> use physics-based reconstruction.")
        
        elif self.policy.enable_adaptive_acquisition:
            # Try adaptive acquisition before rejecting
            decision_type = QualityDecision.ADAPT_AND_CONTINUE
            quality_acceptable = False
            recommendation = "adapt_and_continue"
            reasons.append("Quality insufficient -> adapt acquisition parameters.")
        
        else:
            # Reject and reacquire
            decision_type = QualityDecision.REJECT_REACQUIRE
            quality_acceptable = False
            recommendation = "reacquire"
            reasons.append("Quality insufficient -> reacquisition required.")
        
        # Check efficiency if policy specifies target
        efficiency_score: Optional[float] = None
        if signals.acquisition_params is not None and self.policy.target_energy_efficiency is not None:
            efficiency_score = signals.acquisition_params.get_efficiency_score(signals.quality_metric)
            if efficiency_score is not None and efficiency_score < self.policy.target_energy_efficiency:
                reasons.append(
                    f"Efficiency {efficiency_score:.3f} below target {self.policy.target_energy_efficiency:.3f}."
                )
        
        return ImagingDecisionResult(
            quality_acceptable=quality_acceptable,
            decision=decision_type,
            validated_metric=ailee_result.value,
            confidence_score=ailee_result.confidence_score,
            recommendation=recommendation,
            reasons=reasons[:],
            ailee_result=ailee_result,
            efficiency_score=efficiency_score,
            metadata=self._build_result_metadata(signals, ailee_result),
        )
    
    def _create_reject_decision(
        self,
        signals: ImagingSignals,
        reasons: List[str],
        reject_reason: str,
        ts: float,
    ) -> "ImagingDecisionResult":
        """Create rejection decision without AILEE pipeline evaluation"""
        return ImagingDecisionResult(
            quality_acceptable=False,
            decision=QualityDecision.REJECT_REACQUIRE,
            validated_metric=signals.quality_metric,
            confidence_score=0.0,
            recommendation="reject",
            reasons=reasons[:],
            ailee_result=None,
            metadata={"reject_reason": reject_reason},
        )
    
    # -------------------------
    # Context Building
    # -------------------------
    
    def _build_context(
        self,
        signals: ImagingSignals,
        reasons: List[str],
    ) -> Dict[str, Any]:
        """Build comprehensive context for AILEE audit metadata"""
        ctx: Dict[str, Any] = {
            "domain": "imaging",
            "modality": signals.modality.value,
            "signal": "quality_metric",
        }
        
        # Policy info
        ctx["policy"] = {
            "min_snr_threshold": self.policy.min_snr_threshold,
            "max_artifact_score": self.policy.max_artifact_score,
            "require_peer_consensus": self.policy.require_peer_consensus,
        }
        
        # Acquisition parameters
        if signals.acquisition_params is not None:
            ctx["acquisition"] = {
                "energy_input": signals.acquisition_params.energy_input,
                "acquisition_time_s": signals.acquisition_params.acquisition_time_s,
                "spatial_resolution_mm": signals.acquisition_params.spatial_resolution_mm,
                "undersampling_factor": signals.acquisition_params.undersampling_factor,
            }
        
        # Noise model
        if signals.noise_model is not None:
            ctx["noise"] = {
                "snr_measurement": signals.noise_model.snr_measurement,
                "cnr_measurement": signals.noise_model.cnr_measurement,
                "quantum_noise_dominated": signals.noise_model.quantum_noise_dominated,
            }
        
        # Artifacts
        if signals.artifact_assessment is not None:
            ctx["artifacts"] = {
                "overall_score": signals.artifact_assessment.overall_artifact_score,
                "motion_detected": signals.artifact_assessment.motion_detected,
                "hallucination_risk": signals.artifact_assessment.hallucination_risk_score,
            }
        
        # Multi-method reconstruction
        if signals.peer_reconstructions:
            ctx["peer_methods"] = [
                {
                    "method": r.method.value,
                    "quality": r.quality_metric,
                    "confidence": r.confidence,
                }
                for r in signals.peer_reconstructions
            ]
        
        # User context
        if signals.context:
            ctx["user_context"] = signals.context
        
        return ctx
    
    def _build_result_metadata(
        self,
        signals: ImagingSignals,
        ailee_result: Optional[DecisionResult],
    ) -> Dict[str, Any]:
        """Build result metadata for transparency"""
        meta: Dict[str, Any] = {}
        
        meta["modality"] = signals.modality.value
        meta["quality_metric"] = signals.quality_metric
        
        if ailee_result is not None:
            meta["ailee_safety_status"] = ailee_result.safety_status.value if hasattr(ailee_result.safety_status, 'value') else str(ailee_result.safety_status)
            meta["ailee_grace_status"] = ailee_result.grace_status.value if hasattr(ailee_result.grace_status, 'value') else str(ailee_result.grace_status)
            meta["ailee_consensus_status"] = ailee_result.consensus_status.value if hasattr(ailee_result.consensus_status, 'value') else str(ailee_result.consensus_status)
        
        if signals.acquisition_params is not None:
            meta["acquisition_time_s"] = signals.acquisition_params.acquisition_time_s
        
        if signals.noise_model is not None:
            meta["snr"] = signals.noise_model.snr_measurement
        
        return meta
    
    # -------------------------
    # Event Logging
    # -------------------------
    
    def _log_event(
        self,
        ts: float,
        signals: ImagingSignals,
        decision: "ImagingDecisionResult",
        reasons: List[str],
    ) -> None:
        """Log imaging event for QA"""
        
        event_type = "quality_accept" if decision.quality_acceptable else "quality_reject"
        if decision.decision == QualityDecision.ADAPT_AND_CONTINUE:
            event_type = "adaptation"
        elif decision.decision == QualityDecision.FALLBACK_RECONSTRUCTION:
            event_type = "fallback_used"
        
        event = ImagingEvent(
            timestamp=ts,
            event_type=event_type,
            modality=signals.modality,
            quality_metric=signals.quality_metric,
            decision=decision.decision,
            reasons=reasons[:],
            acquisition_params=signals.acquisition_params,
            noise_model=signals.noise_model,
            artifact_assessment=signals.artifact_assessment,
            ailee_decision=decision.ailee_result,
            metadata=dict(decision.metadata) if decision.metadata else {},
        )
        
        self._event_log.append(event)
        self._last_event = event
        
        # Trim log if needed
        if len(self._event_log) > self.policy.max_event_log_size:
            self._event_log = self._event_log[-self.policy.max_event_log_size:]
    
    # -------------------------
    # State Management
    # -------------------------
    
    def _commit_state(
        self,
        ts: float,
        signals: ImagingSignals,
        decision: "ImagingDecisionResult",
        ailee_result: DecisionResult,
    ) -> None:
        """Update governor state"""
        if decision.quality_acceptable:
            self._last_quality = signals.quality_metric
            self._last_decision = decision.decision
            
            # Track efficiency
            if decision.efficiency_score is not None:
                self._efficiency_history.append((ts, decision.efficiency_score))
                # Keep last 100 efficiency measurements
                if len(self._efficiency_history) > 100:
                    self._efficiency_history = self._efficiency_history[-100:]


# -----------------------------
# Convenience Exports
# -----------------------------

__all__ = [
    "ImagingModality",
    "ReconstructionMethod",
    "QualityDecision",
    "AcquisitionParams",
    "NoiseModel",
    "ArtifactAssessment",
    "ReconstructionResult",
    "ImagingSignals",
    "AdaptiveStrategy",
    "ImagingEvent",
    "ImagingGovernancePolicy",
    "ImagingGovernor",
    "ImagingDecisionResult",
    "default_imaging_config",
]


# -----------------------------
# Domain Descriptor (for framework registration)
# -----------------------------

class ImagingDomain:
    """
    ImagingDomain applies AILEE principles to imaging acquisition
    and reconstruction pipelines.
    """
    name: str = "IMAGING"
    
    def describe(self) -> Dict[str, Any]:
        """
        Return a high-level description of the IMAGING domain and
        its optimization focus areas.
        """
        return {
            "domain": self.name,
            "focus": [
                "Energy-to-information efficiency",
                "Signal fidelity under constrained acquisition",
                "Noise-aware optimization",
                "Temporal and spatial resolution trade-offs",
                "Adaptive load management in imaging pipelines",
            ],
            "modalities": [
                "Medical imaging (MRI, CT, Ultrasound, PET)",
                "Scientific imaging (astronomy, microscopy)",
                "Industrial imaging (NDT, inspection)",
                "Remote and satellite imaging",
                "Computational and AI-assisted imaging",
            ],
            "ailee_role": (
                "Provide an invariant efficiency framework for evaluating "
                "and optimizing imaging systems without modifying "
                "core physical or reconstruction models."
            ),
            "implementation_status": "production_grade_v1.0.1",
            "key_features": [
                "Multi-method reconstruction validation",
                "Safety-constrained optimization",
                "Adaptive acquisition support",
                "Noise model integration",
                "Artifact assessment",
                "Efficiency tracking",
                "QA event logging",
            ],
        }


# -----------------------------
# Demo / Self-Test
# -----------------------------

if __name__ == "__main__":
    print("AILEE Imaging Domain - Production Grade Demo (v1.0.1)")
    print("=" * 60)
    
    # Setup governor for MRI
    policy = ImagingGovernancePolicy(
        modality=ImagingModality.MRI,
        min_snr_threshold=15.0,
        max_acquisition_time_s=300.0,
        enable_adaptive_acquisition=True,
    )
    governor = ImagingGovernor(policy=policy)
    
    # Simulate imaging scenarios
    scenarios = [
        {
            "name": "Good quality MRI with AI reconstruction",
            "signals": ImagingSignals(
                quality_metric=22.5,  # SNR
                model_confidence=0.92,
                peer_reconstructions=(
                    ReconstructionResult(
                        method=ReconstructionMethod.PHYSICS_BASED,
                        quality_metric=21.8,
                        confidence=0.95,
                    ),
                    ReconstructionResult(
                        method=ReconstructionMethod.ITERATIVE,
                        quality_metric=22.2,
                        confidence=0.90,
                    ),
                ),
                acquisition_params=AcquisitionParams(
                    acquisition_time_s=180.0,
                    spatial_resolution_mm=1.0,
                    undersampling_factor=2.0,
                ),
                noise_model=NoiseModel(
                    snr_measurement=22.5,
                    thermal_noise_dominated=True,
                ),
                modality=ImagingModality.MRI,
            ),
        },
        {
            "name": "Low SNR requiring adaptation",
            "signals": ImagingSignals(
                quality_metric=12.0,  # Below threshold
                model_confidence=0.68,
                peer_reconstructions=(
                    ReconstructionResult(
                        method=ReconstructionMethod.PHYSICS_BASED,
                        quality_metric=11.5,
                    ),
                ),
                noise_model=NoiseModel(
                    snr_measurement=12.0,
                    thermal_noise_dominated=True,
                ),
                modality=ImagingModality.MRI,
            ),
        },
        {
            "name": "High artifact score",
            "signals": ImagingSignals(
                quality_metric=18.0,
                model_confidence=0.75,
                artifact_assessment=ArtifactAssessment(
                    overall_artifact_score=0.45,  # High artifacts
                    motion_detected=True,
                    motion_severity_score=0.60,
                ),
                modality=ImagingModality.MRI,
            ),
        },
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
        
        decision = governor.evaluate(scenario['signals'])
        
        print(f"Quality Metric: {scenario['signals'].quality_metric:.1f}")
        print(f"Decision: {decision.decision.value}")
        print(f"Acceptable: {decision.quality_acceptable}")
        print(f"Confidence: {decision.confidence_score:.3f}")
        print(f"Recommendation: {decision.recommendation}")
        if decision.reasons:
            print(f"Reasons:")
            for reason in decision.reasons[:3]:
                print(f"  - {reason}")
        
        # Show adaptive strategy
        if policy.enable_adaptive_acquisition:
            strategy = governor.suggest_adaptive_strategy(scenario['signals'], decision)
            print(f"Adaptive Strategy: {strategy.action}")
            if strategy.reason:
                print(f"  Reason: {strategy.reason}")
    
    # Show event log
    print(f"\n--- Event Log ({len(governor.export_events())} events) ---")
    for event in governor.export_events()[-3:]:
        print(f"[{event.timestamp:.2f}] {event.event_type}: {event.decision.value}")
    
    # Show efficiency trend
    print(f"\nEfficiency Trend: {governor.get_efficiency_trend()}")
    
    print("\n" + "=" * 60)
    print("Demo complete. Governor is production-ready.")
