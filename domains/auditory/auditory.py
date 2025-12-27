if decision.warning:
            lines.append("⚠️  WARNING:")
            lines.append("-" * 70)
            lines.append(f"→ {decision.warning}")
            lines.append("")
        
        lines.append("RECOMMENDED ACTION:")
        lines.append("-" * 70)
        lines.append(f"→ {decision.recommendation.replace('_', ' ').title()}")
        lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    # -------------------------
    # Public API - State Queries
    # -------------------------
    
    def get_last_event(self) -> Optional[AuditoryEvent]:
        """Get most recent auditory governance event"""
        return self._last_event
    
    def export_events(self, since_ts: Optional[float] = None) -> List[AuditoryEvent]:
        """Export events for medical device compliance reporting"""
        if since_ts is None:
            return self._event_log[:]
        return [e for e in self._event_log if e.timestamp >= since_ts]
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for clinical monitoring"""
        if not self._event_log:
            return {"status": "no_history"}
        
        total_events = len(self._event_log)
        
        # Count by authorization level
        level_counts = {}
        for event in self._event_log:
            level_name = event.authorized_level.name
            level_counts[level_name] = level_counts.get(level_name, 0) + 1
        
        # Count warnings
        warning_count = sum(1 for e in self._event_log if e.warning is not None)
        
        # Average output level
        output_levels = [e.output_db_cap for e in self._event_log]
        avg_output = statistics.mean(output_levels) if output_levels else 0.0
        
        # Average trust score
        trust_scores = [e.validated_trust_score for e in self._event_log]
        avg_trust = statistics.mean(trust_scores) if trust_scores else 0.0
        
        return {
            "total_events": total_events,
            "authorization_level_distribution": level_counts,
            "warning_count": warning_count,
            "average_output_db": avg_output,
            "average_trust_score": avg_trust,
            "last_event_timestamp": self._event_log[-1].timestamp if self._event_log else None,
        }
    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get safety summary for clinical review"""
        recent_events = self._event_log[-100:] if len(self._event_log) > 100 else self._event_log
        
        if not recent_events:
            return {"status": "no_recent_history"}
        
        # Check for concerning patterns
        high_output_count = sum(1 for e in recent_events if e.output_db_cap > 90.0)
        discomfort_events = sum(
            1 for e in recent_events 
            if e.comfort_metrics and e.comfort_metrics.discomfort_score and e.comfort_metrics.discomfort_score > 0.35
        )
        hardware_issues = sum(
            1 for e in recent_events
            if e.device_health and (e.device_health.feedback_detected or e.device_health.hardware_faults)
        )
        
        return {
            "recent_event_count": len(recent_events),
            "high_output_events": high_output_count,
            "discomfort_events": discomfort_events,
            "hardware_issue_events": hardware_issues,
            "safety_concerns": high_output_count > 10 or discomfort_events > 20 or hardware_issues > 5,
        }


# -----------------------------
# Convenience Functions
# -----------------------------

def create_auditory_governor(
    user_safety_profile: UserSafetyProfile = UserSafetyProfile.STANDARD,
    max_output_db_spl: float = 100.0,
    max_allowed_level: OutputAuthorizationLevel = OutputAuthorizationLevel.COMFORT_OPTIMIZED,
    **policy_overrides
) -> AuditoryGovernor:
    """
    Convenience factory for creating auditory governor with common configurations.
    
    Args:
        user_safety_profile: User safety category
        max_output_db_spl: Absolute maximum output in dB SPL
        max_allowed_level: Maximum authorization level allowed by policy
        **policy_overrides: Additional policy parameters
    
    Returns:
        Configured AuditoryGovernor instance
    
    Example:
        governor = create_auditory_governor(
            user_safety_profile=UserSafetyProfile.PEDIATRIC,
            max_output_db_spl=85.0,
            max_allowed_level=OutputAuthorizationLevel.COMFORT_OPTIMIZED,
            enable_automatic_volume_limiting=True,
        )
    """
    policy_kwargs = {
        "user_safety_profile": user_safety_profile,
        "max_output_db_spl": max_output_db_spl,
        "max_allowed_level": max_allowed_level,
    }
    
    # Profile-specific defaults
    if user_safety_profile == UserSafetyProfile.PEDIATRIC:
        policy_kwargs.update({
            "max_output_db_spl": min(max_output_db_spl, 85.0),
            "max_continuous_output_db": 75.0,
            "max_allowed_level": min(max_allowed_level, OutputAuthorizationLevel.COMFORT_OPTIMIZED),
            "enable_automatic_volume_limiting": True,
        })
    elif user_safety_profile == UserSafetyProfile.TINNITUS_RISK:
        policy_kwargs.update({
            "max_output_db_spl": min(max_output_db_spl, 90.0),
            "max_discomfort_score": 0.25,
            "enable_predictive_warnings": True,
        })
    elif user_safety_profile == UserSafetyProfile.PROFESSIONAL:
        policy_kwargs.update({
            "min_speech_intelligibility": 0.75,
            "min_noise_reduction": 0.65,
            "max_latency_ms": 15.0,
        })
    
    policy_kwargs.update(policy_overrides)
    policy = AuditoryGovernancePolicy(**policy_kwargs)
    
    cfg = default_auditory_config()
    
    return AuditoryGovernor(cfg=cfg, policy=policy)


def validate_auditory_signals(signals: AuditorySignals) -> Tuple[bool, List[str]]:
    """
    Pre-flight validation of auditory signals structure.
    
    Args:
        signals: AuditorySignals to validate
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues: List[str] = []
    
    # Check score range
    if not (0.0 <= signals.proposed_action_trust_score <= 1.0):
        issues.append(f"proposed_action_trust_score={signals.proposed_action_trust_score} outside [0.0, 1.0]")
    
    # Check peer enhancement scores
    for i, score in enumerate(signals.peer_enhancement_scores):
        if not (0.0 <= score <= 1.0):
            issues.append(f"peer_enhancement_scores[{i}]={score} outside [0.0, 1.0]")
    
    # Validate enhancement metrics if present
    if signals.enhancement:
        enhancement = signals.enhancement
        if enhancement.speech_intelligibility_score is not None:
            if not (0.0 <= enhancement.speech_intelligibility_score <= 1.0):
                issues.append(f"speech_intelligibility_score outside [0.0, 1.0]")
        
        if enhancement.noise_reduction_score is not None:
            if not (0.0 <= enhancement.noise_reduction_score <= 1.0):
                issues.append(f"noise_reduction_score outside [0.0, 1.0]")
        
        if enhancement.enhancement_latency_ms is not None:
            if enhancement.enhancement_latency_ms < 0:
                issues.append(f"enhancement_latency_ms cannot be negative")
    
    # Validate comfort metrics if present
    if signals.comfort:
        comfort = signals.comfort
        if comfort.discomfort_score is not None:
            if not (0.0 <= comfort.discomfort_score <= 1.0):
                issues.append(f"discomfort_score outside [0.0, 1.0]")
        
        if comfort.fatigue_risk_score is not None:
            if not (0.0 <= comfort.fatigue_risk_score <= 1.0):
                issues.append(f"fatigue_risk_score outside [0.0, 1.0]")
    
    # Validate device health if present
    if signals.device_health:
        device = signals.device_health
        if device.mic_health_score is not None:
            if not (0.0 <= device.mic_health_score <= 1.0):
                issues.append(f"mic_health_score outside [0.0, 1.0]")
        
        if device.battery_level is not None:
            if not (0.0 <= device.battery_level <= 1.0):
                issues.append(f"battery_level outside [0.0, 1.0]")
    
    # Validate hearing profile if present
    if signals.hearing_profile:
        profile = signals.hearing_profile
        if profile.max_safe_output_db < profile.preferred_output_db:
            issues.append("max_safe_output_db must be >= preferred_output_db")
        
        if profile.max_safe_output_db > 120.0:
            issues.append("max_safe_output_db exceeds safe limits (>120 dB)")
    
    return len(issues) == 0, issues


# -----------------------------
# Module Exports
# -----------------------------

__all__ = [
    # Enums
    "OutputAuthorizationLevel",
    "ListeningMode",
    "UserSafetyProfile",
    "DecisionOutcome",
    "RegulatoryGateResult",
    
    # Data structures
    "HearingProfile",
    "EnvironmentMetrics",
    "EnhancementMetrics",
    "ComfortMetrics",
    "DeviceHealth",
    "AuditoryUncertainty",
    "AuditoryDecisionDelta",
    "AuditorySignals",
    
    # Configuration
    "AuditoryGovernancePolicy",
    "AuditoryDecision",
    
    # Events
    "AuditoryEvent",
    
    # Governance components
    "AuditoryPolicyEvaluator",
    "AuditoryUncertaintyCalculator",
    "AuditoryGovernor",
    
    # Utilities
    "default_auditory_config",
    "create_auditory_governor",
    "validate_auditory_signals",
    
    # Constants
    "AUDITORY_FLAG_SEVERITY",
]    @staticmethod
    def compute_aggregate_uncertainty(
        signals: AuditorySignals
    ) -> AuditoryUncertainty:
        """Aggregate uncertainty from all auditory sources"""
        components = {}
        
        # 1. Enhancement uncertainty
        enhancement_unc = 0.0
        if signals.enhancement:
            if signals.enhancement.ai_confidence is not None:
                enhancement_unc = 1.0 - signals.enhancement.ai_confidence
            else:
                enhancement_unc = 0.3  # Default uncertainty
            
            # Increase uncertainty if artifacts detected
            if signals.enhancement.artifacts_detected:
                enhancement_unc = min(1.0, enhancement_unc + 0.2)
        else:
            enhancement_unc = 0.5  # No enhancement data
        
        components["enhancement"] = enhancement_unc
        
        # 2. Environmental uncertainty
        environmental_unc = 0.0
        if signals.environment:
            # High uncertainty in challenging environments
            if signals.environment.ambient_noise_db and signals.environment.ambient_noise_db > 75.0:
                environmental_unc += 0.2
            
            if signals.environment.snr_db and signals.environment.snr_db < 5.0:
                environmental_unc += 0.2
            
            if signals.environment.scene_confidence is not None:
                environmental_unc += (1.0 - signals.environment.scene_confidence) * 0.3
            else:
                environmental_unc += 0.2  # Unknown scene
        else:
            environmental_unc = 0.4  # No environmental data
        
        components["environmental"] = min(1.0, environmental_unc)
        
        # 3. Device uncertainty
        device_unc = 0.0
        if signals.device_health:
            device = signals.device_health
            
            if device.mic_health_score is not None:
                device_unc += (1.0 - device.mic_health_score) * 0.5
            
            if device.hardware_faults:
                device_unc += 0.3
            
            if not device.calibration_valid:
                device_unc += 0.2
        else:
            device_unc = 0.3  # No device health data
        
        components["device"] = min(1.0, device_unc)
        
        # 4. Comfort uncertainty
        comfort_unc = 0.0
        if signals.comfort:
            # Higher uncertainty if user showing discomfort but no clear measurements
            if signals.comfort.discomfort_score is None:
                comfort_unc = 0.25
            else:
                # Lower uncertainty when we have measurements
                comfort_unc = 0.1
        else:
            comfort_unc = 0.35  # No comfort data
        
        components["comfort"] = comfort_unc
        
        # Aggregate using weighted average
        weights = {
            "enhancement": 0.40,    # Most important for quality
            "environmental": 0.30,  # Context matters
            "device": 0.20,         # Hardware reliability
            "comfort": 0.10,        # User feedback
        }
        
        aggregate = sum(components[k] * weights[k] for k in components.keys())
        
        # Find dominant source
        dominant_source = max(components.items(), key=lambda x: x[1])[0]
        
        return AuditoryUncertainty(
            aggregate_uncertainty_score=aggregate,
            enhancement_uncertainty=components["enhancement"],
            environmental_uncertainty=components["environmental"],
            device_uncertainty=components["device"],
            comfort_uncertainty=components["comfort"],
            dominant_uncertainty_source=dominant_source,
            uncertainty_sources=components,
        )


# ===== AUDITORY GOVERNOR =====
#
# AUTHORITATIVE DECISION MAKER
#
# This is the ONLY component with decision authority.
# All other components (PolicyEvaluator, UncertaintyCalculator) provide
# advisory inputs. Only AuditoryGovernor.evaluate() makes final authorization.

class AuditoryGovernor:
    """
    Production-grade governance controller for auditory enhancement systems.
    
    AUTHORITATIVE ROLE: Makes final authorization decisions.
    
    Validates enhancement by integrating:
    - AILEE trust pipeline (core validation)
    - Policy evaluator (formalized gate checks) [ADVISORY]
    - Device health assessment [CONSTRAINT CHECK]
    - User comfort monitoring [CONSTRAINT CHECK]
    - Environmental context [CONSTRAINT CHECK]
    - Output safety constraints [COMPUTATION]
    - Uncertainty aggregation [COMPUTATION]
    
    Philosophy: This is a SAFETY system for medical/assistive devices.
    Default bias: Conservative output levels until quality proven.
    """
    
    def __init__(
        self,
        cfg: Optional[AileeConfig] = None,
        policy: Optional[AuditoryGovernancePolicy] = None,
    ):
        if AileeTrustPipeline is None or AileeConfig is None:
            raise RuntimeError("AILEE core imports unavailable")
        
        self.policy = policy or AuditoryGovernancePolicy()
        self.cfg = cfg or default_auditory_config()
        
        self.cfg.hard_min = 0.0
        self.cfg.hard_max = 1.0
        
        self.pipeline = AileeTrustPipeline(self.cfg)
        self.policy_evaluator = AuditoryPolicyEvaluator(self.policy)
        
        # State tracking for delta computation
        self._last_decision: Optional[OutputAuthorizationLevel] = None
        self._last_trust_score: Optional[float] = None
        self._last_output_db: Optional[float] = None
        self._last_comfort_score: Optional[float] = None
        self._last_enhancement_quality: Optional[float] = None
        self._last_uncertainty: Optional[float] = None
        self._last_decision_time: Optional[float] = None
        self._last_listening_mode: Optional[ListeningMode] = None
        
        # Event logging (medical device compliance)
        self._event_log: List[AuditoryEvent] = []
        self._last_event: Optional[AuditoryEvent] = None
    
    def evaluate(self, signals: AuditorySignals) -> AuditoryDecision:
        """
        Evaluate auditory enhancement proposal and produce governance decision.
        
        AUTHORITATIVE DECISION: This method has sole decision authority.
        All other components provide advisory inputs.
        
        Philosophy: Layer multiple checks, each can veto.
        Safety first, quality second, comfort third.
        
        Returns:
            AuditoryDecision with authorized level and constraints
        """
        ts = float(signals.timestamp)
        reasons: List[str] = []
        precautionary_flags: List[str] = []
        
        # 0) Compute aggregate uncertainty FIRST
        aggregate_unc = AuditoryUncertaintyCalculator.compute_aggregate_uncertainty(signals)
        
        # Check if uncertainty alone blocks action
        unc_ok, unc_reason = aggregate_unc.is_uncertainty_acceptable(max_aggregate=0.35)
        if not unc_ok:
            reasons.append(unc_reason)
            precautionary_flags.append("aggregate_uncertainty_excessive")
        
        # 1) Device health gate (highest priority - hardware safety)
        device_ok, device_issues, max_device_level = \
            self.policy_evaluator.check_device_health_gates(signals)
        
        if not device_ok:
            reasons.extend([f"Device: {issue}" for issue in device_issues])
            if max_device_level == OutputAuthorizationLevel.DIAGNOSTIC_ONLY:
                precautionary_flags.append("hardware_fault_critical")
            elif max_device_level == OutputAuthorizationLevel.SAFETY_LIMITED:
                precautionary_flags.append("hardware_degraded")
        
        # 2) Comfort gate (user safety)
        comfort_ok, comfort_issues = self.policy_evaluator.check_comfort_gates(signals)
        if not comfort_ok:
            reasons.extend([f"Comfort: {issue}" for issue in comfort_issues])
            precautionary_flags.append("excessive_discomfort")
        
        # 3) Quality gate (enhancement effectiveness)
        quality_ok, quality_issues = self.policy_evaluator.check_quality_gates(signals)
        if not quality_ok:
            reasons.extend([f"Quality: {issue}" for issue in quality_issues])
            
            # Classify quality issues
            if any("speech_intelligibility" in issue for issue in quality_issues):
                precautionary_flags.append("speech_unintelligible")
            if any("noise_reduction" in issue for issue in quality_issues):
                precautionary_flags.append("noise_reduction_poor")
            if any("latency" in issue for issue in quality_issues):
                precautionary_flags.append("latency_excessive")
        
        # 4) Environmental assessment (context awareness)
        env_ok, env_issues = self.policy_evaluator.check_environmental_gates(signals)
        if not env_ok:
            reasons.extend([f"Environment: {issue}" for issue in env_issues])
        
        # 5) Mode-specific restrictions
        if signals.listening_mode == ListeningMode.NOISY and not self.policy.allow_full_in_noisy:
            if signals.desired_level == OutputAuthorizationLevel.FULL_ENHANCEMENT:
                reasons.append("Noisy mode limits full enhancement")
                precautionary_flags.append("noisy_environment_restriction")
        
        # 6) Policy maximum level
        policy_max_level = self.policy.max_allowed_level
        if signals.desired_level > policy_max_level:
            reasons.append(f"Policy cap to {int(policy_max_level)}")
        
        # 7) Compute severity-weighted precautionary penalty
        precautionary_penalty, penalty_explanation = \
            self.policy_evaluator.compute_precautionary_penalty(precautionary_flags)
        
        # 8) Extract peer values (multi-model enhancement scores)
        peer_values = list(signals.peer_enhancement_scores)
        
        # 9) Build context for AILEE pipeline
        ctx = self._build_context(signals, reasons, precautionary_flags, aggregate_unc)
        
        # 10) Apply penalties
        base_score = signals.proposed_action_trust_score
        score_after_precaution = base_score * (1.0 - precautionary_penalty)
        uncertainty_penalty = aggregate_unc.aggregate_uncertainty_score * 0.20
        final_adjusted_score = score_after_precaution * (1.0 - uncertainty_penalty)
        
        ctx["precautionary_penalty"] = precautionary_penalty
        ctx["penalty_explanation"] = penalty_explanation
        ctx["uncertainty_penalty"] = uncertainty_penalty
        ctx["final_adjustment"] = base_score - final_adjusted_score
        
        # 11) AILEE pipeline
        ailee_result = self.pipeline.process(
            raw_value=float(final_adjusted_score),
            raw_confidence=float(1.0 - aggregate_unc.aggregate_uncertainty_score),
            peer_values=peer_values,
            timestamp=ts,
            context=ctx,
        )
        
        # 12) Determine maximum allowed level from uncertainty
        max_uncertainty_level = self._compute_level_ceiling_from_uncertainty(aggregate_unc)
        
        # 13) Make decision with all constraints
        decision = self._make_auditory_decision(
            signals,
            ailee_result,
            reasons,
            precautionary_flags,
            aggregate_unc,
            ts,
            max_device_level,
            max_uncertainty_level,
            policy_max_level
        )
        
        # 14) Compute decision delta
        decision_delta = self._compute_decision_delta(signals, decision, aggregate_unc, ts)
        if decision_delta:
            decision.metadata["decision_delta"] = decision_delta
        
        # 15) Generate predictive warning if enabled
        warning = None
        if self.policy.enable_predictive_warnings:
            warning = self._predict_warning(signals, decision, aggregate_unc)
        
        # 16) Log and track
        self._log_and_track(ts, signals, decision, reasons, aggregate_unc, warning)
        
        return decision
    
    def _compute_level_ceiling_from_uncertainty(
        self,
        aggregate_unc: AuditoryUncertainty
    ) -> OutputAuthorizationLevel:
        """
        Compute maximum allowed level based on uncertainty.
        
        COMPUTATION: Translates uncertainty metrics to authorization ceiling.
        Result used as one of multiple ceilings in evaluate().
        
        Philosophy: High uncertainty limits output level regardless of score.
        """
        unc_score = aggregate_unc.aggregate_uncertainty_score
        
        if unc_score > 0.50:
            # Extreme uncertainty: diagnostic only
            return OutputAuthorizationLevel.DIAGNOSTIC_ONLY
        if unc_score > 0.35:
            # High uncertainty: safety limited
            return OutputAuthorizationLevel.SAFETY_LIMITED
        if unc_score > 0.25:
            # Moderate uncertainty: comfort optimized max
            return OutputAuthorizationLevel.COMFORT_OPTIMIZED
        
        # Low uncertainty: no ceiling
        return OutputAuthorizationLevel.FULL_ENHANCEMENT
    
    def _build_context(
        self,
        signals: AuditorySignals,
        reasons: List[str],
        precautionary_flags: List[str],
        aggregate_unc: AuditoryUncertainty
    ) -> Dict[str, Any]:
        """Build context dictionary for AILEE pipeline"""
        ctx = {
            "domain": "auditory",
            "listening_mode": signals.listening_mode.value,
            "desired_level": int(signals.desired_level),
            "aggregate_uncertainty": aggregate_unc.aggregate_uncertainty_score,
            "dominant_uncertainty_source": aggregate_unc.dominant_uncertainty_source,
            "reasons": reasons[:],
            "precautionary_flags": precautionary_flags[:],
        }
        
        if signals.environment:
            if signals.environment.ambient_noise_db is not None:
                ctx["ambient_noise_db"] = signals.environment.ambient_noise_db
            if signals.environment.snr_db is not None:
                ctx["snr_db"] = signals.environment.snr_db
        
        if signals.enhancement:
            if signals.enhancement.speech_intelligibility_score is not None:
                ctx["speech_intelligibility"] = signals.enhancement.speech_intelligibility_score
            if signals.enhancement.enhancement_latency_ms is not None:
                ctx["latency_ms"] = signals.enhancement.enhancement_latency_ms
        
        if signals.comfort:
            if signals.comfort.discomfort_score is not None:
                ctx["discomfort"] = signals.comfort.discomfort_score
        
        ctx.update(signals.context)
        return ctx
    
    def _make_auditory_decision(
        self,
        signals: AuditorySignals,
        ailee_result: DecisionResult,
        reasons: List[str],
        precautionary_flags: List[str],
        aggregate_unc: AuditoryUncertainty,
        ts: float,
        max_device_level: OutputAuthorizationLevel,
        max_uncertainty_level: OutputAuthorizationLevel,
        policy_max_level: OutputAuthorizationLevel
    ) -> AuditoryDecision:
        """Convert AILEE result to auditory-specific decision with multiple ceilings"""
        
        validated_score = ailee_result.validated_value
        
        # Determine base level from score
        if validated_score >= 0.85:
            score_level = OutputAuthorizationLevel.FULL_ENHANCEMENT
        elif validated_score >= 0.70:
            score_level = OutputAuthorizationLevel.COMFORT_OPTIMIZED
        elif validated_score >= 0.50:
            score_level = OutputAuthorizationLevel.SAFETY_LIMITED
        elif validated_score >= 0.25:
            score_level = OutputAuthorizationLevel.DIAGNOSTIC_ONLY
        else:
            score_level = OutputAuthorizationLevel.NO_OUTPUT
        
        # Apply multiple ceilings (take minimum)
        authorized_level = min(
            score_level,
            max_device_level,
            max_uncertainty_level,
            policy_max_level,
            signals.desired_level
        )
        
        # Determine outcome
        if authorized_level == OutputAuthorizationLevel.NO_OUTPUT:
            outcome = DecisionOutcome.SUPPRESSED
            recommendation = "suppress_enhancement_quality_insufficient"
        elif authorized_level == OutputAuthorizationLevel.DIAGNOSTIC_ONLY:
            outcome = DecisionOutcome.DIAGNOSTIC_FALLBACK
            recommendation = "diagnostic_mode_only"
        elif authorized_level < signals.desired_level:
            outcome = DecisionOutcome.LIMITED
            recommendation = f"limited_to_{authorized_level.name.lower()}"
        else:
            outcome = DecisionOutcome.AUTHORIZED
            recommendation = f"authorized_{authorized_level.name.lower()}"
        
        # Add ceiling explanations
        metadata_flags = {}
        if max_device_level < score_level:
            reasons.append(f"Device health limited to {max_device_level.name}")
            metadata_flags["device_ceiling_applied"] = True
        
        if max_uncertainty_level < score_level:
            reasons.append(f"Uncertainty limited to {max_uncertainty_level.name}")
            metadata_flags["uncertainty_ceiling_applied"] = True
        
        # Compute output dB cap
        output_db_cap = self._compute_output_db_cap(signals, authorized_level)
        
        # Compute safety margin
        safety_margin = None
        if signals.hearing_profile:
            max_safe = signals.hearing_profile.max_safe_output_db
            safety_margin_value = signals.hearing_profile.get_dynamic_safety_margin()
            safety_margin = max_safe - safety_margin_value - output_db_cap
        
        # Generate enhancement constraints
        enhancement_constraints = None
        if authorized_level >= OutputAuthorizationLevel.SAFETY_LIMITED:
            enhancement_constraints = self._generate_enhancement_constraints(
                signals, precautionary_flags, authorized_level
            )
        
        return AuditoryDecision(
            authorized_level=authorized_level,
            decision_outcome=outcome,
            validated_trust_score=validated_score,
            confidence_score=ailee_result.confidence_score,
            output_db_cap=output_db_cap,
            enhancement_constraints=enhancement_constraints,
            recommendation=recommendation,
            reasons=reasons[:],
            ailee_result=ailee_result,
            safety_margin_db=safety_margin,
            precautionary_flags=precautionary_flags[:],
            metadata={
                "timestamp": ts,
                "listening_mode": signals.listening_mode.value,
                "aggregate_uncertainty": aggregate_unc.aggregate_uncertainty_score,
                "dominant_uncertainty_source": aggregate_unc.dominant_uncertainty_source,
                "max_device_level": int(max_device_level),
                "max_uncertainty_level": int(max_uncertainty_level),
                **metadata_flags,
            }
        )
    
    def _compute_output_db_cap(
        self,
        signals: AuditorySignals,
        authorized_level: OutputAuthorizationLevel
    ) -> float:
        """
        Compute safe output dB SPL cap.
        
        COMPUTATION: Calculates output limit based on multiple factors.
        Result enforced by hearing aid hardware.
        
        Philosophy: Start with user preference, adjust for context, enforce safety.
        """
        # Base output from hearing profile
        default_output = 70.0
        if signals.hearing_profile:
            default_output = signals.hearing_profile.preferred_output_db
        
        # Ambient noise compensation
        ambient_boost = 0.0
        if signals.environment and signals.environment.ambient_noise_db:
            ambient_noise = signals.environment.ambient_noise_db
            if ambient_noise >= 80.0:
                ambient_boost = 8.0
            elif ambient_noise >= 70.0:
                ambient_boost = 4.0
            elif ambient_noise >= 60.0:
                ambient_boost = 2.0
        
        # Level-based reduction
        level_reduction = 0.0
        if authorized_level == OutputAuthorizationLevel.DIAGNOSTIC_ONLY:
            level_reduction = 15.0  # Very conservative
        elif authorized_level == OutputAuthorizationLevel.SAFETY_LIMITED:
            level_reduction = 8.0   # Conservative
        elif authorized_level == OutputAuthorizationLevel.COMFORT_OPTIMIZED:
            level_reduction = 3.0   # Slight reduction
        
        output_target = default_output + ambient_boost - level_reduction
        
        # Safety ceiling (absolute maximum)
        max_safe = self.policy.max_output_db_spl
        
        # User-specific ceiling
        if signals.hearing_profile:
            user_max = signals.hearing_profile.max_safe_output_db
            safety_margin = signals.hearing_profile.get_dynamic_safety_margin()
            max_safe = min(max_safe, user_max - safety_margin)
        
        # Apply safety profile adjustments
        if self.policy.user_safety_profile == UserSafetyProfile.PEDIATRIC:
            max_safe = min(max_safe, 85.0)  # Extra protection for children
        elif self.policy.user_safety_profile == UserSafetyProfile.TINNITUS_RISK:
            max_safe = min(max_safe, 90.0)  # Reduced for tinnitus risk
        
        return max(0.0, min(max_safe, output_target))
    
    def _generate_enhancement_constraints(
        self,
        signals: AuditorySignals,
        precautionary_flags: List[str],
        authorized_level: OutputAuthorizationLevel
    ) -> Dict[str, Any]:
        """Generate constraints for enhancement execution"""
        constraints = {
            "monitoring_required": True,
            "max_output_db": self._compute_output_db_cap(signals, authorized_level),
        }
        
        # Latency constraints
        if signals.enhancement and signals.enhancement.enhancement_latency_ms:
            constraints["max_latency_ms"] = self.policy.max_latency_ms
        
        # Safety features
        if self.policy.enable_automatic_volume_limiting:
            constraints["automatic_volume_limiting"] = True
            constraints["volume_limiter_attack_ms"] = 5.0
            constraints["volume_limiter_release_ms"] = 100.0
        
        # Mode-specific constraints
        if signals.listening_mode == ListeningMode.MUSIC:
            constraints["preserve_dynamic_range"] = True
            constraints["limit_compression_ratio"] = 3.0
        elif signals.listening_mode == ListeningMode.SPEECH_FOCUS:
            constraints["prioritize_speech_bands"] = True
            constraints["speech_frequency_range_hz"] = (500, 4000)
        
        # Precautionary measures
        if len(precautionary_flags) > 0:
            constraints["precautionary_measures"] = precautionary_flags[:]
            constraints["enhanced_monitoring"] = True
        
        # Feedback prevention
        if signals.device_health and signals.device_health.feedback_detected:
            constraints["feedback_cancellation_aggressive"] = True
            constraints["gain_reduction_db"] = 6.0
        
        return constraints
    
    def _predict_warning(
        self,
        signals: AuditorySignals,
        decision: AuditoryDecision,
        aggregate_unc: AuditoryUncertainty
    ) -> Optional[str]:
        """
        Generate predictive warning if risks detected.
        
        ADVISORY: Suggests user actions but does not block authorization.
        Warnings are informational only and do not affect decision outcome.
        
        Returns:
            Warning message if risks detected, None otherwise
        """
        
        # Feedback escalation
        if signals.device_health and signals.device_health.feedback_detected:
            return "Feedback detected: consider repositioning device"
        
        # Fatigue risk
        if signals.comfort and signals.comfort.fatigue_risk_score:
            if signals.comfort.fatigue_risk_score > 0.60:
                return "Listening fatigue rising: suggest break within 30 minutes"
        
        # Quality degradation
        if decision.validated_trust_score < 0.50:
            return "Enhancement quality declining: may need recalibration"
        
        # Battery warning
        if signals.device_health and signals.device_health.battery_level:
            if signals.device_health.battery_level < 0.15:
                return "Battery low: enhanced features may be reduced soon"
        
        # Uncertainty warning
        if aggregate_unc.aggregate_uncertainty_score > 0.40:
            return f"High uncertainty in {aggregate_unc.dominant_uncertainty_source}: verify measurements"
        
        return None
    
    def _compute_decision_delta(
        self,
        signals: AuditorySignals,
        decision: AuditoryDecision,
        aggregate_unc: AuditoryUncertainty,
        ts: float
    ) -> Optional[AuditoryDecisionDelta]:
        """Compute change since last decision with null safety"""
        
        # Null safety check
        if None in (
            self._last_trust_score,
            self._last_output_db,
            self._last_enhancement_quality,
            self._last_uncertainty,
        ):
            return None
        
        trust_delta = decision.validated_trust_score - self._last_trust_score
        output_delta = decision.output_db_cap - self._last_output_db
        
        # Compute enhancement quality (aggregate of enhancement metrics)
        current_quality = signals.proposed_action_trust_score
        quality_delta = current_quality - self._last_enhancement_quality
        
        # Comfort delta
        comfort_delta = 0.0
        if signals.comfort and signals.comfort.discomfort_score is not None:
            if self._last_comfort_score is not None:
                comfort_delta = signals.comfort.discomfort_score - self._last_comfort_score
        
        authorization_changed = (
            decision.authorized_level != self._last_decision
        )
        
        mode_changed = (
            signals.listening_mode != self._last_listening_mode
        )
        
        time_delta = ts - (self._last_decision_time or ts)
        
        return AuditoryDecisionDelta(
            trust_score_delta=trust_delta,
            output_level_delta_db=output_delta,
            comfort_delta=comfort_delta,
            enhancement_quality_delta=quality_delta,
            authorization_level_changed=authorization_changed,
            listening_mode_changed=mode_changed,
            time_since_last_decision_seconds=time_delta,
        )
    
    def _log_and_track(
        self,
        ts: float,
        signals: AuditorySignals,
        decision: AuditoryDecision,
        reasons: List[str],
        aggregate_unc: AuditoryUncertainty,
        warning: Optional[str]
    ):
        """Combined logging and state tracking"""
        
        event_type_map = {
            DecisionOutcome.AUTHORIZED: "enhancement_authorized",
            DecisionOutcome.LIMITED: "enhancement_limited",
            DecisionOutcome.DIAGNOSTIC_FALLBACK: "diagnostic_fallback",
            DecisionOutcome.SUPPRESSED: "enhancement_suppressed",
            DecisionOutcome.HARDWARE_FAULT: "hardware_fault",
        }
        
        event = AuditoryEvent(
            timestamp=ts,
            event_type=event_type_map.get(decision.decision_outcome, "unknown"),
            listening_mode=signals.listening_mode,
            desired_level=signals.desired_level,
            authorized_level=decision.authorized_level,
            decision_outcome=decision.decision_outcome,
            proposed_action_trust_score=signals.proposed_action_trust_score,
            output_db_cap=decision.output_db_cap,
            validated_trust_score=decision.validated_trust_score,
            confidence_score=decision.confidence_score,
            reasons=reasons[:],
            environment_metrics=signals.environment,
            enhancement_metrics=signals.enhancement,
            comfort_metrics=signals.comfort,
            device_health=signals.device_health,
            aggregate_uncertainty=aggregate_unc,
            ailee_decision=decision.ailee_result,
            warning=warning,
            metadata=decision.metadata,
        )
        
        self._event_log.append(event)
        
        # Trim log if needed
        if len(self._event_log) > self.policy.max_event_log_size:
            self._event_log = self._event_log[-self.policy.max_event_log_size:]
        
        self._last_event = event
        
        # Update state tracking
        self._last_decision = decision.authorized_level
        self._last_trust_score = decision.validated_trust_score
        self._last_output_db = decision.output_db_cap
        self._last_enhancement_quality = signals.proposed_action_trust_score
        self._last_uncertainty = aggregate_unc.aggregate_uncertainty_score
        self._last_decision_time = ts
        self._last_listening_mode = signals.listening_mode
        
        if signals.comfort and signals.comfort.discomfort_score is not None:
            self._last_comfort_score = signals.comfort.discomfort_score
    
    # -------------------------
    # Public API - Explainability
    # -------------------------
    
    def explain_decision(self, decision: AuditoryDecision) -> str:
        """Generate plain-language explanation of decision"""
        lines = []
        
        lines.append("=" * 70)
        lines.append("AUDITORY GOVERNANCE DECISION EXPLANATION")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append(f"DECISION: {decision.decision_outcome.value}")
        lines.append(f"Authorized Level: {decision.authorized_level.name}")
        lines.append(f"Output Cap: {decision.output_db_cap:.1f} dB SPL")
        if decision.safety_margin_db is not None:
            lines.append(f"Safety Margin: {decision.safety_margin_db:.1f} dB")
        lines.append("")
        
        lines.append(f"Validated Trust Score: {decision.validated_trust_score:.2f} / 1.00")
        lines.append(f"Decision Confidence: {decision.confidence_score:.2f} / 1.00")
        lines.append("")
        
        lines.append("WHAT MATTERED MOST:")
        lines.append("-" * 70)
        
        if decision.ailee_result:
            lines.append(f"• AILEE Pipeline Status: {decision.ailee_result.status.value}")
            if decision.ailee_result.grace_applied:
                lines.append("  → Grace conditions applied")
            if decision.ailee_result.consensus_status:
                lines.append(f"  → Consensus: {decision.ailee_result.consensus_status}")
        
        if decision.metadata.get("aggregate_uncertainty"):
            unc = decision.metadata["aggregate_uncertainty"]
            source = decision.metadata.get("dominant_uncertainty_source", "unknown")
            lines.append(f"• Aggregate Uncertainty: {unc:.2f} (primary: {source})")
        
        if decision.metadata.get("uncertainty_ceiling_applied"):
            lines.append("• ⚠️  Uncertainty limited output level")
        
        if decision.metadata.get("device_ceiling_applied"):
            lines.append("• ⚠️  Device health limited output level")
        
        lines.append("")
        
        if decision.reasons:
            lines.append("LIMITING FACTORS:")
            lines.append("-" * 70)
            for reason in decision.reasons:
                lines.append(f"• {reason}")
            lines.append("")
        
        if decision.precautionary_flags:
            lines.append("PRECAUTIONARY FLAGS:")
            lines.append("-" * 70)
            for flag in decision.precautionary_flags:
                severity = AUDITORY_FLAG_SEVERITY.get(flag, 0.03)
                lines.append(f"• {flag} (severity: {severity:.2f})")
            lines.append("")
        
        if decision.enhancement_constraints:
            lines.append("ENHANCEMENT CONSTRAINTS:")
            lines.append("-" * 70)
            for key, value in decision.enhancement_constraints.items():
                lines.append(f"• {key}: {value}")
            lines.append("")
        
        if decision.warning:
            lines.append("⚠️  WARNING:")
            lines.append("-" * 70)
            lines.append(f"→ {decision.warning}")
            """
AILEE Trust Layer — AUDITORY Domain
Version: 1.0.1 - Production Grade

Auditory governance domain for AI-enhanced hearing and assistive audio systems.

This domain does NOT implement DSP, beamforming, or hearing-aid firmware.
It governs whether enhanced audio is safe and beneficial enough to deploy based on:
- Environmental noise and signal quality
- Speech intelligibility and enhancement confidence
- Output loudness safety and user comfort
- Device health and feedback risk
- Latency constraints for natural listening

Primary governed signals:
- Enhancement quality scores (speech intelligibility, noise reduction)
- Output loudness caps (dB SPL) aligned to user profiles
- Device health and feedback detection
- Listening mode constraints (quiet, noisy, speech focus, music)

Trust level semantics:
    0 = NO_OUTPUT         Suppress enhancement; fall back to passive/diagnostic
    1 = DIAGNOSTIC_ONLY   Minimal processing for diagnostics
    2 = SAFETY_LIMITED    Safe, conservative enhancement
    3 = COMFORT_OPTIMIZED Comfort-optimized enhancement within safety caps
    4 = FULL_ENHANCEMENT  Full enhancement authorized

DECISION PHILOSOPHY:
===================
Auditory systems must balance:
1. Safety (hearing damage prevention, comfortable loudness)
2. Quality (speech intelligibility, natural sound)
3. Comfort (fatigue prevention, user adaptation)
4. Latency (real-time processing for natural listening)

Therefore, AILEE Auditory Governance implements:
- **Hearing safety first**: Output caps based on audiological guidelines
- **Quality-driven authorization**: Require proven enhancement benefit
- **Comfort monitoring**: Track fatigue and discomfort over time
- **Device health awareness**: Degrade gracefully on hardware issues
- **Latency constraints**: Prioritize real-time over maximum enhancement

CRITICAL: This is a SAFETY system for medical/assistive devices.
Default bias: Conservative output levels until quality proven.

INTEGRATION EXAMPLE:

    # Setup (once)
    policy = AuditoryGovernancePolicy(
        max_allowed_level=OutputAuthorizationLevel.COMFORT_OPTIMIZED,
        max_output_db_spl=100.0,
        user_safety_profile=UserSafetyProfile.STANDARD,
    )
    governor = AuditoryGovernor(policy=policy)
    
    # Per-decision evaluation
    while device_active:
        signals = AuditorySignals(
            proposed_action_trust_score=0.82,  # Enhancement quality aggregate
            desired_level=OutputAuthorizationLevel.FULL_ENHANCEMENT,
            listening_mode=ListeningMode.SPEECH_FOCUS,
            environment=EnvironmentMetrics(
                ambient_noise_db=68.0,
                snr_db=12.0,
                reverberation_time_s=0.45,
                transient_noise_score=0.25,
            ),
            enhancement=EnhancementMetrics(
                speech_intelligibility_score=0.82,
                noise_reduction_score=0.74,
                enhancement_latency_ms=8.0,
                ai_confidence=0.88,
            ),
            comfort=ComfortMetrics(
                perceived_loudness_db=78.0,
                discomfort_score=0.15,
                fatigue_risk_score=0.12,
            ),
            device_health=DeviceHealth(
                mic_health_score=0.98,
                battery_level=0.62,
                feedback_detected=False,
            ),
            hearing_profile=HearingProfile(
                max_safe_output_db=95.0,
                preferred_output_db=75.0,
            ),
        )
        
        decision = governor.evaluate(signals)
        
        if decision.authorized_level >= OutputAuthorizationLevel.SAFETY_LIMITED:
            hearing_aid.apply_enhancement(
                level=decision.authorized_level,
                output_cap=decision.output_db_cap,
                constraints=decision.enhancement_constraints
            )
        else:
            hearing_aid.fallback_pass_through()
        
        # Log for audiologist review
        if decision.warning:
            alert_user(decision.warning)
        
        # Compliance logging
        medical_device_logger.record_decision(governor.get_last_event())

This module is designed for:
- Hearing aids and assistive listening devices
- Cochlear implant sound processors
- Hearables and augmented audio devices
- Tinnitus management systems
- Professional audio monitoring systems
- Voice enhancement for communication devices

ARCHITECTURAL NOTE:
This module is deterministic, side-effect-free (except logging),
and suitable for real-time decision support systems operating at
10Hz - 100Hz (every 10ms to 100ms for audio processing decisions).

REGULATORY COMPLIANCE:
Designed to support FDA Class I/II medical device requirements,
IEC 60601-1 safety standards, and ISO 13485 quality management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple
import statistics
import time
import math


# ---- Core imports ----
try:
    from ailee_trust_pipeline_v1 import (
        AileeTrustPipeline,
        AileeConfig,
        DecisionResult,
        SafetyStatus
    )
except Exception:
    AileeTrustPipeline = None
    AileeConfig = None
    DecisionResult = None
    SafetyStatus = None


# ===== SEVERITY WEIGHTING FOR FLAGS =====

AUDITORY_FLAG_SEVERITY: Dict[str, float] = {
    "hearing_damage_risk": 0.15,
    "feedback_detected": 0.12,
    "hardware_fault_critical": 0.10,
    "excessive_discomfort": 0.08,
    "clipping_detected": 0.07,
    "latency_excessive": 0.06,
    "speech_unintelligible": 0.05,
    "battery_critical": 0.04,
    "microphone_degraded": 0.03,
    "noise_reduction_poor": 0.03,
}


# -----------------------------
# Output Authorization Levels
# -----------------------------

class OutputAuthorizationLevel(IntEnum):
    """
    Discrete output authorization levels for auditory enhancement.
    
    Staged escalation from passive to full AI enhancement.
    """
    NO_OUTPUT = 0           # Suppress all enhancement
    DIAGNOSTIC_ONLY = 1     # Minimal processing for diagnostics
    SAFETY_LIMITED = 2      # Safe, conservative enhancement
    COMFORT_OPTIMIZED = 3   # Comfort-optimized within safety
    FULL_ENHANCEMENT = 4    # Full AI enhancement authorized


class ListeningMode(str, Enum):
    """Listening modes for hearing systems"""
    QUIET = "QUIET"
    SPEECH_FOCUS = "SPEECH_FOCUS"
    NOISY = "NOISY"
    MUSIC = "MUSIC"
    OUTDOOR_WINDY = "OUTDOOR_WINDY"
    EMERGENCY_ALERTS = "EMERGENCY_ALERTS"
    TELECOIL = "TELECOIL"
    UNKNOWN = "UNKNOWN"


class UserSafetyProfile(str, Enum):
    """User safety profiles for output constraints"""
    PEDIATRIC = "PEDIATRIC"          # Most conservative
    STANDARD = "STANDARD"            # Normal adult
    TINNITUS_RISK = "TINNITUS_RISK" # Extra caution
    PROFESSIONAL = "PROFESSIONAL"    # Musicians, audio professionals
    UNKNOWN = "UNKNOWN"


class DecisionOutcome(str, Enum):
    """Auditory governance decision outcomes"""
    AUTHORIZED = "AUTHORIZED"
    LIMITED = "LIMITED"
    DIAGNOSTIC_FALLBACK = "DIAGNOSTIC_FALLBACK"
    SUPPRESSED = "SUPPRESSED"
    HARDWARE_FAULT = "HARDWARE_FAULT"


class RegulatoryGateResult(str, Enum):
    """Regulatory compliance results"""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


# -----------------------------
# Hearing Profile & Safety
# -----------------------------

@dataclass(frozen=True)
class HearingProfile:
    """
    User hearing profile and safety limits.
    
    Based on audiological assessment and regulatory guidelines.
    """
    max_safe_output_db: float  # Absolute maximum (e.g., 95 dB SPL)
    preferred_output_db: float  # User preference (e.g., 75 dB SPL)
    
    # Frequency-specific loss (optional, for personalization)
    frequency_loss_profile: Optional[Dict[str, float]] = None  # Hz -> dB loss
    
    # User risk factors
    tinnitus_history: bool = False
    noise_induced_loss: bool = False
    age_years: Optional[int] = None
    
    # Audiologist notes
    notes: Optional[str] = None
    last_assessment_date: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_dynamic_safety_margin(self) -> float:
        """
        Compute safety margin based on risk factors.
        
        Returns dB reduction from max_safe_output_db.
        """
        margin = 0.0
        
        if self.tinnitus_history:
            margin += 5.0  # Extra 5 dB margin
        
        if self.noise_induced_loss:
            margin += 3.0
        
        if self.age_years is not None:
            if self.age_years < 18:
                margin += 10.0  # Pediatric protection
            elif self.age_years > 65:
                margin += 2.0   # Presbycusis consideration
        
        return margin


# -----------------------------
# Environmental Metrics
# -----------------------------

@dataclass(frozen=True)
class EnvironmentMetrics:
    """
    Environmental acoustic metrics.
    
    Measured or estimated from microphone signals.
    """
    ambient_noise_db: Optional[float] = None  # Background noise level
    snr_db: Optional[float] = None            # Signal-to-noise ratio
    reverberation_time_s: Optional[float] = None  # RT60
    transient_noise_score: Optional[float] = None  # 0-1, sudden loud sounds
    wind_level: Optional[float] = None        # 0-1, wind noise presence
    
    # Acoustic scene classification
    scene_confidence: Optional[float] = None
    scene_type: Optional[str] = None  # "restaurant", "office", "street"
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_challenging_environment(self) -> Tuple[bool, List[str]]:
        """Check if environment is acoustically challenging"""
        issues: List[str] = []
        
        if self.ambient_noise_db is not None:
            if self.ambient_noise_db > 80.0:
                issues.append(f"high_ambient_noise ({self.ambient_noise_db:.1f} dB)")
        
        if self.snr_db is not None:
            if self.snr_db < 5.0:
                issues.append(f"poor_snr ({self.snr_db:.1f} dB)")
        
        if self.reverberation_time_s is not None:
            if self.reverberation_time_s > 1.0:
                issues.append(f"high_reverberation ({self.reverberation_time_s:.2f}s)")
        
        if self.transient_noise_score is not None:
            if self.transient_noise_score > 0.60:
                issues.append("frequent_transients")
        
        return len(issues) > 0, issues


# -----------------------------
# Enhancement Metrics
# -----------------------------

@dataclass(frozen=True)
class EnhancementMetrics:
    """
    Model-driven enhancement quality metrics.
    
    Outputs from DSP/AI enhancement algorithms.
    """
    speech_intelligibility_score: Optional[float] = None  # 0-1
    noise_reduction_score: Optional[float] = None         # 0-1
    spectral_balance_score: Optional[float] = None        # 0-1, naturalness
    enhancement_latency_ms: Optional[float] = None        # Processing delay
    ai_confidence: Optional[float] = None                 # Model confidence
    
    # Specific enhancement features
    beamforming_active: bool = False
    noise_suppression_db: Optional[float] = None
    compression_ratio: Optional[float] = None
    
    # Quality degradation indicators
    artifacts_detected: bool = False
    distortion_score: Optional[float] = None  # 0-1
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_quality_acceptable(
        self,
        min_intelligibility: float = 0.65,
        min_noise_reduction: float = 0.55,
        max_latency_ms: float = 20.0
    ) -> Tuple[bool, List[str]]:
        """Check if enhancement quality meets thresholds"""
        issues: List[str] = []
        
        if self.speech_intelligibility_score is not None:
            if self.speech_intelligibility_score < min_intelligibility:
                issues.append(
                    f"speech_intelligibility={self.speech_intelligibility_score:.2f} "
                    f"< {min_intelligibility}"
                )
        
        if self.noise_reduction_score is not None:
            if self.noise_reduction_score < min_noise_reduction:
                issues.append(
                    f"noise_reduction={self.noise_reduction_score:.2f} "
                    f"< {min_noise_reduction}"
                )
        
        if self.enhancement_latency_ms is not None:
            if self.enhancement_latency_ms > max_latency_ms:
                issues.append(
                    f"latency={self.enhancement_latency_ms:.1f}ms > {max_latency_ms}ms"
                )
        
        if self.artifacts_detected:
            issues.append("artifacts_detected_in_output")
        
        return len(issues) == 0, issues


# -----------------------------
# Comfort Metrics
# -----------------------------

@dataclass(frozen=True)
class ComfortMetrics:
    """
    User comfort and fatigue indicators.
    
    Can be measured (e.g., via sensors) or self-reported.
    """
    perceived_loudness_db: Optional[float] = None  # Psychoacoustic loudness
    discomfort_score: Optional[float] = None       # 0-1, immediate discomfort
    fatigue_risk_score: Optional[float] = None     # 0-1, listening fatigue
    
    # User interaction indicators
    volume_adjustments_count: Optional[int] = None
    device_removal_events: Optional[int] = None
    
    # Temporal factors
    continuous_usage_hours: Optional[float] = None
    time_since_last_break_minutes: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_comfort_acceptable(
        self,
        max_discomfort: float = 0.35,
        max_fatigue: float = 0.60
    ) -> Tuple[bool, List[str]]:
        """Check if comfort metrics are within acceptable range"""
        issues: List[str] = []
        
        if self.discomfort_score is not None:
            if self.discomfort_score > max_discomfort:
                issues.append(
                    f"discomfort={self.discomfort_score:.2f} exceeds {max_discomfort}"
                )
        
        if self.fatigue_risk_score is not None:
            if self.fatigue_risk_score > max_fatigue:
                issues.append(
                    f"fatigue_risk={self.fatigue_risk_score:.2f} exceeds {max_fatigue}"
                )
        
        if self.continuous_usage_hours is not None:
            if self.continuous_usage_hours > 8.0:
                issues.append(
                    f"continuous_usage={self.continuous_usage_hours:.1f}hrs > 8hrs"
                )
        
        return len(issues) == 0, issues


# -----------------------------
# Device Health
# -----------------------------

@dataclass(frozen=True)
class DeviceHealth:
    """
    Device status and safety indicators.
    
    Hardware health monitoring for safe operation.
    """
    mic_health_score: Optional[float] = None     # 0-1, microphone quality
    speaker_health_score: Optional[float] = None # 0-1, speaker quality
    battery_level: Optional[float] = None        # 0-1
    temperature_c: Optional[float] = None
    
    # Critical faults
    hardware_faults: Tuple[str, ...] = ()
    feedback_detected: bool = False
    occlusion_detected: bool = False
    clipping_detected: bool = False
    
    # Calibration status
    last_calibration_timestamp: Optional[float] = None
    calibration_valid: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_health_issues(
        self,
        min_mic_health: float = 0.75,
        min_battery: float = 0.10
    ) -> Tuple[bool, List[str]]:
        """Check for device health issues"""
        issues: List[str] = []
        
        if self.hardware_faults:
            issues.extend([f"hardware_fault: {f}" for f in self.hardware_faults])
        
        if self.feedback_detected:
            issues.append("acoustic_feedback_detected")
        
        if self.clipping_detected:
            issues.append("output_clipping_detected")
        
        if self.occlusion_detected:
            issues.append("ear_canal_occlusion")
        
        if self.mic_health_score is not None:
            if self.mic_health_score < min_mic_health:
                issues.append(f"mic_health={self.mic_health_score:.2f} < {min_mic_health}")
        
        if self.battery_level is not None:
            if self.battery_level < min_battery:
                issues.append(f"battery_critical={self.battery_level:.2%}")
        
        if not self.calibration_valid:
            issues.append("calibration_expired")
        
        return len(issues) > 0, issues


# -----------------------------
# Aggregate Uncertainty
# -----------------------------

@dataclass(frozen=True)
class AuditoryUncertainty:
    """
    Explicit aggregation of uncertainty sources in auditory domain.
    
    Philosophy: High confidence requires low uncertainty in:
    - Enhancement quality measurement
    - Environmental characterization
    - Device health
    - User comfort assessment
    """
    aggregate_uncertainty_score: float  # [0.0 = certain, 1.0 = maximum uncertainty]
    
    # Component uncertainties
    enhancement_uncertainty: float
    environmental_uncertainty: float
    device_uncertainty: float
    comfort_uncertainty: float
    
    # Breakdown for diagnostics
    dominant_uncertainty_source: str
    uncertainty_sources: Dict[str, float] = field(default_factory=dict)
    
    def is_uncertainty_acceptable(self, max_aggregate: float = 0.35) -> Tuple[bool, str]:
        """Check if aggregate uncertainty is within acceptable bounds"""
        if self.aggregate_uncertainty_score > max_aggregate:
            return False, (
                f"aggregate_uncertainty={self.aggregate_uncertainty_score:.2f} "
                f"exceeds {max_aggregate:.2f} (source: {self.dominant_uncertainty_source})"
            )
        return True, "uncertainty_acceptable"


# -----------------------------
# Decision Delta Tracking
# -----------------------------

@dataclass(frozen=True)
class AuditoryDecisionDelta:
    """
    Change tracking since last decision.
    
    Critical for:
    - Real-time monitoring
    - User experience tracking
    - Device performance trends
    """
    trust_score_delta: float
    output_level_delta_db: float
    comfort_delta: float
    enhancement_quality_delta: float
    
    authorization_level_changed: bool
    listening_mode_changed: bool
    
    time_since_last_decision_seconds: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Domain Inputs
# -----------------------------

@dataclass(frozen=True)
class AuditorySignals:
    """
    Governance signals for auditory enhancement assessment.
    Primary input structure for the auditory governor.
    
    Philosophy: We assess enhancement trustworthiness before deployment.
    """
    # Primary trust metric
    proposed_action_trust_score: float  # Aggregate enhancement quality [0-1]
    
    # Desired output level
    desired_level: OutputAuthorizationLevel
    
    # Listening context
    listening_mode: ListeningMode
    
    # Measurement bundles
    environment: Optional[EnvironmentMetrics] = None
    enhancement: Optional[EnhancementMetrics] = None
    comfort: Optional[ComfortMetrics] = None
    device_health: Optional[DeviceHealth] = None
    hearing_profile: Optional[HearingProfile] = None
    
    # Multi-model validation (if multiple enhancement algorithms)
    peer_enhancement_scores: Tuple[float, ...] = ()
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', time.time())


# -----------------------------
# Domain Configuration
# -----------------------------

@dataclass(frozen=True)
class AuditoryGovernancePolicy:
    """
    Domain policy for auditory enhancement governance.
    
    Philosophy: Safety first, quality second, comfort third.
    """
    # Authorization constraints
    max_allowed_level: OutputAuthorizationLevel = OutputAuthorizationLevel.COMFORT_OPTIMIZED
    
    # Output safety limits
    max_output_db_spl: float = 100.0  # Absolute maximum
    max_continuous_output_db: float = 85.0  # For extended use
    
    # User safety profile
    user_safety_profile: UserSafetyProfile = UserSafetyProfile.STANDARD
    
    # Quality thresholds
    min_speech_intelligibility: float = 0.65
    min_noise_reduction: float = 0.55
    min_enhancement_confidence: float = 0.70
    
    # Latency constraints
    max_latency_ms: float = 20.0  # Real-time requirement
    
    # Comfort limits
    max_discomfort_score: float = 0.35
    max_fatigue_risk: float = 0.60
    
    # Device health requirements
    min_mic_health_score: float = 0.75
    min_battery_level: float = 0.10
    
    # Mode-specific policies
    allow_full_in_noisy: bool = False
    require_calibration: bool = True
    
    # Safety features
    enable_predictive_warnings: bool = True
    enable_automatic_volume_limiting: bool = True
    
    # Event logging
    max_event_log_size: int = 5000
    
    metadata: Dict[str, Any] = field(default_factory=dict)


def default_auditory_config() -> AileeConfig:
    """
    Safe defaults for auditory governance pipeline configuration.
    
    Philosophy: Auditory systems prioritize stability and agreement.
    Real-time constraints favor faster convergence.
    """
    if AileeConfig is None:
        raise RuntimeError("AILEE core imports unavailable")
    
    return AileeConfig(
        accept_threshold=0.85,
        borderline_low=0.65,
        borderline_high=0.85,
        
        # Weights: Balance stability with responsiveness
        w_stability=0.40,      # Moderate (faster than ocean, slower than robotics)
        w_agreement=0.35,      # Multi-model consensus important
        w_likelihood=0.25,     # Statistical plausibility
        
        history_window=80,     # ~80 decisions (shorter memory)
        forecast_window=12,    # Look ahead 12 steps
        
        grace_peer_delta=0.20,
        grace_min_peer_agreement_ratio=0.60,
        grace_forecast_epsilon=0.18,
        grace_max_abs_z=2.8,
        
        consensus_quorum=2,    # At least 2 sources
        consensus_delta=0.18,
        consensus_pass_ratio=0.70,
        
        fallback_mode="last_good",
        fallback_clamp_min=0.0,
        fallback_clamp_max=1.0,
        
        hard_min=0.0,
        hard_max=1.0,
        
        enable_grace=True,
        enable_consensus=True,
        enable_audit_metadata=True,
    )


# -----------------------------
# Result Structure
# -----------------------------

@dataclass(frozen=True)
class AuditoryDecision:
    """
    Auditory governance decision result.
    
    Philosophy: Return not just yes/no, but specific output constraints.
    """
    authorized_level: OutputAuthorizationLevel
    decision_outcome: DecisionOutcome
    
    validated_trust_score: float
    confidence_score: float
    
    # Output constraints
    output_db_cap: float
    enhancement_constraints: Optional[Dict[str, Any]] = None
    
    recommendation: str
    reasons: List[str]
    
    # Pipeline results
    ailee_result: Optional[DecisionResult] = None
    
    # Safety summary
    safety_margin_db: Optional[float] = None
    precautionary_flags: Optional[List[str]] = None
    
    # Predictive warnings
    warning: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Auditory Events (Compliance Logging)
# -----------------------------

@dataclass(frozen=True)
class AuditoryEvent:
    """
    Structured event for medical device compliance and user safety logging.
    
    Critical for: FDA/regulatory audits, clinical studies, user safety monitoring.
    """
    timestamp: float
    event_type: str  # "enhancement_authorized" | "limited" | "suppressed"
    
    listening_mode: ListeningMode
    desired_level: OutputAuthorizationLevel
    authorized_level: OutputAuthorizationLevel
    decision_outcome: DecisionOutcome
    
    proposed_action_trust_score: float
    output_db_cap: float
    
    validated_trust_score: float
    confidence_score: float
    
    reasons: List[str]
    
    # Context
    environment_metrics: Optional[EnvironmentMetrics] = None
    enhancement_metrics: Optional[EnhancementMetrics] = None
    comfort_metrics: Optional[ComfortMetrics] = None
    device_health: Optional[DeviceHealth] = None
    aggregate_uncertainty: Optional[AuditoryUncertainty] = None
    
    # Pipeline results
    ailee_decision: Optional[DecisionResult] = None
    
    # Warnings
    warning: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===== POLICY EVALUATOR =====
#
# ROLE: Evaluates constraints and computes recommendations.
# These methods assess safety limits, quality thresholds, and environmental
# challenges. They return constraint violations and suggested ceilings.
#
# IMPORTANT: Final authorization decisions are made by AuditoryGovernor.evaluate().
# PolicyEvaluator outputs are advisory inputs to the decision process.
#
# Decision Flow:
#   PolicyEvaluator → [constraints, ceilings, recommendations]
#   UncertaintyCalculator → [uncertainty metrics]
#   AuditoryGovernor.evaluate() → [AUTHORITATIVE DECISION]

class AuditoryPolicyEvaluator:
    """
    Formalized policy-derived gate checks for auditory governance.
    
    ADVISORY ROLE: Evaluates constraints and recommends ceilings.
    All outputs are inputs to AuditoryGovernor.evaluate() for final decision.
    
    Philosophy: Separates safety policy from decision flow logic.
    """
    
    def __init__(self, policy: AuditoryGovernancePolicy):
        self.policy = policy
    
    def check_device_health_gates(
        self,
        signals: AuditorySignals
    ) -> Tuple[bool, List[str], OutputAuthorizationLevel]:
        """
        Check device health constraints and recommend ceiling.
        
        CONSTRAINT CHECK: Evaluates hardware safety limits.
        Returns violation status and suggested maximum authorization level.
        Final authorization determined by AuditoryGovernor.evaluate().
        
        Returns:
            (passed, issues, max_safe_level)
        """
        issues = []
        max_level = OutputAuthorizationLevel.FULL_ENHANCEMENT
        
        if not signals.device_health:
            return True, [], max_level
        
        device = signals.device_health
        
        # Critical faults
        has_issues, health_issues = device.get_health_issues(
            min_mic_health=self.policy.min_mic_health_score,
            min_battery=self.policy.min_battery_level
        )
        
        if has_issues:
            issues.extend(health_issues)
            
            # Determine severity
            if device.hardware_faults or device.feedback_detected:
                max_level = OutputAuthorizationLevel.DIAGNOSTIC_ONLY
            elif device.clipping_detected or device.mic_health_score and device.mic_health_score < 0.60:
                max_level = OutputAuthorizationLevel.SAFETY_LIMITED
        
        passed = len(issues) == 0
        return passed, issues, max_level
    
    def check_comfort_gates(
        self,
        signals: AuditorySignals
    ) -> Tuple[bool, List[str]]:
        """
        Check user comfort constraints.
        
        CONSTRAINT CHECK: Evaluates comfort thresholds for user safety.
        Returns violation status. Final authorization by AuditoryGovernor.evaluate().
        
        Returns:
            (passed, issues)
        """
        issues = []
        
        if not signals.comfort:
            return True, []
        
        comfort_ok, comfort_issues = signals.comfort.is_comfort_acceptable(
            max_discomfort=self.policy.max_discomfort_score,
            max_fatigue=self.policy.max_fatigue_risk
        )
        
        if not comfort_ok:
            issues.extend(comfort_issues)
        
        return len(issues) == 0, issues
    
    def check_quality_gates(
        self,
        signals: AuditorySignals
    ) -> Tuple[bool, List[str]]:
        """
        Check enhancement quality constraints.
        
        CONSTRAINT CHECK: Evaluates enhancement effectiveness thresholds.
        Returns violation status. Final authorization by AuditoryGovernor.evaluate().
        
        Returns:
            (passed, issues)
        """
        issues = []
        
        if not signals.enhancement:
            return True, []
        
        quality_ok, quality_issues = signals.enhancement.is_quality_acceptable(
            min_intelligibility=self.policy.min_speech_intelligibility,
            min_noise_reduction=self.policy.min_noise_reduction,
            max_latency_ms=self.policy.max_latency_ms
        )
        
        if not quality_ok:
            issues.extend(quality_issues)
        
        return len(issues) == 0, issues
    
    def check_environmental_gates(
        self,
        signals: AuditorySignals
    ) -> Tuple[bool, List[str]]:
        """
        Check environmental acoustic constraints.
        
        CONSTRAINT CHECK: Evaluates environmental challenges.
        Returns violation status. Final authorization by AuditoryGovernor.evaluate().
        
        Returns:
            (passed, issues)
        """
        issues = []
        
        if not signals.environment:
            return True, []
        
        challenging, env_issues = signals.environment.is_challenging_environment()
        
        if challenging:
            issues.extend(env_issues)
        
        return len(issues) == 0, issues
    
    def compute_precautionary_penalty(
        self,
        precautionary_flags: List[str]
    ) -> Tuple[float, str]:
        """
        Compute severity-weighted penalty from precautionary flags.
        
        COMPUTATION: Pure calculation based on flag severities.
        Result used by AuditoryGovernor.evaluate() for score adjustment.
        
        Returns:
            (total_penalty, explanation)
        """
        if not precautionary_flags:
            return 0.0, "no_flags"
        
        total_severity = sum(
            AUDITORY_FLAG_SEVERITY.get(flag, 0.03)
            for flag in precautionary_flags
        )
        
        capped_penalty = min(0.25, total_severity)
        
        severities = [(AUDITORY_FLAG_SEVERITY.get(f, 0.03), f) for f in precautionary_flags]
        most_severe_score, most_severe_flag = max(severities, key=lambda x: x[0])
        
        explanation = (
            f"{len(precautionary_flags)} flags, "
            f"most_severe={most_severe_flag} ({most_severe_score:.2f}), "
            f"total_penalty={capped_penalty:.2f}"
        )
        
        return capped_penalty, explanation


# ===== UNCERTAINTY CALCULATOR =====
#
# ROLE: Computes aggregate uncertainty from multiple measurement sources.
# This is a pure calculation module with no policy decisions.
#
# Decision Flow:
#   UncertaintyCalculator → [uncertainty metrics]
#   AuditoryGovernor.evaluate() → uses metrics for ceiling determination

class AuditoryUncertaintyCalculator:
    """
    Explicit uncertainty aggregation for auditory domain.
    
    COMPUTATION ROLE: Pure calculation, no policy decisions.
    """
    
    @staticmethod
    def compute_aggregate_uncertainty(
        signals: AuditorySignals
    ) -> AuditoryUncertainty:
        """
        Aggregate uncertainty from all auditory sources.
        
        COMPUTATION: Pure calculation, no policy decisions.
        Result used by AuditoryGovernor.evaluate() for ceiling determination.
        
        Returns:
            AuditoryUncertainty with aggregate score and component breakdown
        """
        components = {}
