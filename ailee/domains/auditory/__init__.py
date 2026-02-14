"""
AILEE Trust Layer — AUDITORY Domain
Version: 1.0.1 - Production Grade

Auditory governance domain for AI-enhanced hearing and assistive audio systems.

This module provides governance for hearing aids, cochlear implants, hearables,
and other assistive listening devices. It does NOT implement DSP or firmware,
but rather governs whether AI-enhanced audio is safe and beneficial to deploy.

Quick Start:
-----------
    from ailee_auditory import (
        AuditoryGovernor,
        AuditoryGovernancePolicy,
        AuditorySignals,
        OutputAuthorizationLevel,
        ListeningMode,
        create_auditory_governor,
    )
    
    # Create governor with safety policy
    governor = create_auditory_governor(
        user_safety_profile=UserSafetyProfile.STANDARD,
        max_output_db_spl=100.0,
        max_allowed_level=OutputAuthorizationLevel.COMFORT_OPTIMIZED,
    )
    
    # Evaluate enhancement request
    signals = AuditorySignals(
        proposed_action_trust_score=0.82,
        desired_level=OutputAuthorizationLevel.FULL_ENHANCEMENT,
        listening_mode=ListeningMode.SPEECH_FOCUS,
        # ... add metrics
    )
    
    decision = governor.evaluate(signals)
    
    if decision.authorized_level >= OutputAuthorizationLevel.SAFETY_LIMITED:
        # Apply enhancement
        apply_enhancement(
            level=decision.authorized_level,
            output_cap=decision.output_db_cap,
            constraints=decision.enhancement_constraints
        )

Key Concepts:
------------
- **Output Authorization Levels**: Discrete safety levels from NO_OUTPUT to FULL_ENHANCEMENT
- **Trust Score**: Aggregate quality metric for enhancement [0-1]
- **Output dB Cap**: Maximum safe output loudness in dB SPL
- **Uncertainty Aggregation**: Explicit tracking of measurement confidence
- **Regulatory Compliance**: Event logging for medical device requirements

Safety Philosophy:
-----------------
This is a SAFETY system for medical/assistive devices.
Default bias: Conservative output levels until quality proven.

Priority ordering:
1. Hearing damage prevention (output caps, safety margins)
2. Enhancement quality (speech intelligibility, noise reduction)
3. User comfort (fatigue prevention, discomfort monitoring)
4. Real-time constraints (latency limits for natural listening)

For detailed documentation, see the module docstring in ailee_auditory.py
"""

from .ailee_auditory import (
    # Enums
    OutputAuthorizationLevel,
    ListeningMode,
    UserSafetyProfile,
    DecisionOutcome,
    RegulatoryGateResult,
    
    # Data structures
    HearingProfile,
    EnvironmentMetrics,
    EnhancementMetrics,
    ComfortMetrics,
    DeviceHealth,
    AuditoryUncertainty,
    AuditoryDecisionDelta,
    AuditorySignals,
    
    # Configuration
    AuditoryGovernancePolicy,
    AuditoryDecision,
    
    # Events
    AuditoryEvent,
    
    # Governance components
    AuditoryPolicyEvaluator,
    AuditoryUncertaintyCalculator,
    AuditoryGovernor,
    
    # Utilities
    default_auditory_config,
    create_auditory_governor,
    validate_auditory_signals,
    
    # Constants
    AUDITORY_FLAG_SEVERITY,
)

__version__ = "1.0.1"
__author__ = "AILEE Trust Layer Development Team"
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
]
