"""
AILEE Cross-Ecosystem Translation Domain — v1.0.0

Governance for semantic state and intent translation between incompatible
software-hardware ecosystems (e.g., iOS ↔ Android, proprietary wearables).

This module provides:
- Translation trust evaluation for cross-platform state sharing
- Consent and privacy boundary enforcement
- Semantic fidelity assessment
- Context preservation validation
- Ecosystem capability alignment
- Complete audit trails for compliance

Quick Start:
    >>> from ailee_cross_ecosystem import (
    ...     CrossEcosystemGovernor,
    ...     CrossEcosystemSignals,
    ...     ConsentStatus,
    ...     TranslationTrustLevel,
    ... )
    >>> 
    >>> governor = CrossEcosystemGovernor()
    >>> 
    >>> signals = CrossEcosystemSignals(
    ...     desired_level=TranslationTrustLevel.CONSTRAINED_TRUST,
    ...     semantic_fidelity=0.88,
    ...     source_ecosystem="ios_healthkit",
    ...     target_ecosystem="android_health_connect",
    ...     consent_status=ConsentStatus(
    ...         user_consent_granted=True,
    ...         data_categories=["activity", "heart_rate"],
    ...         allows_automation=True,
    ...     ),
    ...     data_category="heart_rate",
    ... )
    >>> 
    >>> level, decision = governor.evaluate(signals)
    >>> if level >= TranslationTrustLevel.CONSTRAINED_TRUST:
    ...     print("Translation authorized")

⚠️  Important: This module does NOT bypass platform security or modify hardware.
    It determines whether semantic translation is trustworthy enough to act upon.

For detailed documentation, see: https://github.com/your-org/ailee-cross-ecosystem
"""

from .cross_ecosystem_governor import (
    # Core Classes
    CrossEcosystemGovernor,
    CrossEcosystemPolicy,
    CrossEcosystemSignals,
    
    # Trust Levels
    TranslationTrustLevel,
    
    # Consent and Privacy
    ConsentStatus,
    PrivacyBoundaries,
    
    # Context and Capabilities
    ContextPreservation,
    EcosystemCapabilities,
    TranslationPath,
    
    # Events and Audit
    GovernanceEvent,
    
    # Known Ecosystems
    KNOWN_ECOSYSTEMS,
    
    # Configuration
    default_cross_ecosystem_config,
    
    # Convenience Functions
    create_default_governor,
    create_health_data_signals,
    create_wearable_continuity_signals,
    create_consent_violation_signals,
    create_low_fidelity_signals,
    validate_signals,
    export_events_to_dict,
    get_translation_path_info,
)

# Version info
__version__ = "1.0.0"
__author__ = "AILEE Project"
__license__ = "MIT"

# Public API
__all__ = [
    # Core Classes
    "CrossEcosystemGovernor",
    "CrossEcosystemPolicy",
    "CrossEcosystemSignals",
    
    # Trust Levels
    "TranslationTrustLevel",
    
    # Consent and Privacy
    "ConsentStatus",
    "PrivacyBoundaries",
    
    # Context and Capabilities
    "ContextPreservation",
    "EcosystemCapabilities",
    "TranslationPath",
    
    # Events and Audit
    "GovernanceEvent",
    
    # Known Ecosystems
    "KNOWN_ECOSYSTEMS",
    
    # Configuration
    "default_cross_ecosystem_config",
    
    # Factory Functions
    "create_default_governor",
    "create_strict_governor",
    "create_permissive_governor",
    
    # Signal Builders
    "create_health_data_signals",
    "create_wearable_continuity_signals",
    "create_consent_violation_signals",
    "create_low_fidelity_signals",
    
    # Utilities
    "validate_signals",
    "export_events_to_dict",
    "get_translation_path_info",
    
    # Version
    "__version__",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Factory Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_strict_governor(**policy_overrides) -> CrossEcosystemGovernor:
    """
    Create a CrossEcosystemGovernor with strict security configuration.
    
    Enables all security features:
    - Requires explicit consent
    - Enforces privacy boundaries
    - Blocks PII translation
    - Requires intent preservation
    - Strict capability alignment
    - Full audit metadata
    
    Args:
        **policy_overrides: Optional policy parameters to override defaults
        
    Returns:
        CrossEcosystemGovernor: Strictly configured governor instance
        
    Example:
        >>> governor = create_strict_governor()
        >>> signals = create_health_data_signals()
        >>> level, decision = governor.evaluate(signals)
    """
    from .cross_ecosystem_governor import CrossEcosystemPolicy
    
    strict_policy = {
        "min_semantic_fidelity_for_advisory": 0.80,
        "min_semantic_fidelity_for_constrained": 0.85,
        "min_semantic_fidelity_for_full": 0.95,
        "require_explicit_consent": True,
        "max_consent_age_hours": 168.0,  # 7 days
        "max_signal_age_hours": 12.0,
        "warn_signal_age_hours": 6.0,
        "enforce_privacy_boundaries": True,
        "allow_pii_translation": False,
        "min_context_preservation": 0.85,
        "require_intent_preservation": True,
        "require_translation_consensus": True,
        "min_consensus_agreement": 0.90,
        "enforce_capability_alignment": True,
        "min_capability_overlap": 0.75,
        "max_allowed_level": TranslationTrustLevel.CONSTRAINED_TRUST,
        "enable_rate_limiting": True,
        "max_event_log_size": 10000,
    }
    strict_policy.update(policy_overrides)
    
    policy = CrossEcosystemPolicy(**strict_policy)
    cfg = default_cross_ecosystem_config()
    return CrossEcosystemGovernor(cfg=cfg, policy=policy)


def create_permissive_governor(**policy_overrides) -> CrossEcosystemGovernor:
    """
    Create a CrossEcosystemGovernor with permissive configuration for testing/development.
    
    Relaxes many requirements:
    - Lower fidelity thresholds
    - Longer consent validity
    - More permissive privacy boundaries
    - Relaxed capability alignment
    
    WARNING: Not recommended for production use with sensitive data!
    
    Args:
        **policy_overrides: Optional policy parameters to override defaults
        
    Returns:
        CrossEcosystemGovernor: Permissively configured governor instance
        
    Example:
        >>> governor = create_permissive_governor()
        >>> signals = create_health_data_signals(fidelity=0.65)
        >>> level, decision = governor.evaluate(signals)
    """
    from .cross_ecosystem_governor import CrossEcosystemPolicy
    
    permissive_policy = {
        "min_semantic_fidelity_for_advisory": 0.50,
        "min_semantic_fidelity_for_constrained": 0.60,
        "min_semantic_fidelity_for_full": 0.75,
        "require_explicit_consent": True,  # Still require consent for safety
        "max_consent_age_hours": 2160.0,  # 90 days
        "max_signal_age_hours": 72.0,
        "warn_signal_age_hours": 48.0,
        "enforce_privacy_boundaries": True,
        "allow_pii_translation": False,  # Still block PII for safety
        "min_context_preservation": 0.50,
        "require_intent_preservation": False,
        "require_translation_consensus": False,
        "enforce_capability_alignment": False,
        "min_capability_overlap": 0.30,
        "max_allowed_level": TranslationTrustLevel.FULL_TRUST,
        "enable_rate_limiting": False,
    }
    permissive_policy.update(policy_overrides)
    
    policy = CrossEcosystemPolicy(**permissive_policy)
    cfg = default_cross_ecosystem_config()
    return CrossEcosystemGovernor(cfg=cfg, policy=policy)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Additional Signal Builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_fitness_tracker_signals(
    source: str = "fitbit",
    target: str = "garmin",
    fidelity: float = 0.82,
) -> CrossEcosystemSignals:
    """
    Create signals for fitness tracker data translation.
    
    Args:
        source: Source ecosystem identifier
        target: Target ecosystem identifier
        fidelity: Semantic fidelity score (0.0 to 1.0)
        
    Returns:
        CrossEcosystemSignals: Configured signals for fitness tracker scenario
        
    Example:
        >>> signals = create_fitness_tracker_signals()
        >>> governor = create_default_governor()
        >>> level, decision = governor.evaluate(signals)
    """
    import time
    from .cross_ecosystem_translation import (
        CrossEcosystemSignals,
        ConsentStatus,
        PrivacyBoundaries,
        ContextPreservation,
    )
    
    return CrossEcosystemSignals(
        desired_level=TranslationTrustLevel.CONSTRAINED_TRUST,
        semantic_fidelity=fidelity,
        source_ecosystem=source,
        target_ecosystem=target,
        translation_path=[source, "ailee_semantic", target],
        consent_status=ConsentStatus(
            user_consent_granted=True,
            data_categories=["activity", "sleep", "heart_rate"],
            consent_timestamp=time.time() - 7200,  # 2 hours ago
            allows_automation=True,
            allows_aggregation=True,
        ),
        privacy_boundaries=PrivacyBoundaries(
            pii_allowed=False,
            location_precision="none",
            temporal_resolution="daily",
        ),
        context_preservation=ContextPreservation(
            intent_maintained=True,
            semantic_loss=0.18,
            capability_alignment=0.82,
            user_activity_preserved=True,
            temporal_context_preserved=True,
        ),
        signal_freshness_ms=900_000.0,  # 15 minutes
        data_category="activity",
        timestamp=time.time(),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def is_translation_authorized(
    level: TranslationTrustLevel,
    min_required: TranslationTrustLevel = TranslationTrustLevel.CONSTRAINED_TRUST,
) -> bool:
    """
    Check if translation is authorized at the required level.
    
    Args:
        level: Actual authorized level from evaluation
        min_required: Minimum required trust level
        
    Returns:
        bool: True if translation is authorized
        
    Example:
        >>> level, decision = governor.evaluate(signals)
        >>> if is_translation_authorized(level):
        ...     perform_translation()
    """
    return level >= min_required


def get_supported_ecosystems() -> list[str]:
    """
    Get list of all known/supported ecosystems.
    
    Returns:
        list[str]: List of ecosystem identifiers
        
    Example:
        >>> ecosystems = get_supported_ecosystems()
        >>> print(f"Supported: {', '.join(ecosystems)}")
    """
    return list(KNOWN_ECOSYSTEMS.keys())


def check_ecosystem_compatibility(
    source: str,
    target: str,
    data_category: str,
) -> dict[str, any]:
    """
    Check if two ecosystems are compatible for a given data category.
    
    Args:
        source: Source ecosystem identifier
        target: Target ecosystem identifier
        data_category: Data category to check (e.g., "heart_rate")
        
    Returns:
        dict: Compatibility information with keys:
            - compatible: bool
            - reason: str (if not compatible)
            - common_categories: list[str]
            - estimated_fidelity: float
            
    Example:
        >>> compat = check_ecosystem_compatibility(
        ...     "ios_healthkit",
        ...     "android_health_connect",
        ...     "heart_rate"
        ... )
        >>> if compat["compatible"]:
        ...     print(f"Estimated fidelity: {compat['estimated_fidelity']:.2f}")
    """
    source_caps = KNOWN_ECOSYSTEMS.get(source)
    target_caps = KNOWN_ECOSYSTEMS.get(target)
    
    if not source_caps:
        return {
            "compatible": False,
            "reason": f"Unknown source ecosystem: {source}",
        }
    
    if not target_caps:
        return {
            "compatible": False,
            "reason": f"Unknown target ecosystem: {target}",
        }
    
    # Check if both support the data category
    if data_category not in source_caps.supported_categories:
        return {
            "compatible": False,
            "reason": f"Source '{source}' doesn't support '{data_category}'",
        }
    
    if data_category not in target_caps.supported_categories:
        return {
            "compatible": False,
            "reason": f"Target '{target}' doesn't support '{data_category}'",
        }
    
    # Calculate compatibility metrics
    common_categories = source_caps.supported_categories & target_caps.supported_categories
    
    if len(source_caps.supported_categories) > 0:
        overlap_ratio = len(common_categories) / len(source_caps.supported_categories)
    else:
        overlap_ratio = 0.0
    
    # Check for known semantic gaps
    fidelity_penalty = 0.0
    for gap in source_caps.semantic_gaps:
        if gap.lower() in data_category.lower():
            fidelity_penalty += 0.15
    
    for gap in target_caps.semantic_gaps:
        if gap.lower() in data_category.lower():
            fidelity_penalty += 0.15
    
    estimated_fidelity = max(0.0, min(1.0, overlap_ratio - fidelity_penalty))
    
    return {
        "compatible": True,
        "common_categories": list(common_categories),
        "estimated_fidelity": estimated_fidelity,
        "source_gaps": source_caps.semantic_gaps,
        "target_gaps": target_caps.semantic_gaps,
        "realtime_possible": (
            source_caps.realtime_available and target_caps.realtime_available
        ),
        "bidirectional": source_caps.bidirectional and target_caps.bidirectional,
    }


def get_trust_level_description(level: TranslationTrustLevel) -> str:
    """
    Get human-readable description of a translation trust level.
    
    Args:
        level: TranslationTrustLevel to describe
        
    Returns:
        str: Description of the trust level
        
    Example:
        >>> desc = get_trust_level_description(TranslationTrustLevel.CONSTRAINED_TRUST)
        >>> print(desc)
    """
    descriptions = {
        TranslationTrustLevel.NO_TRUST: (
            "Do not use translated state. Translation is not trustworthy due to "
            "missing consent, low fidelity, or policy violations."
        ),
        TranslationTrustLevel.ADVISORY_TRUST: (
            "Display only, no automated actions. Translation can inform the user "
            "but cannot trigger automated state changes."
        ),
        TranslationTrustLevel.CONSTRAINED_TRUST: (
            "Limited automation within privacy bounds. Translation is trustworthy "
            "for scoped actions with explicit boundaries and monitoring."
        ),
        TranslationTrustLevel.FULL_TRUST: (
            "Full cross-ecosystem continuity. Translation has high fidelity and "
            "full authorization for seamless state synchronization."
        ),
    }
    return descriptions.get(
        level,
        "Unknown translation trust level"
    )


def create_consent_from_user_preferences(
    allowed_categories: list[str],
    allows_automation: bool = False,
    validity_hours: float = 720.0,  # 30 days default
) -> ConsentStatus:
    """
    Create a ConsentStatus from user preferences.
    
    Args:
        allowed_categories: List of data categories user consents to
        allows_automation: Whether user allows automated actions
        validity_hours: How long consent is valid (hours)
        
    Returns:
        ConsentStatus: Configured consent status
        
    Example:
        >>> consent = create_consent_from_user_preferences(
        ...     allowed_categories=["activity", "heart_rate"],
        ...     allows_automation=True,
        ...     validity_hours=168.0  # 1 week
        ... )
    """
    import time
    from .cross_ecosystem_translation import ConsentStatus
    
    now = time.time()
    return ConsentStatus(
        user_consent_granted=True,
        data_categories=allowed_categories,
        consent_timestamp=now,
        consent_expiry=now + (validity_hours * 3600.0),
        allows_automation=allows_automation,
        allows_aggregation=True,
        allows_third_party=False,
    )


def create_privacy_boundaries_from_level(
    privacy_level: str = "standard"
) -> PrivacyBoundaries:
    """
    Create PrivacyBoundaries from a privacy level preset.
    
    Args:
        privacy_level: One of "minimal", "standard", "strict"
        
    Returns:
        PrivacyBoundaries: Configured privacy boundaries
        
    Example:
        >>> boundaries = create_privacy_boundaries_from_level("strict")
    """
    from .cross_ecosystem_translation import PrivacyBoundaries
    
    presets = {
        "minimal": PrivacyBoundaries(
            pii_allowed=False,
            anonymize_required=False,
            location_precision="none",
            temporal_resolution="daily",
            max_retention_hours=168.0,  # 1 week
            cross_border_allowed=True,
        ),
        "standard": PrivacyBoundaries(
            pii_allowed=False,
            anonymize_required=False,
            location_precision="city",
            temporal_resolution="hourly",
            max_retention_hours=720.0,  # 30 days
            cross_border_allowed=True,
        ),
        "strict": PrivacyBoundaries(
            pii_allowed=False,
            anonymize_required=True,
            location_precision="country",
            temporal_resolution="daily",
            max_retention_hours=168.0,  # 1 week
            cross_border_allowed=False,
            allowed_regions=["US", "EU"],
        ),
    }
    
    return presets.get(privacy_level, presets["standard"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


TRUST_LEVEL_NAMES = {
    TranslationTrustLevel.NO_TRUST: "No Trust",
    TranslationTrustLevel.ADVISORY_TRUST: "Advisory Trust",
    TranslationTrustLevel.CONSTRAINED_TRUST: "Constrained Trust",
    TranslationTrustLevel.FULL_TRUST: "Full Trust",
}

COMMON_DATA_CATEGORIES = [
    "activity",
    "heart_rate",
    "sleep",
    "nutrition",
    "location",
    "stress",
    "mindfulness",
    "environmental_audio",
]

HEALTH_DATA_CATEGORIES = [
    "activity",
    "heart_rate",
    "blood_pressure",
    "blood_glucose",
    "sleep",
    "nutrition",
    "hydration",
    "body_temperature",
    "respiratory_rate",
    "oxygen_saturation",
]

FITNESS_DATA_CATEGORIES = [
    "steps",
    "distance",
    "calories",
    "active_minutes",
    "heart_rate",
    "workout",
    "elevation",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Package Metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


__doc_url__ = "https://github.com/your-org/ailee-cross-ecosystem/docs"
__source_url__ = "https://github.com/your-org/ailee-cross-ecosystem"
__bug_tracker_url__ = "https://github.com/your-org/ailee-cross-ecosystem/issues"

__description__ = (
    "AILEE Cross-Ecosystem Translation Domain: Governance for semantic state "
    "and intent translation between incompatible software-hardware ecosystems."
)

__keywords__ = [
    "ailee",
    "governance",
    "cross-ecosystem",
    "translation",
    "consent",
    "privacy",
    "semantic-fidelity",
    "health-data",
    "wearables",
    "interoperability",
]
