"""
AILEE Governance Domain — v1.0.0

A layered governance pipeline for evaluating trust and authorization in AI systems.

This module provides:
- Authority verification and validation
- Jurisdictional scope enforcement
- Temporal validity tracking
- Delegation chain verification
- Graduated trust level determination
- Complete audit trails

Quick Start:
    >>> from ailee_governance import GovernanceGovernor, GovernanceConfig, GovernanceTrustLevel
    >>> 
    >>> config = GovernanceConfig()
    >>> governor = GovernanceGovernor(config)
    >>> 
    >>> result = governor.evaluate(
    ...     source="election_authority_xyz",
    ...     jurisdiction="state:PA",
    ...     authority_level="certified_official",
    ...     mandate="voter_registration_update",
    ...     consent_proof="signed_consent_hash_abc"
    ... )
    >>> 
    >>> if result.authorized_level >= GovernanceTrustLevel.CONSTRAINED_TRUST:
    ...     print("Authorized for constrained action")

For detailed documentation, see: https://github.com/your-org/ailee-governance
"""

from .governance import (
    # Core Classes
    GovernanceGovernor,
    GovernanceConfig,
    GovernanceSignal,
    GovernanceDecision,
    
    # Enums
    GovernanceTrustLevel,
    AuthorityStatus,
    ScopeStatus,
    TemporalStatus,
)

# Version info
__version__ = "1.0.0"
__author__ = "AILEE Project"
__license__ = "MIT"

# Public API
__all__ = [
    # Core Classes
    "GovernanceGovernor",
    "GovernanceConfig",
    "GovernanceSignal",
    "GovernanceDecision",
    
    # Enums
    "GovernanceTrustLevel",
    "AuthorityStatus",
    "ScopeStatus",
    "TemporalStatus",
    
    # Version
    "__version__",
]


# Convenience functions for common use cases

def create_default_governor(**config_overrides) -> GovernanceGovernor:
    """
    Create a GovernanceGovernor with default configuration.
    
    Args:
        **config_overrides: Optional configuration parameters to override defaults
        
    Returns:
        GovernanceGovernor: Configured governor instance
        
    Example:
        >>> governor = create_default_governor(require_consent=False)
        >>> result = governor.evaluate(source="test", mandate="test_op")
    """
    config = GovernanceConfig(**config_overrides)
    return GovernanceGovernor(config)


def create_strict_governor(**config_overrides) -> GovernanceGovernor:
    """
    Create a GovernanceGovernor with strict security configuration.
    
    Enables all security features:
    - Requires mandate, consent, and jurisdiction
    - Enforces scope boundaries
    - Blocks revoked sources
    - Requires temporal bounds
    - Enables full audit metadata
    
    Args:
        **config_overrides: Optional configuration parameters to override defaults
        
    Returns:
        GovernanceGovernor: Strictly configured governor instance
        
    Example:
        >>> governor = create_strict_governor()
        >>> result = governor.evaluate(
        ...     source="authority",
        ...     jurisdiction="state:PA",
        ...     authority_level="certified_official",
        ...     mandate="update_records",
        ...     consent_proof="consent_hash",
        ...     valid_from=time.time(),
        ...     valid_until=time.time() + 86400
        ... )
    """
    strict_config = {
        "require_mandate": True,
        "require_consent": True,
        "require_jurisdiction": True,
        "require_temporal_bounds": True,
        "enforce_scope_boundaries": True,
        "block_revoked_sources": True,
        "enable_audit_metadata": True,
        "track_decision_history": True,
        "allow_cross_jurisdictional_delegation": False,
    }
    strict_config.update(config_overrides)
    config = GovernanceConfig(**strict_config)
    return GovernanceGovernor(config)


def create_permissive_governor(**config_overrides) -> GovernanceGovernor:
    """
    Create a GovernanceGovernor with permissive configuration for testing/development.
    
    Relaxes many requirements:
    - Does not require mandate, consent, or jurisdiction
    - Does not enforce scope boundaries
    - Does not require temporal bounds
    - Allows cross-jurisdictional delegation
    
    WARNING: Not recommended for production use!
    
    Args:
        **config_overrides: Optional configuration parameters to override defaults
        
    Returns:
        GovernanceGovernor: Permissively configured governor instance
        
    Example:
        >>> governor = create_permissive_governor()
        >>> result = governor.evaluate(source="test_source")
    """
    permissive_config = {
        "require_mandate": False,
        "require_consent": False,
        "require_jurisdiction": False,
        "require_temporal_bounds": False,
        "enforce_scope_boundaries": False,
        "allow_cross_jurisdictional_delegation": True,
    }
    permissive_config.update(config_overrides)
    config = GovernanceConfig(**permissive_config)
    return GovernanceGovernor(config)


# Module-level constants for common trust level checks

TRUST_LEVEL_NAMES = {
    GovernanceTrustLevel.NO_TRUST: "No Trust",
    GovernanceTrustLevel.ADVISORY_TRUST: "Advisory Trust",
    GovernanceTrustLevel.CONSTRAINED_TRUST: "Constrained Trust",
    GovernanceTrustLevel.FULL_TRUST: "Full Trust",
}

TRUST_LEVEL_DESCRIPTIONS = {
    GovernanceTrustLevel.NO_TRUST: (
        "Signal exists but cannot be used for any action. "
        "This may indicate missing credentials, revoked authority, or invalid scope."
    ),
    GovernanceTrustLevel.ADVISORY_TRUST: (
        "Informational only - no action permitted. "
        "Signal can inform decisions but cannot authorize actions."
    ),
    GovernanceTrustLevel.CONSTRAINED_TRUST: (
        "Limited, scoped execution within defined boundaries. "
        "Actions are permitted within specific constraints and jurisdictional limits."
    ),
    GovernanceTrustLevel.FULL_TRUST: (
        "Fully authorized action within defined mandate. "
        "Signal has complete authority to perform operations within its scope."
    ),
}


def is_actionable(decision: GovernanceDecision) -> bool:
    """
    Check if a governance decision permits action.
    
    Args:
        decision: GovernanceDecision to evaluate
        
    Returns:
        bool: True if action is permitted (CONSTRAINED_TRUST or higher)
        
    Example:
        >>> if is_actionable(result):
        ...     perform_action()
    """
    return decision.actionable and decision.authorized_level >= GovernanceTrustLevel.CONSTRAINED_TRUST


def requires_full_trust(decision: GovernanceDecision) -> bool:
    """
    Check if a decision has full trust level.
    
    Args:
        decision: GovernanceDecision to evaluate
        
    Returns:
        bool: True if decision has FULL_TRUST level
        
    Example:
        >>> if requires_full_trust(result):
        ...     perform_unrestricted_action()
    """
    return decision.authorized_level == GovernanceTrustLevel.FULL_TRUST


def get_trust_level_description(level: GovernanceTrustLevel) -> str:
    """
    Get human-readable description of a trust level.
    
    Args:
        level: GovernanceTrustLevel to describe
        
    Returns:
        str: Description of the trust level
        
    Example:
        >>> desc = get_trust_level_description(GovernanceTrustLevel.CONSTRAINED_TRUST)
        >>> print(desc)
    """
    return TRUST_LEVEL_DESCRIPTIONS.get(
        level,
        "Unknown trust level"
    )


# Package metadata
__doc_url__ = "https://github.com/your-org/ailee-governance/docs"
__source_url__ = "https://github.com/your-org/ailee-governance"
__bug_tracker_url__ = "https://github.com/your-org/ailee-governance/issues"
