"""
AILEE Trust Layer — Automotive Autonomy Governance Domain

This package provides a governance-only authorization layer for
autonomous driving systems.

IMPORTANT:
- This domain does NOT perform perception, planning, or control.
- It does NOT command vehicle actuation.
- It ONLY determines the maximum allowed autonomy level at runtime.

The outputs of this domain are intended to be used as an
AUTHORIZATION CEILING by an external autonomy stack.

Designed for:
- Deterministic behavior
- Auditability and black-box logging
- Safe-by-default integration
- ISO 26262 / UL 4600 aligned workflows
"""

from .autonomy_governor import (
    # Core enums / signals
    AutonomyLevel,
    AutonomySignals,

    # Governor + policy
    AutonomyGovernor,
    AutonomyGovernancePolicy,
    default_autonomy_config,

    # Safety & domain models
    SafetyMonitorSignals,
    OperationalDesignDomain,
    DriverState,
    SystemHealth,

    # Degradation & confidence
    DegradationStrategy,
    ConfidenceTracker,

    # Scenario & events
    ScenarioPolicy,
    GovernanceEvent,

    # Test utilities
    create_test_signals,
)

__all__ = [
    # Core
    "AutonomyLevel",
    "AutonomySignals",

    # Governance
    "AutonomyGovernor",
    "AutonomyGovernancePolicy",
    "default_autonomy_config",

    # Safety & domain context
    "SafetyMonitorSignals",
    "OperationalDesignDomain",
    "DriverState",
    "SystemHealth",

    # Degradation & confidence
    "DegradationStrategy",
    "ConfidenceTracker",

    # Scenario & logging
    "ScenarioPolicy",
    "GovernanceEvent",

    # Utilities
    "create_test_signals",
]
