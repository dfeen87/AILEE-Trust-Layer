"""
AILEE Trust Layer — Power Grid Governance Domain

This package provides a governance-only authorization layer for
electric power grid and energy infrastructure systems.

IMPORTANT:
- This domain does NOT perform power dispatch, control, or actuation.
- It does NOT command breakers, generators, or storage systems.
- It ONLY determines the maximum allowed operational mode or authority
  level at runtime based on safety, stability, and system health.

The outputs of this domain are intended to be used as an
AUTHORIZATION CEILING by external grid control and energy management systems.

Designed for:
- Deterministic, fail-safe governance
- Auditability and black-box logging
- Stability-first operation (avoid oscillations and unsafe transitions)
- Integration with AI-assisted grid management
- Regulatory and critical-infrastructure compliance workflows
  (e.g., NERC, IEC 61850, ISO 55000)
"""

from .grid_governor import (
    # Core enums / signals
    GridAuthorityLevel,
    GridSignals,

    # Governor + policy
    GridGovernor,
    GridGovernancePolicy,
    default_grid_config,

    # Safety & domain models
    GridSafetySignals,
    GridOperationalDomain,
    GridOperatorState,
    GridSystemHealth,

    # Degradation & confidence
    GridDegradationStrategy,
    GridConfidenceTracker,

    # Scenario & events
    GridScenarioPolicy,
    GridGovernanceEvent,

    # Test utilities
    create_test_grid_signals,
)

__all__ = [
    # Core
    "GridAuthorityLevel",
    "GridSignals",

    # Governance
    "GridGovernor",
    "GridGovernancePolicy",
    "default_grid_config",

    # Safety & domain context
    "GridSafetySignals",
    "GridOperationalDomain",
    "GridOperatorState",
    "GridSystemHealth",

    # Degradation & confidence
    "GridDegradationStrategy",
    "GridConfidenceTracker",

    # Scenario & logging
    "GridScenarioPolicy",
    "GridGovernanceEvent",

    # Utilities
    "create_test_grid_signals",
]
