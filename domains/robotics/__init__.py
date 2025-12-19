"""
AILEE Trust Layer — ROBOTICS Domain
Production-grade governance for robotic safety, autonomy, and human interaction.

This module exposes the public API for the ROBOTICS domain while
intentionally hiding internal implementation details.

Design principles:
- Stable imports
- Explicit exports
- No side effects at import time
- Domain-level semantics (not device-level)
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Core Enums
# -----------------------------------------------------------------------------

from .robotics import (
    RobotType,
    OperationalMode,
    ActionType,
    SafetyDecision,
)

# -----------------------------------------------------------------------------
# Core Data Structures
# -----------------------------------------------------------------------------

from .robotics import (
    WorkspaceState,
    PerformanceMetrics,
    SensorReading,
    UncertaintyEstimate,
    ValidationResult,
    RoboticsSignals,
    AdaptiveStrategy,
    RoboticsEvent,
)

# -----------------------------------------------------------------------------
# Governance Configuration & Results
# -----------------------------------------------------------------------------

from .robotics import (
    RoboticsGovernancePolicy,
    RoboticsDecisionResult,
)

# -----------------------------------------------------------------------------
# Main Governor
# -----------------------------------------------------------------------------

from .robotics import (
    RoboticsGovernor,
)

# -----------------------------------------------------------------------------
# Convenience & Utilities
# -----------------------------------------------------------------------------

from .robotics import (
    default_robotics_config,
    create_robotics_governor,
    validate_robotics_signals,
)

# -----------------------------------------------------------------------------
# Domain Descriptor
# -----------------------------------------------------------------------------

class RoboticsDomain:
    """
    Domain descriptor for robotics governance.

    This class is used for discovery, documentation, and optional
    framework-level registration. It does not affect runtime behavior.
    """

    name: str = "ROBOTICS"

    def describe(self) -> dict:
        return {
            "domain": self.name,
            "focus": [
                "Robotic action safety validation",
                "Human-aware autonomy governance",
                "Multi-sensor consensus",
                "Uncertainty-aware decision gating",
                "Fail-safe enforcement",
                "Auditable robotic behavior",
            ],
            "applications": [
                "Industrial robotics",
                "Collaborative robots (cobots)",
                "Autonomous vehicles",
                "Service robots",
                "Surgical and medical robotics",
                "Research and experimental platforms",
            ],
            "ailee_role": (
                "Provide a deterministic trust and safety layer for robotic "
                "decision-making without modifying control, planning, or "
                "perception algorithms."
            ),
            "implementation_status": "production_grade_v1.0.0",
        }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # Enums
    "RobotType",
    "OperationalMode",
    "ActionType",
    "SafetyDecision",

    # Data structures
    "WorkspaceState",
    "PerformanceMetrics",
    "SensorReading",
    "UncertaintyEstimate",
    "ValidationResult",
    "RoboticsSignals",
    "AdaptiveStrategy",
    "RoboticsEvent",

    # Governance
    "RoboticsGovernancePolicy",
    "RoboticsDecisionResult",
    "RoboticsGovernor",

    # Utilities
    "default_robotics_config",
    "create_robotics_governor",
    "validate_robotics_signals",

    # Domain descriptor
    "RoboticsDomain",
]
