"""
AILEE Trust Layer — Telecommunications Domain

Deterministic governance for communication systems operating under
latency, loss, bandwidth constraints, and uncertainty.

This domain does NOT implement networking protocols or transport logic.
It governs whether received communications are trustworthy enough
to act upon.
"""

from .telecommunications import (
    # Trust levels
    CommunicationTrustLevel,
    TrustLevel,

    # Core signals
    TelecomSignals,
    LinkQualitySignals,
    NetworkOperationalDomain,
    SystemHealth,
    RedundancyState,

    # Governance
    TelecomGovernancePolicy,
    ScenarioPolicy,
    TelecomGovernor,

    # Monitoring & events
    GovernanceEvent,
    ConfidenceTracker,

    # Defaults & helpers
    default_telecom_config,
    create_default_governor,
    create_example_signals,
    create_degraded_signals,
    validate_signals,
    export_events_to_dict,
)

__all__ = [
    # Trust levels
    "CommunicationTrustLevel",
    "TrustLevel",

    # Signals
    "TelecomSignals",
    "LinkQualitySignals",
    "NetworkOperationalDomain",
    "SystemHealth",
    "RedundancyState",

    # Governance
    "TelecomGovernancePolicy",
    "ScenarioPolicy",
    "TelecomGovernor",

    # Events & tracking
    "GovernanceEvent",
    "ConfidenceTracker",

    # Helpers
    "default_telecom_config",
    "create_default_governor",
    "create_example_signals",
    "create_degraded_signals",
    "validate_signals",
    "export_events_to_dict",
]

__version__ = "2.0.0"
__status__ = "Production/Stable"
