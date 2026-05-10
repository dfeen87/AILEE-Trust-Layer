"""
AILEE Trust Layer — Light Transition Domain

Governance for optical, photonic, laser, fiber, free-space, and light-carried
data signals. This module governs whether light-transition data is trustworthy
enough to act upon; it does not implement physical transport or bypass the
speed of light.
"""

from .light_transition import (
    LIGHT_CLOCK_SYNCHRONIZATION,
    OPTICAL_PATH_RELIABILITY,
    PHOTONIC_SIGNAL_INTEGRITY,
    SPEED_OF_LIGHT_M_PER_S,
    LightTransitionConfig,
    LightTransitionControlAction,
    LightTransitionControlDomain,
    LightTransitionDecision,
    LightTransitionEvent,
    LightTransitionGovernor,
    LightTransitionHealthStatus,
    LightTransitionPolicy,
    LightTransitionSignals,
    LightTransitionTrustLevel,
    LightTrustLevel,
    OpticalReading,
    PropagationMedium,
    create_degraded_signals,
    create_default_governor,
    create_example_signals,
    create_light_transition_governor,
    create_permissive_governor,
    create_strict_governor,
    default_light_transition_config,
    export_events_to_dict,
    get_decision_history,
    get_events,
    get_health,
    get_metrics,
    get_subsystem_health,
    validate_light_transition_signals,
    validate_signals,
)

__all__ = [
    "LIGHT_CLOCK_SYNCHRONIZATION",
    "OPTICAL_PATH_RELIABILITY",
    "PHOTONIC_SIGNAL_INTEGRITY",
    "SPEED_OF_LIGHT_M_PER_S",
    "LightTransitionConfig",
    "LightTransitionControlAction",
    "LightTransitionControlDomain",
    "LightTransitionDecision",
    "LightTransitionEvent",
    "LightTransitionGovernor",
    "LightTransitionHealthStatus",
    "LightTransitionPolicy",
    "LightTransitionSignals",
    "LightTransitionTrustLevel",
    "LightTrustLevel",
    "OpticalReading",
    "PropagationMedium",
    "create_degraded_signals",
    "create_default_governor",
    "create_example_signals",
    "create_light_transition_governor",
    "create_permissive_governor",
    "create_strict_governor",
    "default_light_transition_config",
    "export_events_to_dict",
    "get_decision_history",
    "get_events",
    "get_health",
    "get_metrics",
    "get_subsystem_health",
    "validate_light_transition_signals",
    "validate_signals",
]

__version__ = "4.7.0"
__status__ = "Production/Stable"
