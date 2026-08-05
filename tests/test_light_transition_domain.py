# Licensed under the PolyForm Noncommercial License 1.0.0
from ailee.domains.light_transition import (
    LightTransitionControlAction,
    LightTransitionControlDomain,
    LightTransitionHealthStatus,
    LightTransitionSignals,
    LightTransitionTrustLevel,
    OpticalReading,
    PropagationMedium,
    REFRACTIVE_INDEX_BY_MEDIUM,
    create_default_governor,
    create_degraded_signals,
    create_example_signals,
    create_permissive_governor,
    create_strict_governor,
    get_decision_history,
    get_events,
    get_health,
    get_metrics,
    get_subsystem_health,
    validate_light_transition_signals,
)


def test_light_transition_enums():
    assert LightTransitionTrustLevel.NO_ACTION == 0
    assert LightTransitionHealthStatus.OPTIMAL == "OPTIMAL"
    assert LightTransitionControlDomain.PHOTONIC_SIGNAL_INTEGRITY.value == "PHOTONIC_SIGNAL_INTEGRITY"
    assert LightTransitionControlAction.ACCEPT_FRAME.value == "ACCEPT_FRAME"


def test_light_transition_factory_and_monitor():
    gov = create_strict_governor()
    assert gov is not None
    assert get_health(gov) is not None
    assert isinstance(get_subsystem_health(gov), dict)
    assert isinstance(get_metrics(gov), dict)
    assert isinstance(get_events(gov), list)
    assert isinstance(get_decision_history(gov), list)


def test_light_transition_evaluates_clean_signal():
    gov = create_default_governor()
    decision = gov.evaluate(create_example_signals())
    assert decision.authorized_level >= LightTransitionTrustLevel.GUARDED
    assert decision.actionable is True
    assert decision.safety_flags == []
    assert get_metrics(gov)["decisions_total"] == 1


def test_light_transition_flags_physics_boundary_violation():
    gov = create_permissive_governor()
    signals = LightTransitionSignals(
        control_domain=LightTransitionControlDomain.PHOTONIC_SIGNAL_INTEGRITY,
        proposed_action=LightTransitionControlAction.ACCEPT_FRAME,
        ai_value=0.95,
        ai_confidence=0.95,
        optical_readings=[
            OpticalReading(0.95, 1.0, "sensor_a"),
            OpticalReading(0.94, 1.0, "sensor_b"),
            OpticalReading(0.96, 1.0, "sensor_c"),
        ],
        medium=PropagationMedium.FREE_SPACE,
        distance_m=299_792_458.0,
        measured_time_of_flight_ns=500_000_000.0,
    )
    decision = gov.evaluate(signals)
    assert any(flag.startswith("physics_bound_violation") for flag in decision.safety_flags)
    assert decision.actionable is False


def test_light_transition_uses_medium_aware_speed_limit():
    gov = create_permissive_governor()
    signals = LightTransitionSignals(
        control_domain=LightTransitionControlDomain.PHOTONIC_SIGNAL_INTEGRITY,
        proposed_action=LightTransitionControlAction.ACCEPT_FRAME,
        ai_value=0.95,
        ai_confidence=0.95,
        optical_readings=[
            OpticalReading(0.95, 1.0, "sensor_a"),
            OpticalReading(0.94, 1.0, "sensor_b"),
            OpticalReading(0.96, 1.0, "sensor_c"),
        ],
        medium=PropagationMedium.FIBER,
        distance_m=299_792_458.0,
        measured_time_of_flight_ns=1_250_000_000.0,
    )

    decision = gov.evaluate(signals)

    assert REFRACTIVE_INDEX_BY_MEDIUM[PropagationMedium.FIBER] == 1.467
    assert decision.metadata["dynamic_speed_limit_fraction_c"] < 0.69
    assert any(
        flag.startswith("physics_bound_violation:") and "for FIBER" in flag
        for flag in decision.safety_flags
    )


def test_light_transition_accepts_refractive_index_override():
    gov = create_permissive_governor()
    signals = LightTransitionSignals(
        control_domain=LightTransitionControlDomain.PHOTONIC_SIGNAL_INTEGRITY,
        proposed_action=LightTransitionControlAction.ACCEPT_FRAME,
        ai_value=0.95,
        ai_confidence=0.95,
        optical_readings=[
            OpticalReading(0.95, 1.0, "sensor_a"),
            OpticalReading(0.94, 1.0, "sensor_b"),
            OpticalReading(0.96, 1.0, "sensor_c"),
        ],
        medium=PropagationMedium.FIBER,
        distance_m=299_792_458.0,
        measured_time_of_flight_ns=1_250_000_000.0,
        refractive_index_override=1.0,
    )

    decision = gov.evaluate(signals)

    assert decision.metadata["refractive_index"] == 1.0
    assert decision.metadata["dynamic_speed_limit_fraction_c"] == 1.005
    assert not any(
        flag.startswith("physics_bound_violation")
        for flag in decision.safety_flags
    )


def test_light_transition_validation_and_degraded_signal():
    valid, issues = validate_light_transition_signals(create_degraded_signals())
    assert valid is True
    assert issues == []

    invalid = create_example_signals()
    invalid.ai_confidence = 1.5
    valid, issues = validate_light_transition_signals(invalid)
    assert valid is False
    assert "ai_confidence must be in [0.0, 1.0]" in issues
