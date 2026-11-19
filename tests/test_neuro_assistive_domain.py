# Licensed under the PolyForm Noncommercial License 1.0.0

from dataclasses import replace

import pytest
from ailee import AileeConfig
from ailee.domains.neuro_assistive import (
    AssistanceOutcome,
    CognitiveState,
    ConsentRecord,
    ConsentStatus,
    NeuroGovernor,
    NeuroSignals,
    get_health,
    get_subsystem_health,
    get_metrics,
    get_events,
    get_decision_history,
    create_strict_governor,
    create_permissive_governor,
    create_default_governor,
    NeuroTrustLevel,
    NeuroHealthStatus,
    NeuroControlDomain,
    NeuroControlAction,
)

def test_neuro_assistive_enums():
    assert NeuroTrustLevel.NO_ACTION == 0
    assert NeuroHealthStatus.OPTIMAL == "OPTIMAL"

def test_neuro_assistive_factory_and_monitor():
    gov = create_strict_governor()
    assert gov is not None
    assert get_health(gov) is not None
    assert isinstance(get_subsystem_health(gov), dict)
    assert isinstance(get_metrics(gov), dict)
    assert isinstance(get_events(gov), list)
    assert isinstance(get_decision_history(gov), list)

def test_neuro_assistive_backward_compatibility():
    pass


def test_evaluate_uses_governed_value_and_withholds_unvalidated_candidate():
    governor = NeuroGovernor(
        cfg=AileeConfig(accept_threshold=0.50, borderline_low=0.50, enable_grace=False)
    )
    signals = NeuroSignals(
        assistance_trust_score=0.94,
        interpretation_confidence=0.94,
        user_input="Show the next item",
        interpreted_intent="show_next_item",
        input_modality="text",
        current_cognitive_state=CognitiveState.BASELINE_STABLE,
        feature_requested="show_item",
        consent_records={"show_item": ConsentRecord("show_item", ConsentStatus.GRANTED)},
        timestamp=100.0,
    )
    approved = governor.evaluate(signals)
    assert approved.assistance_authorized is True
    assert approved.decision_outcome is AssistanceOutcome.FULL_APPROVED
    assert approved.validated_trust_score == approved.ailee_result.value == 0.94
    assert not approved.ailee_result.used_fallback

    denied = governor.evaluate(
        replace(signals, assistance_trust_score=0.30, timestamp=101.0)
    )
    assert denied.ailee_result.used_fallback
    assert denied.validated_trust_score == denied.ailee_result.value == 0.94
    assert denied.assistance_authorized is False
    assert denied.decision_outcome is AssistanceOutcome.DENIED
    assert get_decision_history(governor) == [approved, denied]
