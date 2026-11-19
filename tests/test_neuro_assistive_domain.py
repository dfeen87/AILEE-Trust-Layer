# Licensed under the PolyForm Noncommercial License 1.0.0

from dataclasses import replace
from math import nextafter

import pytest
from ailee import AileeConfig
from ailee.domains.neuro_assistive import (
    AssistanceLevel,
    AssistanceOutcome,
    CognitiveLoadMetrics,
    CognitiveState,
    ConsentRecord,
    ConsentStatus,
    NeuroAssistivePolicy,
    NeuroGovernor,
    NeuroSignals,
    SessionMetrics,
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


@pytest.fixture
def governor():
    return NeuroGovernor(
        cfg=AileeConfig(accept_threshold=0.50, borderline_low=0.50, enable_grace=False)
    )


@pytest.fixture
def signals():
    return NeuroSignals(
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


def test_evaluate_uses_governed_value_and_withholds_unvalidated_candidate(governor, signals):
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
    assert denied.assistance_level is AssistanceLevel.NO_ASSISTANCE
    assert denied.decision_outcome is AssistanceOutcome.DENIED
    assert denied.recommendation == "assistance_denied_insufficient_trust"
    assert denied.assistance_constraints.max_suggestions == 0
    assert get_decision_history(governor) == [approved, denied]


@pytest.mark.parametrize(
    "state, level, outcome, authorized, recommendation, suggestions, complexity",
    [
        (CognitiveState.ACUTE_DISTRESS, AssistanceLevel.SIMPLE_PROMPTS,
         AssistanceOutcome.PROMPT_APPROVED, True, "simple_prompts_approved", 2, "simple"),
        (CognitiveState.IMPAIRED_CAPACITY, AssistanceLevel.GUIDED_OPTIONS,
         AssistanceOutcome.GUIDANCE_APPROVED, True, "guided_options_approved", 3, "moderate"),
        (CognitiveState.UNKNOWN, AssistanceLevel.PASSIVE_MONITORING,
         AssistanceOutcome.OBSERVE_ONLY, False, "observe_only_no_assistance", 0, None),
        (CognitiveState.BASELINE_STABLE, AssistanceLevel.FULL_ASSISTANCE,
         AssistanceOutcome.FULL_APPROVED, True, "full_assistance_approved", 5, "full"),
    ],
)
def test_evaluate_reports_effective_authority(
    governor, signals, state, level, outcome, authorized, recommendation, suggestions, complexity
):
    decision = governor.evaluate(replace(signals, current_cognitive_state=state))

    assert decision.assistance_level is level
    assert decision.decision_outcome is outcome
    assert decision.assistance_authorized is authorized
    assert decision.recommendation == recommendation
    assert decision.assistance_constraints.max_suggestions == suggestions
    assert decision.assistance_constraints.complexity_limit == complexity
    assert decision.assistance_constraints.require_user_confirmation is (
        level is AssistanceLevel.SIMPLE_PROMPTS
    )
    assert governor.get_last_event().decision_outcome is outcome


@pytest.mark.parametrize(
    "state, expected",
    [
        (CognitiveState.ACUTE_DISTRESS, AssistanceLevel.SIMPLE_PROMPTS),
        (CognitiveState.UNKNOWN, AssistanceLevel.PASSIVE_MONITORING),
    ],
)
def test_evaluate_applies_tighter_current_ceiling_during_hysteresis(
    governor, signals, state, expected
):
    assert governor.evaluate(signals).assistance_level is AssistanceLevel.FULL_ASSISTANCE
    decision = governor.evaluate(
        replace(signals, current_cognitive_state=state, timestamp=signals.timestamp + 1)
    )
    assert governor.state_tracker.get_current_state() is CognitiveState.BASELINE_STABLE
    assert decision.assistance_level is expected


@pytest.mark.parametrize(
    "threshold, lower_level, at_level",
    [
        (0.60, AssistanceLevel.NO_ASSISTANCE, AssistanceLevel.PASSIVE_MONITORING),
        (0.70, AssistanceLevel.PASSIVE_MONITORING, AssistanceLevel.SIMPLE_PROMPTS),
        (0.80, AssistanceLevel.SIMPLE_PROMPTS, AssistanceLevel.GUIDED_OPTIONS),
        (0.90, AssistanceLevel.GUIDED_OPTIONS, AssistanceLevel.FULL_ASSISTANCE),
    ],
)
def test_evaluate_assistance_threshold_boundaries(governor, signals, threshold, lower_level, at_level):
    def evaluate(score):
        return governor.evaluate(
            replace(signals, assistance_trust_score=score, interpretation_confidence=score)
        )

    assert evaluate(nextafter(threshold, 0.0)).assistance_level is lower_level
    assert evaluate(threshold).assistance_level is at_level


@pytest.fixture
def emergency_signals(signals):
    return replace(
        signals,
        current_cognitive_state=CognitiveState.CRISIS_MODE,
        cognitive_load_metrics=CognitiveLoadMetrics(estimated_load=0.93, load_confidence=0.94),
    )


@pytest.mark.parametrize(
    "consent",
    [
        None,
        ConsentRecord("show_item", ConsentStatus.DENIED),
        ConsentRecord("show_item", ConsentStatus.GRANTED, expires_at=99.0),
        ConsentRecord("show_item", ConsentStatus.REVOKED),
        ConsentRecord("show_item", ConsentStatus.GRANTED, revoked_at=99.0),
        ConsentRecord("show_item", ConsentStatus.GRANTED, expires_at=float("nan")),
        ConsentRecord("other_item", ConsentStatus.GRANTED),
        ConsentRecord("show_item", "unknown"),
    ],
    ids=["missing", "denied", "expired", "revoked", "effective-revocation",
         "invalid-expiry", "wrong-feature", "unknown-status"],
)
def test_consent_denies_before_emergency(governor, emergency_signals, consent):
    decision = governor.evaluate(
        replace(emergency_signals, consent_records={"show_item": consent} if consent else {})
    )
    assert decision.assistance_authorized is False
    assert decision.assistance_level is AssistanceLevel.NO_ASSISTANCE
    assert decision.decision_outcome is AssistanceOutcome.DENIED
    assert decision.recommendation == "assistance_denied_consent_required"
    assert decision.requires_consent
    assert decision.ailee_result is None
    assert governor.state_tracker.get_current_state() is CognitiveState.UNKNOWN


@pytest.mark.parametrize("state", [CognitiveState.CRISIS_MODE, CognitiveState.BASELINE_STABLE])
def test_eligible_emergency_simplification_is_minimal(governor, emergency_signals, state):
    decision = governor.evaluate(replace(emergency_signals, current_cognitive_state=state))
    assert decision.assistance_authorized is True
    assert decision.assistance_level is AssistanceLevel.EMERGENCY_SIMPLIFICATION
    assert decision.decision_outcome is AssistanceOutcome.EMERGENCY_SIMPLIFICATION_ACTIVE
    assert decision.recommendation == "emergency_simplification_minimal_interaction"
    assert decision.assistance_constraints.max_suggestions == 1
    assert decision.assistance_constraints.complexity_limit == "minimal"
    assert decision.assistance_constraints.allowed_modalities == ["text"]
    assert decision.assistance_constraints.interaction_timeout_seconds == 10.0


@pytest.mark.parametrize("fallback", [False, True], ids=["policy-ceiling", "hard-envelope"])
def test_emergency_respects_hard_authority_and_numeric_bounds(emergency_signals, fallback):
    governor = NeuroGovernor(
        cfg=AileeConfig(hard_max=0.80 if fallback else 1.0),
        policy=NeuroAssistivePolicy(
            max_assistance_level=(
                AssistanceLevel.FULL_ASSISTANCE if fallback else AssistanceLevel.NO_ASSISTANCE
            )
        ),
    )
    decision = governor.evaluate(emergency_signals)
    assert decision.assistance_authorized is False
    assert decision.assistance_level is AssistanceLevel.NO_ASSISTANCE
    assert decision.decision_outcome is AssistanceOutcome.DENIED
    assert decision.assistance_constraints.max_suggestions == 0


@pytest.mark.parametrize("duration", [31.0, float("nan")], ids=["over-limit", "invalid"])
def test_temporal_break_blocks_emergency_assistance(governor, emergency_signals, duration):
    session = SessionMetrics(
        session_duration_minutes=duration,
        time_since_last_break_minutes=5.0,
        assistance_events_count=0,
        user_initiated_requests=1,
        system_proactive_suggestions=0,
        user_acceptance_rate=1.0,
        user_rejection_rate=0.0,
        consecutive_sessions_today=1,
        total_interaction_minutes_today=31.0,
    )
    decision = governor.evaluate(replace(emergency_signals, session_metrics=session))
    assert decision.requires_break
    assert decision.assistance_authorized is False
    assert decision.assistance_level is AssistanceLevel.NO_ASSISTANCE
    assert decision.decision_outcome is AssistanceOutcome.BREAK_REQUIRED
    assert decision.recommendation == "assistance_paused_break_required"
    assert decision.assistance_constraints.max_suggestions == 0


def test_crisis_ceiling_cannot_release_ordinary_assistance(governor, signals):
    decision = governor.evaluate(replace(signals, current_cognitive_state=CognitiveState.CRISIS_MODE))
    assert decision.assistance_level is AssistanceLevel.PASSIVE_MONITORING
    assert decision.assistance_authorized is False
    assert decision.decision_outcome is AssistanceOutcome.OBSERVE_ONLY
    assert decision.assistance_constraints.max_suggestions == 0


def test_malformed_trust_score_cannot_commit_cognitive_state(governor, emergency_signals):
    with pytest.raises(ValueError, match="finite"):
        governor.evaluate(replace(emergency_signals, assistance_trust_score=float("nan")))
    assert governor.state_tracker.get_current_state() is CognitiveState.UNKNOWN
    assert not get_decision_history(governor)
    assert governor.pipeline.last_result is None


def test_explain_decision_reports_accepted_pipeline_statuses(governor, signals):
    decision = governor.evaluate(
        replace(
            signals,
            cognitive_load_metrics=CognitiveLoadMetrics(estimated_load=0.06, load_confidence=0.94),
        )
    )
    explanation = governor.explain_decision(decision)

    assert "NEURO-ASSISTIVE DECISION EXPLANATION" in explanation
    assert "DECISION: FULL_APPROVED" in explanation
    assert "Validated Trust Score: 0.94 / 1.00" in explanation
    assert "WHAT MATTERED MOST:" in explanation
    assert "• AILEE Pipeline Status: ACCEPTED" in explanation
    assert "  → Grace: SKIPPED" in explanation
    assert "  → Consensus: PASS" in explanation
    assert "Fallback" not in explanation


@pytest.mark.parametrize(
    "use_history, fallback_score", [(False, "0.00"), (True, "0.94")],
    ids=["empty-history", "historical-value"],
)
def test_explain_decision_labels_rejected_pipeline_fallback(
    governor, signals, use_history, fallback_score
):
    if use_history:
        governor.evaluate(signals)
    decision = governor.evaluate(
        replace(signals, assistance_trust_score=0.30, timestamp=101.0)
    )
    explanation = governor.explain_decision(decision)

    assert "DECISION: DENIED" in explanation
    assert "Assistance Authorized: NO" in explanation
    assert "• AILEE Pipeline Status: OUTRIGHT_REJECTED" in explanation
    assert "  → Grace: SKIPPED" in explanation
    assert "  → Consensus: SKIPPED" in explanation
    assert "  → Fallback used: YES" in explanation
    assert f"Fallback Value: {fallback_score} / 1.00" in explanation
    assert "Validated Trust Score:" not in explanation
    assert "Increase trust score" not in explanation


def test_explain_decision_handles_consent_denial_without_pipeline_result(governor, signals):
    decision = governor.evaluate(replace(signals, consent_records={}))
    explanation = governor.explain_decision(decision)

    assert "DECISION: DENIED" in explanation
    assert "Assistance Authorized: NO" in explanation
    assert "Validated Trust Score: unavailable (pipeline not run)" in explanation
    assert "CONSENT REQUIRED" in explanation
    assert "AILEE Pipeline Status:" not in explanation
    assert "Increase trust score" not in explanation


@pytest.mark.parametrize("ceiling_source", ["cognitive-state", "policy"])
def test_explain_decision_does_not_suggest_score_increase_to_remove_ceiling(
    governor, signals, ceiling_source
):
    signals = replace(signals, assistance_trust_score=0.75, interpretation_confidence=0.75)
    if ceiling_source == "cognitive-state":
        signals = replace(signals, current_cognitive_state=CognitiveState.UNKNOWN)
    else:
        governor = NeuroGovernor(
            cfg=governor.cfg,
            policy=NeuroAssistivePolicy(max_assistance_level=AssistanceLevel.NO_ASSISTANCE),
        )
    decision = governor.evaluate(signals)
    assert decision.ailee_result is not None and not decision.ailee_result.used_fallback
    assert decision.validated_trust_score == 0.75
    assert not decision.assistance_authorized

    explanation = governor.explain_decision(decision)
    assert "Validated Trust Score: 0.75 / 1.00" in explanation
    assert "Increase trust score" not in explanation
    assert "Review the decision reasons and applicable authority limits." in explanation


def test_malformed_state_cannot_commit_pipeline_history(governor, signals):
    with pytest.raises(TypeError, match="current_cognitive_state"):
        governor.evaluate(replace(signals, current_cognitive_state="UNKNOWN"))
    assert governor.state_tracker.get_current_state() is CognitiveState.UNKNOWN
    assert not get_decision_history(governor)
    assert governor.pipeline.last_result is None
