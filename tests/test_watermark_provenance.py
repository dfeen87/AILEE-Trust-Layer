# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

import time
import pytest

from ailee import (
    WatermarkProvenanceGovernor,
    WatermarkProvenancePolicy,
    WatermarkProvenanceSignals,
    WatermarkProvenanceDecision,
    WatermarkProvenanceTrustLevel,
    WatermarkProvenanceHealthStatus,
    WatermarkProvenanceControlDomain,
    WatermarkProvenanceControlAction,
    ProvenanceQualifier,
    ProvenanceEventNode,
    WatermarkSignalData,
    create_watermark_provenance_governor,
    create_default_watermark_provenance_governor,
    create_strict_watermark_provenance_governor,
    create_permissive_watermark_provenance_governor,
    create_watermark_provenance_wrapper,
    validate_watermark_provenance_signals,
)


def test_basic_evaluation_and_qualifiers():
    governor = create_default_watermark_provenance_governor()

    signal_data = WatermarkSignalData(
        detector_name="SynthID-Text",
        raw_score=0.88,
        presence_detected=True,
        confidence=0.95,
    )

    custody = [
        ProvenanceEventNode(
            event_id="evt_01",
            stage="G0_generation",
            actor="model:claude-3-5-sonnet",
            action="draft",
        ),
        ProvenanceEventNode(
            event_id="evt_02",
            stage="E1_edit",
            actor="human:editor_jane",
            action="edit",
        ),
    ]

    signals = WatermarkProvenanceSignals(
        control_domain=WatermarkProvenanceControlDomain.WORKFLOW_EVALUATION,
        proposed_action=WatermarkProvenanceControlAction.MONITOR,
        watermark_signals=[signal_data],
        custody_chain=custody,
        workflow_context={"touch_level": "heavy"},
    )

    decision = governor.evaluate(signals)

    assert decision.actionable is True
    assert decision.provenance_confidence > 0.5
    assert ProvenanceQualifier.MODEL_INVOLVED_NOT_AUTHORED in decision.qualifiers
    assert ProvenanceQualifier.HUMAN_EDITED in decision.qualifiers
    assert decision.metadata["requires_human_review"] is True
    assert decision.metadata["challenge_available"] is True


def test_binary_label_rejection_over_interpretation():
    governor = create_default_watermark_provenance_governor()

    signals = WatermarkProvenanceSignals(
        control_domain=WatermarkProvenanceControlDomain.WORKFLOW_EVALUATION,
        proposed_action=WatermarkProvenanceControlAction.FLAG_FOR_REVIEW,
        watermark_signals=[
            WatermarkSignalData("Claude-Watermark", raw_score=0.9, presence_detected=True, confidence=0.9)
        ],
        attempted_binary_label="AI-generated",
    )

    decision = governor.evaluate(signals)

    assert decision.over_interpreted is True
    assert "OVER_INTERPRETATION_DETECTED" in decision.safety_flags
    assert "warning" in decision.metadata


def test_high_stakes_corroboration_safeguard():
    governor = create_strict_watermark_provenance_governor()

    # Uncorroborated high stakes signal
    uncorroborated_signals = WatermarkProvenanceSignals(
        control_domain=WatermarkProvenanceControlDomain.HIGH_STAKES_COMPLIANCE,
        proposed_action=WatermarkProvenanceControlAction.REQUIRE_CORROBORATION,
        watermark_signals=[
            WatermarkSignalData("SynthID-Text", raw_score=0.92, presence_detected=True, confidence=0.9)
        ],
        custody_chain=[],  # Missing custody chain or human verification
        is_high_stakes=True,
        context_type="disciplinary_decision",
    )

    decision_uncorroborated = governor.evaluate(uncorroborated_signals)
    assert decision_uncorroborated.actionable is False
    assert decision_uncorroborated.authorized_level == WatermarkProvenanceTrustLevel.NO_ACTION
    assert "INSUFFICIENT_PROVENANCE_FOR_ACTION" in decision_uncorroborated.safety_flags
    assert "CORROBORATION_REQUIRED" in decision_uncorroborated.safety_flags

    # Corroborated high stakes signal
    corroborated_signals = WatermarkProvenanceSignals(
        control_domain=WatermarkProvenanceControlDomain.HIGH_STAKES_COMPLIANCE,
        proposed_action=WatermarkProvenanceControlAction.REQUIRE_CORROBORATION,
        watermark_signals=[
            WatermarkSignalData("SynthID-Text", raw_score=0.92, presence_detected=True, confidence=0.9)
        ],
        custody_chain=[
            ProvenanceEventNode("e1", "G0_generation", "model:gpt-4", "draft"),
            ProvenanceEventNode("e2", "V2_verification", "human:compliance_officer", "verify"),
        ],
        workflow_context={"human_verification": True},
        is_high_stakes=True,
        context_type="disciplinary_decision",
    )

    decision_corroborated = governor.evaluate(corroborated_signals)
    assert decision_corroborated.actionable is True
    assert decision_corroborated.authorized_level >= WatermarkProvenanceTrustLevel.SUPERVISED


def test_attack_surface_disruption_penalty():
    governor = create_default_watermark_provenance_governor()

    signals = WatermarkProvenanceSignals(
        control_domain=WatermarkProvenanceControlDomain.WORKFLOW_EVALUATION,
        proposed_action=WatermarkProvenanceControlAction.MONITOR,
        watermark_signals=[
            WatermarkSignalData("SynthID-Text", raw_score=0.85, presence_detected=True, confidence=0.9)
        ],
        suspected_disruptions=["paraphrasing", "public_removal_tool"],
    )

    decision = governor.evaluate(signals)

    assert ProvenanceQualifier.STRUCTURALLY_REWRITTEN in decision.qualifiers
    assert "DISRUPTION_VECTOR_DETECTED_PARAPHRASING" in decision.safety_flags
    assert "DISRUPTION_VECTOR_DETECTED_PUBLIC_REMOVAL_TOOL" in decision.safety_flags
    assert "attack_surface_risk" in decision.metadata


def test_wrapper_integration():
    def mock_legacy_detector(text: str) -> dict:
        return {"detected": True, "score": 0.85}

    wrapper = create_watermark_provenance_wrapper(
        legacy_fn=mock_legacy_detector,
        shadow_mode=False,
    )

    signals = WatermarkProvenanceSignals(
        control_domain=WatermarkProvenanceControlDomain.WORKFLOW_EVALUATION,
        proposed_action=WatermarkProvenanceControlAction.MONITOR,
        watermark_signals=[
            WatermarkSignalData("SynthID-Text", raw_score=0.85, presence_detected=True, confidence=0.9)
        ],
    )

    res = wrapper.execute(signals, "Sample text")
    assert res["blocked"] is False
    assert res["result"]["detected"] is True
    assert "MODEL_INVOLVED_NOT_AUTHORED" in res["qualifiers"]


def test_validation_function():
    signals = WatermarkProvenanceSignals(
        control_domain=WatermarkProvenanceControlDomain.WORKFLOW_EVALUATION,
        proposed_action=WatermarkProvenanceControlAction.MONITOR,
        watermark_signals=[
            WatermarkSignalData("Invalid-Detector", raw_score=1.5, presence_detected=True, confidence=0.8)
        ],
    )

    issues = validate_watermark_provenance_signals(signals)
    assert len(issues) > 0
    assert "raw_score must be between 0.0 and 1.0" in issues[0]
