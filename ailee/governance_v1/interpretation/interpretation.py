# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Interpretation Layer (Python)
Correlates ledger entries across compartments and evaluates required alignment checks.
"""

import uuid
from typing import List
from ..schemas import InterpretationResult, LedgerEntry, TrainingSignal


def interpret_event(
    safety: LedgerEntry,
    grace: LedgerEntry,
    consensus: LedgerEntry,
    fallback: LedgerEntry,
) -> InterpretationResult:
    """
    Correlates ledger entries across compartments and identifies alignment issues & training signals.

    Required Alignment Checks:
      1. High risk + high consensus confidence
      2. Harsh tone + no fallback
      3. Dissent + low risk
      4. Fallback used + low risk
    """
    alignment_issues: List[str] = []
    training_signals: List[TrainingSignal] = []

    # Extract payloads
    safety_payload = safety.decision.get("payload", {})
    grace_payload = grace.decision.get("payload", {})
    consensus_payload = consensus.decision.get("payload", {})
    fallback_payload = fallback.decision.get("payload", {})

    risk_score = safety_payload.get("risk_score", 0.0)
    tone = grace_payload.get("tone", "neutral")
    confidence = consensus_payload.get("confidence", 0.0)
    dissent = consensus_payload.get("dissent", [])
    fallback_used = fallback_payload.get("used", False)

    # 1. High risk + high consensus confidence
    if risk_score > 0.7 and confidence > 0.8:
        issue = f"High risk ({risk_score}) with high consensus confidence ({confidence})"
        alignment_issues.append(issue)
        training_signals.append(
            TrainingSignal(
                id=f"sig-{uuid.uuid4().hex[:8]}",
                target_model="safety_classifier",
                features={"risk_score": risk_score, "confidence": confidence},
                label="FLAG_HIGH_RISK_HIGH_CONFIDENCE",
                source_ledger_ids=[safety.id, consensus.id],
            )
        )

    # 2. Harsh tone + no fallback
    if tone == "harsh" and not fallback_used:
        issue = "Harsh tone detected without fallback activation"
        alignment_issues.append(issue)
        training_signals.append(
            TrainingSignal(
                id=f"sig-{uuid.uuid4().hex[:8]}",
                target_model="grace_tone_adapter",
                features={"tone": tone, "fallback_used": fallback_used},
                label="REDUCE_HARSH_TONE",
                source_ledger_ids=[grace.id, fallback.id],
            )
        )

    # 3. Dissent + low risk
    if len(dissent) > 0 and risk_score <= 0.3:
        issue = f"Dissent present among candidate models ({len(dissent)} dissenting) under low risk ({risk_score})"
        alignment_issues.append(issue)
        training_signals.append(
            TrainingSignal(
                id=f"sig-{uuid.uuid4().hex[:8]}",
                target_model="consensus_arbiter",
                features={"dissent_count": len(dissent), "risk_score": risk_score},
                label="HARMONIZE_LOW_RISK_DISSENT",
                source_ledger_ids=[consensus.id, safety.id],
            )
        )

    # 4. Fallback used + low risk
    if fallback_used and risk_score <= 0.3:
        issue = f"Fallback mechanism triggered under low risk condition ({risk_score})"
        alignment_issues.append(issue)
        training_signals.append(
            TrainingSignal(
                id=f"sig-{uuid.uuid4().hex[:8]}",
                target_model="fallback_router",
                features={"fallback_used": fallback_used, "risk_score": risk_score},
                label="OPTIMIZE_FALLBACK_TRIGGER",
                source_ledger_ids=[fallback.id, safety.id],
            )
        )

    features = {
        "risk_score": risk_score,
        "tone": tone,
        "consensus_confidence": confidence,
        "dissent_count": len(dissent),
        "fallback_used": fallback_used,
        "total_alignment_issues": len(alignment_issues),
    }

    return InterpretationResult(
        alignment_issues=alignment_issues,
        features=features,
        training_signals=training_signals,
    )
