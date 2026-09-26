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

    # Helper function to extract dict payload from decision safely
    def _extract_payload(entry: LedgerEntry) -> dict:
        if not entry or not hasattr(entry, "decision"):
            return {}
        dec = entry.decision
        if isinstance(dec, dict):
            return dec.get("payload", {}) if isinstance(dec.get("payload"), dict) else {}
        if hasattr(dec, "payload"):
            payload = getattr(dec, "payload")
            return payload if isinstance(payload, dict) else {}
        return {}

    def _get_entry_id(entry: LedgerEntry) -> str:
        if not entry:
            return "unknown-id"
        return getattr(entry, "id", "unknown-id")

    # Safe payload extraction
    safety_payload = _extract_payload(safety)
    grace_payload = _extract_payload(grace)
    consensus_payload = _extract_payload(consensus)
    fallback_payload = _extract_payload(fallback)

    # Check for request_id or timestamp mismatch across compartments
    def _get_request_id(entry: LedgerEntry) -> str:
        if not entry or not hasattr(entry, "input_snapshot"):
            return ""
        snap = entry.input_snapshot
        if isinstance(snap, dict):
            return str(snap.get("request_id", ""))
        return ""

    req_ids = {
        "safety": _get_request_id(safety),
        "grace": _get_request_id(grace),
        "consensus": _get_request_id(consensus),
        "fallback": _get_request_id(fallback),
    }
    non_empty_ids = {k: v for k, v in req_ids.items() if v}
    if len(set(non_empty_ids.values())) > 1:
        alignment_issues.append(
            f"Mismatched request_ids across compartments: {non_empty_ids}"
        )

    risk_score = float(safety_payload.get("risk_score", 0.0))
    tone = str(grace_payload.get("tone", "neutral"))
    confidence = float(consensus_payload.get("confidence", 0.0))
    dissent = consensus_payload.get("dissent", [])
    if not isinstance(dissent, list):
        dissent = [str(dissent)]
    fallback_used = bool(fallback_payload.get("used", False))

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
                source_ledger_ids=[_get_entry_id(safety), _get_entry_id(consensus)],
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
                source_ledger_ids=[_get_entry_id(grace), _get_entry_id(fallback)],
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
                source_ledger_ids=[_get_entry_id(consensus), _get_entry_id(safety)],
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
                source_ledger_ids=[_get_entry_id(fallback), _get_entry_id(safety)],
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
