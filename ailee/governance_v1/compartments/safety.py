# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Safety Compartment (Python Microservice Module)
Evaluates raw request, user context, and candidate outputs against rule engines and compliance policies.
"""

import uuid
from datetime import datetime
from typing import List
from ..schemas import CompartmentDecision, GovernanceInput


def evaluate_safety(input_payload: GovernanceInput) -> CompartmentDecision:
    """
    Evaluates safety rules, compliance validation, and risk scoring.
    Inputs: raw request, user context, candidate outputs
    Outputs payload:
      - allow: bool
      - risk_score: float
      - reasons: List[str]
    """
    reasons: List[str] = []
    risk_score = 0.0

    raw_input_lower = (input_payload.raw_input or "").lower()

    # Example safety rule checks
    high_risk_keywords = ["exploit", "malware", "bypass", "attack", "jailbreak", "harmful"]
    for keyword in high_risk_keywords:
        if keyword in raw_input_lower:
            risk_score += 0.4
            reasons.append(f"Detected risk keyword: '{keyword}'")

    # User context checks
    if input_payload.user_context.get("flagged_account", False):
        risk_score += 0.3
        reasons.append("User account flagged for prior policy violations")

    # Error flags check
    if input_payload.error_flags.get("prompt_injection", False):
        risk_score += 0.5
        reasons.append("Prompt injection flag present")

    risk_score = min(1.0, risk_score)
    allow = risk_score <= 0.7

    if not reasons:
        reasons.append("No policy or risk violations detected")

    payload = {
        "allow": allow,
        "risk_score": round(risk_score, 4),
        "reasons": reasons,
    }

    return CompartmentDecision(
        id=f"safety-{uuid.uuid4().hex[:12]}",
        compartment="safety",
        payload=payload,
        created_at=datetime.utcnow(),
    )


evaluate = evaluate_safety
