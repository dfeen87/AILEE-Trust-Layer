# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Fallback Compartment (Python Microservice Module)
Evaluates errors, high-risk flags, policy violations to generate safe-mode responses and escalation routing.
"""

import uuid
from datetime import datetime
from typing import Optional
from ..schemas import CompartmentDecision, GovernanceInput


def evaluate_fallback(input_payload: GovernanceInput) -> CompartmentDecision:
    """
    Evaluates errors, high-risk flags, and policy violations.
    Inputs: errors, high-risk flags, policy violations
    Outputs payload:
      - used: bool
      - fallback_response: Optional[str]
      - escalation_path: Optional[str]
    """
    error_flags = input_payload.error_flags or {}
    has_errors = bool(error_flags)
    force_fallback = error_flags.get("force_fallback", False)

    used = has_errors or force_fallback
    fallback_response: Optional[str] = None
    escalation_path: Optional[str] = None

    if used:
        fallback_response = (
            "An error or high-risk condition was detected. Returning safe-mode fallback output."
        )
        if error_flags.get("security_violation", False):
            escalation_path = "SECURITY_TIER_2_ESCALATION"
        else:
            escalation_path = "STANDARD_OPERATIONAL_FALLBACK"

    payload = {
        "used": used,
        "fallback_response": fallback_response,
        "escalation_path": escalation_path,
    }

    return CompartmentDecision(
        id=f"fallback-{uuid.uuid4().hex[:12]}",
        compartment="fallback",
        payload=payload,
        created_at=datetime.utcnow(),
    )


evaluate = evaluate_fallback
