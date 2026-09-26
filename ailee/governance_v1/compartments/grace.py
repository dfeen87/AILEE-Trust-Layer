# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Grace Compartment (Python Microservice Module)
Evaluates tone analysis, sentiment classification, and cultural sensitivity.
"""

import uuid
from datetime import datetime
from typing import Optional
from ..schemas import CompartmentDecision, GovernanceInput


def evaluate_grace(input_payload: GovernanceInput) -> CompartmentDecision:
    """
    Evaluates tone, sentiment, and cultural sensitivity.
    Inputs: proposed text, conversation context
    Outputs payload:
      - tone: Literal["polite", "neutral", "harsh"]
      - tone_flag: Optional[str]
      - adjustments: dict
    """
    candidates = input_payload.candidate_outputs or []
    proposed_text = candidates[0] if candidates else input_payload.raw_input or ""
    text_lower = proposed_text.lower()

    harsh_keywords = ["stupid", "idiot", "hate", "shut up", "useless", "terrible", "harsh"]
    polite_keywords = ["please", "thank you", "kindly", "appreciate", "polite", "gladly"]

    harsh_count = sum(1 for kw in harsh_keywords if kw in text_lower)
    polite_count = sum(1 for kw in polite_keywords if kw in text_lower)

    tone_flag: Optional[str] = None
    adjustments = {}

    if harsh_count > 0 or "harsh" in input_payload.conversation_state.get("override_tone", ""):
        tone = "harsh"
        tone_flag = "harsh_tone_detected"
        adjustments = {"recommended_prefix": "We respectfully advise that "}
    elif polite_count > 0:
        tone = "polite"
        tone_flag = None
        adjustments = {}
    else:
        tone = "neutral"
        tone_flag = None
        adjustments = {}

    payload = {
        "tone": tone,
        "tone_flag": tone_flag,
        "adjustments": adjustments,
        "evaluated_text": proposed_text[:100],
    }

    return CompartmentDecision(
        id=f"grace-{uuid.uuid4().hex[:12]}",
        compartment="grace",
        payload=payload,
        created_at=datetime.utcnow(),
    )


evaluate = evaluate_grace
