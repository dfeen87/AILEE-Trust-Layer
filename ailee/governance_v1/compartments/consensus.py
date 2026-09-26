# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Consensus Compartment (Python Microservice Module)
Evaluates candidate outputs from multiple models via ensemble voting, arbitration, and dissent detection.
"""

import uuid
from datetime import datetime
from typing import List
from ..schemas import CompartmentDecision, GovernanceInput


def evaluate_consensus(input_payload: GovernanceInput) -> CompartmentDecision:
    """
    Evaluates candidate outputs from multiple models.
    Inputs: candidate outputs from multiple models
    Outputs payload:
      - selected_output: str
      - confidence: float
      - dissent: List[str]
    """
    candidates = input_payload.candidate_outputs or []
    if not candidates:
        selected = input_payload.raw_input or ""
        return CompartmentDecision(
            id=f"consensus-{uuid.uuid4().hex[:12]}",
            compartment="consensus",
            payload={
                "selected_output": selected,
                "confidence": 0.5,
                "dissent": ["No candidate outputs provided; used raw input"],
            },
            created_at=datetime.utcnow(),
        )

    # Check agreement among candidates
    primary = candidates[0]
    dissent: List[str] = []

    for idx, cand in enumerate(candidates[1:], start=2):
        if cand.strip() != primary.strip():
            dissent.append(f"Model {idx} produced dissenting candidate output: '{cand[:50]}...'")

    if not dissent:
        confidence = 0.95
    else:
        confidence = max(0.4, 0.95 - (0.2 * len(dissent)))

    payload = {
        "selected_output": primary,
        "confidence": round(confidence, 4),
        "dissent": dissent,
    }

    return CompartmentDecision(
        id=f"consensus-{uuid.uuid4().hex[:12]}",
        compartment="consensus",
        payload=payload,
        created_at=datetime.utcnow(),
    )


evaluate = evaluate_consensus
