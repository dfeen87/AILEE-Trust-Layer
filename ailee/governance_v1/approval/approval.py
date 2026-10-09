# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Final Approval Gate (Python Engine)
Evaluates governance policies across compartment decisions and records results to the global ledger.
"""

from typing import Optional, Union
from ..ledger import LedgerStore, write_ledger_entry
from ..schemas import (
    ALCOAMetadata,
    DecisionSummary,
    FinalApprovalRequest,
    FinalApprovalResponse,
)


def final_approval(
    request: Union[FinalApprovalRequest, dict],
    store: Optional[LedgerStore] = None,
) -> FinalApprovalResponse:
    """
    Evaluates final approval policies across all 4 compartment decisions:
      - Reject if risk_score > 0.7 (Policy: POLICY_RISK_001)
      - Reject if tone_flag == "harsh" or tone == "harsh" (Policy: POLICY_TONE_003)
      - Approve otherwise (Policy: POLICY_OK)

    Writes the final approval decision to the global governance ledger.
    """
    if isinstance(request, dict):
        safety_dec = request.get("safetyDecision", {})
        grace_dec = request.get("graceDecision", {})
        consensus_dec = request.get("consensusDecision", {})
        fallback_dec = request.get("fallbackDecision", {})

        safety_payload = safety_dec.get("payload", {}) if isinstance(safety_dec, dict) else getattr(safety_dec, "payload", {})
        grace_payload = grace_dec.get("payload", {}) if isinstance(grace_dec, dict) else getattr(grace_dec, "payload", {})
        consensus_payload = consensus_dec.get("payload", {}) if isinstance(consensus_dec, dict) else getattr(consensus_dec, "payload", {})
        fallback_payload = fallback_dec.get("payload", {}) if isinstance(fallback_dec, dict) else getattr(fallback_dec, "payload", {})
    else:
        safety_payload = request.safetyDecision.payload
        grace_payload = request.graceDecision.payload
        consensus_payload = request.consensusDecision.payload
        fallback_payload = request.fallbackDecision.payload

    risk_score = float(safety_payload.get("risk_score", 0.0))
    tone = grace_payload.get("tone", "neutral")
    tone_flag = grace_payload.get("tone_flag")

    approved = True
    reason = "All governance policies passed successfully."
    policy_id = "POLICY_OK"

    # Required Policy Checks
    if risk_score > 0.7:
        approved = False
        reason = f"Rejected due to elevated risk_score ({risk_score} > 0.7 threshold)."
        policy_id = "POLICY_RISK_001"
    elif tone_flag == "harsh" or tone == "harsh":
        approved = False
        reason = f"Rejected due to harsh tone flag in output (tone_flag: {tone_flag})."
        policy_id = "POLICY_TONE_003"

    response = FinalApprovalResponse(
        approved=approved,
        reason=reason,
        policyId=policy_id,
    )

    # Global ledger logging
    approval_decision_payload = {
        "approved": approved,
        "reason": reason,
        "policy_id": policy_id,
        "referenced_compartments": {
            "safety": safety_payload,
            "grace": grace_payload,
            "consensus": consensus_payload,
            "fallback": fallback_payload,
        },
    }

    metadata = ALCOAMetadata(
        attributable_to="approval_gate_engine",
        system_id="global_governance_ledger",
        version="10.0.1",
    )

    write_ledger_entry(
        compartment="global",
        input_payload={"request_type": "final_approval"},
        decision={"id": f"appr-{policy_id}", "payload": approval_decision_payload},
        metadata=metadata,
        store=store,
    )

    return response
