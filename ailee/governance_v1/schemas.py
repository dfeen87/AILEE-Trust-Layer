# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Governance Architecture Data Contracts and Schemas (v1.0)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class GovernanceInput:
    request_id: str
    raw_input: str
    user_context: Dict[str, Any] = field(default_factory=dict)
    candidate_outputs: List[str] = field(default_factory=list)
    conversation_state: Dict[str, Any] = field(default_factory=dict)
    error_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompartmentDecision:
    id: str
    compartment: Literal["safety", "grace", "consensus", "fallback"]
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class ALCOAMetadata:
    attributable_to: str
    system_id: str
    version: str = "10.0.0"
    contemporaneous_timestamp: str = ""
    is_original: bool = True
    validation_status: str = "VALID"
    legible_format: str = "json_v1"

    def __post_init__(self):
        if not self.contemporaneous_timestamp:
            object.__setattr__(
                self, "contemporaneous_timestamp", datetime.utcnow().isoformat()
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributable_to": self.attributable_to,
            "system_id": self.system_id,
            "version": self.version,
            "contemporaneous_timestamp": self.contemporaneous_timestamp,
            "is_original": self.is_original,
            "validation_status": self.validation_status,
            "legible_format": self.legible_format,
        }


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    compartment: str
    input_snapshot: Dict[str, Any]
    decision: Dict[str, Any]
    metadata: Dict[str, Any]
    previous_hash: str
    current_hash: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class TrainingSignal:
    id: str
    target_model: str
    features: Dict[str, Any]
    label: Any
    source_ledger_ids: List[str]


@dataclass(frozen=True)
class InterpretationResult:
    alignment_issues: List[str]
    features: Dict[str, Any]
    training_signals: List[TrainingSignal]


@dataclass
class DecisionSummary:
    id: str
    compartment: str
    payload: Dict[str, Any]


@dataclass
class FinalApprovalRequest:
    safetyDecision: DecisionSummary
    graceDecision: DecisionSummary
    consensusDecision: DecisionSummary
    fallbackDecision: DecisionSummary


@dataclass
class FinalApprovalResponse:
    approved: bool
    reason: str
    policyId: str
