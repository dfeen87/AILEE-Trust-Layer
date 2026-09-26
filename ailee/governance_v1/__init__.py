# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
AILEE Governance Architecture (v1.0) Package
Multi-compartment governance system enforcing strict immutability, traceability, and cross-compartment correlation.
"""

from .schemas import (
    GovernanceInput,
    CompartmentDecision,
    ALCOAMetadata,
    LedgerEntry,
    TrainingSignal,
    InterpretationResult,
    DecisionSummary,
    FinalApprovalRequest,
    FinalApprovalResponse,
)

from .compartments import (
    evaluate_safety,
    evaluate_grace,
    evaluate_consensus,
    evaluate_fallback,
    CompartmentRegistry,
    SAFETY_REGISTRY,
    GRACE_REGISTRY,
    CONSENSUS_REGISTRY,
    FALLBACK_REGISTRY,
)

from .ledger import (
    LedgerStore,
    InMemoryLedgerStore,
    FileLedgerStore,
    write_ledger_entry,
    verify_ledger_integrity,
    get_default_ledger_store,
    set_default_ledger_store,
)

from .interpretation import interpret_event
from .training import (
    DispatcherRegistry,
    apply_training_signal,
    register_training_callback,
)
from .approval import final_approval

__all__ = [
    # Data Contracts
    "GovernanceInput",
    "CompartmentDecision",
    "ALCOAMetadata",
    "LedgerEntry",
    "TrainingSignal",
    "InterpretationResult",
    "DecisionSummary",
    "FinalApprovalRequest",
    "FinalApprovalResponse",
    # Compartments
    "evaluate_safety",
    "evaluate_grace",
    "evaluate_consensus",
    "evaluate_fallback",
    "CompartmentRegistry",
    "SAFETY_REGISTRY",
    "GRACE_REGISTRY",
    "CONSENSUS_REGISTRY",
    "FALLBACK_REGISTRY",
    # Ledger
    "LedgerStore",
    "InMemoryLedgerStore",
    "FileLedgerStore",
    "write_ledger_entry",
    "verify_ledger_integrity",
    "get_default_ledger_store",
    "set_default_ledger_store",
    # Interpretation
    "interpret_event",
    # Micro-Training
    "DispatcherRegistry",
    "apply_training_signal",
    "register_training_callback",
    # Approval
    "final_approval",
]
