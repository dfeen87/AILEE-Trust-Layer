"""
NEURO-ASSISTIVE Domain — AILEE Trust Layer
Version: 1.0.0 - Production Grade

Governance and restraint framework for AI systems that assist human cognition,
communication, and perception while preserving autonomy, consent, identity, and dignity.

Core principle: Stabilizing companion, not cognitive authority.

Primary entry point: NeuroGovernor
Quick start with: create_neuro_governor()
"""

from .neuro_assistive import (
    # === Primary API ===
    NeuroGovernor,
    create_neuro_governor,
    validate_neuro_signals,
    
    # === Core Data Structures ===
    NeuroSignals,
    NeuroDecisionResult,
    NeuroEvent,
    
    # === Configuration ===
    NeuroAssistivePolicy,
    TemporalSafeguards,
    
    # === Enumerations ===
    CognitiveState,
    AssistanceLevel,
    AssistanceOutcome,
    ImpairmentCategory,
    ConsentStatus,
    
    # === Assessment Components ===
    ConsentRecord,
    InterpretationResult,
    CognitiveLoadMetrics,
    SessionMetrics,
    AssistanceConstraints,
    
    # === Analysis Components (Advanced) ===
    PolicyEvaluator,
    CognitiveStateTracker,
)

__version__ = "1.0.0"

__all__ = [
    # Primary API
    "NeuroGovernor",
    "create_neuro_governor",
    "validate_neuro_signals",
    
    # Core structures
    "NeuroSignals",
    "NeuroDecisionResult",
    "NeuroEvent",
    "NeuroAssistivePolicy",
    "TemporalSafeguards",
    
    # Enums
    "CognitiveState",
    "AssistanceLevel",
    "AssistanceOutcome",
    "ImpairmentCategory",
    "ConsentStatus",
    
    # Assessment components
    "ConsentRecord",
    "InterpretationResult",
    "CognitiveLoadMetrics",
    "SessionMetrics",
    "AssistanceConstraints",
    
    # Advanced (optional)
    "PolicyEvaluator",
    "CognitiveStateTracker",
]
