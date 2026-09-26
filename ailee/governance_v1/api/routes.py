# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
FastAPI REST API Microservices for AILEE Governance Architecture v1.0
Exposes endpoints for compartment evaluation, ledger queries, diffing, event correlation, and final approval.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field

from ..compartments.safety import evaluate_safety
from ..compartments.grace import evaluate_grace
from ..compartments.consensus import evaluate_consensus
from ..compartments.fallback import evaluate_fallback
from ..ledger import get_default_ledger_store, write_ledger_entry
from ..interpretation import interpret_event
from ..training import apply_training_signal
from ..approval import final_approval
from ..schemas import (
    ALCOAMetadata,
    DecisionSummary,
    FinalApprovalRequest,
    GovernanceInput,
    TrainingSignal,
)

app = FastAPI(
    title="AILEE Governance Architecture API (v1.0)",
    description="Isolated Python microservice endpoints for AILEE multi-compartment governance.",
    version="1.0.0",
)


class GovernanceInputModel(BaseModel):
    request_id: str
    raw_input: str
    user_context: Dict[str, Any] = Field(default_factory=dict)
    candidate_outputs: List[str] = Field(default_factory=list)
    conversation_state: Dict[str, Any] = Field(default_factory=dict)
    error_flags: Dict[str, Any] = Field(default_factory=dict)


class EvaluateAndLogRequest(BaseModel):
    input: GovernanceInputModel
    attributable_to: Optional[str] = "api_user"


@app.post("/api/v1/evaluate/safety")
def evaluate_safety_endpoint(req: EvaluateAndLogRequest):
    inp = GovernanceInput(**req.input.model_dump())
    decision = evaluate_safety(inp)
    meta = ALCOAMetadata(attributable_to=req.attributable_to or "api_user", system_id="safety_service")
    entry = write_ledger_entry("safety", inp, decision, meta)
    return {"decision": decision, "ledger_entry": entry}


@app.post("/api/v1/evaluate/grace")
def evaluate_grace_endpoint(req: EvaluateAndLogRequest):
    inp = GovernanceInput(**req.input.model_dump())
    decision = evaluate_grace(inp)
    meta = ALCOAMetadata(attributable_to=req.attributable_to or "api_user", system_id="grace_service")
    entry = write_ledger_entry("grace", inp, decision, meta)
    return {"decision": decision, "ledger_entry": entry}


@app.post("/api/v1/evaluate/consensus")
def evaluate_consensus_endpoint(req: EvaluateAndLogRequest):
    inp = GovernanceInput(**req.input.model_dump())
    decision = evaluate_consensus(inp)
    meta = ALCOAMetadata(attributable_to=req.attributable_to or "api_user", system_id="consensus_service")
    entry = write_ledger_entry("consensus", inp, decision, meta)
    return {"decision": decision, "ledger_entry": entry}


@app.post("/api/v1/evaluate/fallback")
def evaluate_fallback_endpoint(req: EvaluateAndLogRequest):
    inp = GovernanceInput(**req.input.model_dump())
    decision = evaluate_fallback(inp)
    meta = ALCOAMetadata(attributable_to=req.attributable_to or "api_user", system_id="fallback_service")
    entry = write_ledger_entry("fallback", inp, decision, meta)
    return {"decision": decision, "ledger_entry": entry}


@app.get("/api/v1/ledger/{compartment}")
def get_ledger_entries(
    compartment: str,
    min_risk: Optional[float] = None,
    max_risk: Optional[float] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    store = get_default_ledger_store()
    entries = store.get_entries(compartment)

    # Optional filtering
    filtered = []
    for e in entries:
        payload = e.decision.get("payload", {})
        risk = payload.get("risk_score")
        if min_risk is not None and (risk is None or risk < min_risk):
            continue
        if max_risk is not None and (risk is None or risk > max_risk):
            continue
        filtered.append(e)

    return {"compartment": compartment, "total": len(filtered), "entries": filtered[:limit]}


@app.get("/api/v1/ledger/diff")
def get_ledger_diff(comp1: str = Query(...), comp2: str = Query(...)):
    store = get_default_ledger_store()
    entries1 = store.get_entries(comp1)
    entries2 = store.get_entries(comp2)

    diff = {
        "comp1": comp1,
        "comp1_count": len(entries1),
        "comp2": comp2,
        "comp2_count": len(entries2),
        "comp1_latest_hash": entries1[-1].current_hash if entries1 else None,
        "comp2_latest_hash": entries2[-1].current_hash if entries2 else None,
        "count_difference": abs(len(entries1) - len(entries2)),
    }
    return diff


@app.get("/api/v1/events/correlation")
def get_event_correlation():
    store = get_default_ledger_store()
    safety_entries = store.get_entries("safety")
    grace_entries = store.get_entries("grace")
    consensus_entries = store.get_entries("consensus")
    fallback_entries = store.get_entries("fallback")
    global_entries = store.get_entries("global")

    correlated = []
    min_len = min(
        len(safety_entries),
        len(grace_entries),
        len(consensus_entries),
        len(fallback_entries),
    )

    for i in range(min_len):
        s, g, c, f = (
            safety_entries[i],
            grace_entries[i],
            consensus_entries[i],
            fallback_entries[i],
        )
        res = interpret_event(s, g, c, f)
        correlated.append(
            {
                "index": i,
                "request_id": s.input_snapshot.get("request_id"),
                "safety_entry_id": s.id,
                "grace_entry_id": g.id,
                "consensus_entry_id": c.id,
                "fallback_entry_id": f.id,
                "alignment_issues": res.alignment_issues,
                "features": res.features,
                "training_signals_count": len(res.training_signals),
            }
        )

    return {"total_correlated_events": len(correlated), "correlated_events": correlated, "global_ledger_count": len(global_entries)}


class FinalApprovalRequestModel(BaseModel):
    safetyDecision: Dict[str, Any]
    graceDecision: Dict[str, Any]
    consensusDecision: Dict[str, Any]
    fallbackDecision: Dict[str, Any]


@app.post("/api/v1/approval/evaluate")
def final_approval_endpoint(req: FinalApprovalRequestModel):
    res = final_approval(req.model_dump())
    return res
