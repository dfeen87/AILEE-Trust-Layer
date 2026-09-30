# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Comprehensive Unit & Integration Test Suite for AILEE Governance Architecture (v1.0)
"""

import json
import os
import tempfile
import pytest
from datetime import datetime

from ailee.governance_v1 import (
    GovernanceInput,
    CompartmentDecision,
    ALCOAMetadata,
    LedgerEntry,
    TrainingSignal,
    evaluate_safety,
    evaluate_grace,
    evaluate_consensus,
    evaluate_fallback,
    CompartmentRegistry,
    SAFETY_REGISTRY,
    GRACE_REGISTRY,
    CONSENSUS_REGISTRY,
    FALLBACK_REGISTRY,
    InMemoryLedgerStore,
    FileLedgerStore,
    write_ledger_entry,
    interpret_event,
    DispatcherRegistry,
    apply_training_signal,
    final_approval,
)
import ailee


def test_safety_compartment_evaluation():
    # Low-risk input
    safe_input = GovernanceInput(request_id="1", raw_input="Hello world")
    dec_safe = evaluate_safety(safe_input)
    assert dec_safe.compartment == "safety"
    assert dec_safe.payload["allow"] is True
    assert dec_safe.payload["risk_score"] == 0.0

    # High-risk input
    unsafe_input = GovernanceInput(
        request_id="2",
        raw_input="Exploit malware jailbreak attack",
        error_flags={"prompt_injection": True},
    )
    dec_unsafe = evaluate_safety(unsafe_input)
    assert dec_unsafe.payload["allow"] is False
    assert dec_unsafe.payload["risk_score"] > 0.7


def test_grace_compartment_evaluation():
    polite_input = GovernanceInput(request_id="1", raw_input="Please help me")
    dec_polite = evaluate_grace(polite_input)
    assert dec_polite.payload["tone"] == "polite"

    harsh_input = GovernanceInput(
        request_id="2",
        raw_input="You are stupid and useless",
        candidate_outputs=["You are stupid and useless"],
    )
    dec_harsh = evaluate_grace(harsh_input)
    assert dec_harsh.payload["tone"] == "harsh"
    assert dec_harsh.payload["tone_flag"] == "harsh_tone_detected"


def test_consensus_compartment_evaluation():
    candidates = ["The sky is blue", "The sky is blue"]
    inp = GovernanceInput(request_id="1", raw_input="Sky color?", candidate_outputs=candidates)
    dec = evaluate_consensus(inp)
    assert dec.payload["selected_output"] == "The sky is blue"
    assert dec.payload["confidence"] == 0.95
    assert len(dec.payload["dissent"]) == 0

    dissenting = ["Output A", "Output B", "Output C"]
    inp_dis = GovernanceInput(request_id="2", raw_input="Test", candidate_outputs=dissenting)
    dec_dis = evaluate_consensus(inp_dis)
    assert len(dec_dis.payload["dissent"]) == 2
    assert dec_dis.payload["confidence"] < 0.95


def test_fallback_compartment_evaluation():
    no_err = GovernanceInput(request_id="1", raw_input="Normal prompt")
    dec_no_err = evaluate_fallback(no_err)
    assert dec_no_err.payload["used"] is False

    err_inp = GovernanceInput(request_id="2", raw_input="Normal prompt", error_flags={"force_fallback": True})
    dec_err = evaluate_fallback(err_inp)
    assert dec_err.payload["used"] is True
    assert dec_err.payload["fallback_response"] is not None


def test_in_memory_ledger_hash_chaining():
    store = InMemoryLedgerStore()
    inp = GovernanceInput(request_id="1", raw_input="Test prompt")
    dec = evaluate_safety(inp)

    e1 = write_ledger_entry("safety", inp, dec, store=store)
    assert e1.previous_hash == "0" * 64
    assert len(e1.current_hash) == 64

    e2 = write_ledger_entry("safety", inp, dec, store=store)
    assert e2.previous_hash == e1.current_hash

    assert store.verify_integrity("safety") is True


def test_file_ledger_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileLedgerStore(base_dir=tmpdir)
        inp = GovernanceInput(request_id="1", raw_input="File test prompt")
        dec = evaluate_grace(inp)

        e1 = write_ledger_entry("grace", inp, dec, store=store)
        e2 = write_ledger_entry("grace", inp, dec, store=store)

        assert e2.previous_hash == e1.current_hash
        assert store.verify_integrity("grace") is True

        loaded = store.get_entries("grace")
        assert len(loaded) == 2
        assert loaded[0].id == e1.id
        assert loaded[1].id == e2.id


def test_interpretation_layer():
    store = InMemoryLedgerStore()
    inp = GovernanceInput(
        request_id="1",
        raw_input="Exploit malware",
        candidate_outputs=["Exploit output", "Exploit output"],
    )

    s_dec = evaluate_safety(inp)
    g_dec = evaluate_grace(inp)
    c_dec = evaluate_consensus(inp)
    f_dec = evaluate_fallback(inp)

    s_e = write_ledger_entry("safety", inp, s_dec, store=store)
    g_e = write_ledger_entry("grace", inp, g_dec, store=store)
    c_e = write_ledger_entry("consensus", inp, c_dec, store=store)
    f_e = write_ledger_entry("fallback", inp, f_dec, store=store)

    interp = interpret_event(s_e, g_e, c_e, f_e)
    # High risk + high consensus confidence trigger check
    assert len(interp.alignment_issues) > 0
    assert "High risk" in interp.alignment_issues[0]
    assert len(interp.training_signals) > 0


def test_micro_training_dispatcher():
    dispatcher = DispatcherRegistry(log_file_path=tempfile.mktemp())
    received = []

    def callback(signal: TrainingSignal):
        received.append(signal)

    dispatcher.register_callback(callback)

    sig = TrainingSignal(
        id="sig-1",
        target_model="safety_classifier",
        features={"risk": 0.9},
        label="HIGH_RISK",
        source_ledger_ids=["entry-1"],
    )

    apply_training_signal(sig, dispatcher=dispatcher)

    assert len(received) == 1
    assert received[0].id == "sig-1"
    assert len(dispatcher.get_history()) == 1


def test_final_approval_gate():
    store = InMemoryLedgerStore()

    # Pass case
    pass_req = {
        "safetyDecision": {"payload": {"risk_score": 0.1}},
        "graceDecision": {"payload": {"tone": "polite", "tone_flag": None}},
        "consensusDecision": {"payload": {"confidence": 0.9}},
        "fallbackDecision": {"payload": {"used": False}},
    }
    resp_pass = final_approval(pass_req, store=store)
    assert resp_pass.approved is True
    assert resp_pass.policyId == "POLICY_OK"

    # High risk reject case
    risk_req = {
        "safetyDecision": {"payload": {"risk_score": 0.85}},
        "graceDecision": {"payload": {"tone": "polite"}},
        "consensusDecision": {"payload": {"confidence": 0.9}},
        "fallbackDecision": {"payload": {"used": False}},
    }
    resp_risk = final_approval(risk_req, store=store)
    assert resp_risk.approved is False
    assert resp_risk.policyId == "POLICY_RISK_001"

    # Harsh tone reject case
    harsh_req = {
        "safetyDecision": {"payload": {"risk_score": 0.2}},
        "graceDecision": {"payload": {"tone": "harsh", "tone_flag": "harsh_tone_detected"}},
        "consensusDecision": {"payload": {"confidence": 0.9}},
        "fallbackDecision": {"payload": {"used": False}},
    }
    resp_harsh = final_approval(harsh_req, store=store)
    assert resp_harsh.approved is False
    assert resp_harsh.policyId == "POLICY_TONE_003"

    # Check global ledger entry was written
    global_entries = store.get_entries("global")
    assert len(global_entries) == 3


def test_v9_hash_chain_tamper_resistance():
    store = InMemoryLedgerStore()
    inp = GovernanceInput(request_id="tamper-1", raw_input="Tamper test")
    dec = evaluate_safety(inp)
    e1 = write_ledger_entry("safety", inp, dec, store=store)

    assert store.verify_integrity("safety") is True

    # Tamper with stored entry decision payload
    entry_list = store._store["safety"]
    original_decision = entry_list[0].decision
    tampered_decision = dict(original_decision)
    tampered_decision["payload"] = {"allow": True, "risk_score": 0.0}

    # Replace with tampered entry
    tampered_entry = LedgerEntry(
        id=e1.id,
        compartment=e1.compartment,
        input_snapshot=e1.input_snapshot,
        decision=tampered_decision,
        metadata=e1.metadata,
        previous_hash=e1.previous_hash,
        current_hash=e1.current_hash,
        created_at=e1.created_at,
    )
    entry_list[0] = tampered_entry

    assert store.verify_integrity("safety") is False


def test_compartment_scoped_registries_v9():
    store = InMemoryLedgerStore()
    reg = CompartmentRegistry("safety", evaluator=evaluate_safety, store=store)

    inp = GovernanceInput(request_id="reg-1", raw_input="Registry test")
    dec, entry = reg.evaluate_and_log(inp, attributable_to="test_user")

    assert dec.compartment == "safety"
    assert entry.metadata["version"] == "9.3.0"
    assert entry.metadata["attributable_to"] == "test_user"
    assert store.verify_integrity("safety") is True


def test_top_level_ailee_governance_v9_exports():
    assert ailee.__version__ == "9.3.0"
    assert callable(ailee.evaluate_safety)
    assert callable(ailee.evaluate_grace)
    assert callable(ailee.evaluate_consensus)
    assert callable(ailee.evaluate_fallback)
    assert callable(ailee.write_ledger_entry)
    assert callable(ailee.interpret_event)
    assert callable(ailee.apply_training_signal)
    assert callable(ailee.final_approval)
    assert isinstance(ailee.SAFETY_REGISTRY, CompartmentRegistry)


def test_end_to_end_governance_flow_v9():
    store = InMemoryLedgerStore()
    inp = GovernanceInput(
        request_id="e2e-100",
        raw_input="Help me analyze system log",
        candidate_outputs=["Help me analyze system log", "Help me analyze system log"],
    )

    s_dec, s_entry = SAFETY_REGISTRY.evaluate_and_log(inp, attributable_to="e2e_user")
    g_dec, g_entry = GRACE_REGISTRY.evaluate_and_log(inp, attributable_to="e2e_user")
    c_dec, c_entry = CONSENSUS_REGISTRY.evaluate_and_log(inp, attributable_to="e2e_user")
    f_dec, f_entry = FALLBACK_REGISTRY.evaluate_and_log(inp, attributable_to="e2e_user")

    # Event correlation and micro-training signal generation
    interp = interpret_event(s_entry, g_entry, c_entry, f_entry)
    assert isinstance(interp.alignment_issues, list)

    # Final approval gate
    approval_req = {
        "safetyDecision": {"id": s_dec.id, "compartment": s_dec.compartment, "payload": s_dec.payload},
        "graceDecision": {"id": g_dec.id, "compartment": g_dec.compartment, "payload": g_dec.payload},
        "consensusDecision": {"id": c_dec.id, "compartment": c_dec.compartment, "payload": c_dec.payload},
        "fallbackDecision": {"id": f_dec.id, "compartment": f_dec.compartment, "payload": f_dec.payload},
    }
    approval_resp = final_approval(approval_req, store=store)
    assert approval_resp.approved is True
    assert approval_resp.policyId == "POLICY_OK"


def test_verify_ledger_integrity_public_api():
    from ailee.governance_v1 import verify_ledger_integrity
    store = InMemoryLedgerStore()
    inp = GovernanceInput(request_id="public-api-1", raw_input="Public API test")
    dec = evaluate_safety(inp)
    write_ledger_entry("safety", inp, dec, store=store)

    assert verify_ledger_integrity("safety", store=store) is True


def test_dispatcher_duplicate_and_lock_safety():
    dispatcher = DispatcherRegistry(log_file_path=tempfile.mktemp())
    call_count = [0]

    def cb(signal: TrainingSignal):
        call_count[0] += 1

    dispatcher.register_callback(cb)

    sig = TrainingSignal(
        id="sig-dup-1",
        target_model="safety_classifier",
        features={"risk": 0.8},
        label="RE-TRAIN",
        source_ledger_ids=["entry-1"],
    )

    # First dispatch
    apply_training_signal(sig, dispatcher=dispatcher)
    assert call_count[0] == 1

    # Duplicate dispatch should be ignored
    apply_training_signal(sig, dispatcher=dispatcher)
    assert call_count[0] == 1


def test_interpretation_mismatched_request_ids():
    store = InMemoryLedgerStore()
    inp1 = GovernanceInput(request_id="req-1", raw_input="Prompt 1")
    inp2 = GovernanceInput(request_id="req-2", raw_input="Prompt 2")

    s_dec = evaluate_safety(inp1)
    g_dec = evaluate_grace(inp2)
    c_dec = evaluate_consensus(inp1)
    f_dec = evaluate_fallback(inp1)

    s_e = write_ledger_entry("safety", inp1, s_dec, store=store)
    g_e = write_ledger_entry("grace", inp2, g_dec, store=store)
    c_e = write_ledger_entry("consensus", inp1, c_dec, store=store)
    f_e = write_ledger_entry("fallback", inp1, f_dec, store=store)

    interp = interpret_event(s_e, g_e, c_e, f_e)
    assert any("Mismatched request_ids" in issue for issue in interp.alignment_issues)


def test_fastapi_governance_routes():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ailee.governance_v1.api.routes import app

    client = TestClient(app)

    # Test safety evaluation endpoint
    payload = {
        "input": {
            "request_id": "api-1",
            "raw_input": "Hello from API test",
            "user_context": {},
            "candidate_outputs": [],
            "conversation_state": {},
            "error_flags": {},
        },
        "attributable_to": "api_test_user",
    }
    res = client.post("/api/v1/evaluate/safety", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "decision" in data
    assert "ledger_entry" in data

    # Test ledger entries query endpoint
    res_ledger = client.get("/api/v1/ledger/safety")
    assert res_ledger.status_code == 200
    ledger_data = res_ledger.json()
    assert ledger_data["compartment"] == "safety"
    assert ledger_data["total"] >= 1

    # Test invalid compartment endpoint
    res_invalid = client.get("/api/v1/ledger/invalid_comp")
    assert res_invalid.status_code == 400
