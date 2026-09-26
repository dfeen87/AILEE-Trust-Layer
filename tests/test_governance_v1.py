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
    InMemoryLedgerStore,
    FileLedgerStore,
    write_ledger_entry,
    interpret_event,
    DispatcherRegistry,
    apply_training_signal,
    final_approval,
)


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
