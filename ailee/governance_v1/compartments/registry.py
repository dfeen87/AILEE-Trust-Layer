# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Compartment-Scoped Registry System (Python)
Provides dedicated registries for Safety, Grace, Consensus, and Fallback compartments,
ensuring every safeguard decision is transformed into an immutable ALCOA-compliant
SHA-256 hash-chained ledger entry.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import threading

from ..schemas import ALCOAMetadata, CompartmentDecision, GovernanceInput, LedgerEntry
from ..ledger import LedgerStore, get_default_ledger_store, write_ledger_entry


from ..training import DispatcherRegistry, apply_training_signal


class CompartmentRegistry:
    """
    Compartment-scoped registry that owns rules, evaluators, dispatchers, and store
    for a specific compartment (safety, grace, consensus, fallback).
    """

    def __init__(
        self,
        compartment: str,
        evaluator: Optional[Callable[[GovernanceInput], CompartmentDecision]] = None,
        store: Optional[LedgerStore] = None,
        dispatcher: Optional[DispatcherRegistry] = None,
    ):
        self.compartment = compartment
        self.evaluator = evaluator
        self._store = store
        self.dispatcher = dispatcher or DispatcherRegistry()
        self._rules: List[Callable[[GovernanceInput], Dict[str, Any]]] = []
        self._lock = threading.Lock()

    @property
    def store(self) -> LedgerStore:
        return self._store or get_default_ledger_store()

    def set_store(self, store: LedgerStore):
        with self._lock:
            self._store = store

    def register_rule(self, rule: Callable[[GovernanceInput], Dict[str, Any]]):
        with self._lock:
            self._rules.append(rule)

    def get_rules(self) -> List[Callable[[GovernanceInput], Dict[str, Any]]]:
        with self._lock:
            return list(self._rules)

    def evaluate_and_log(
        self,
        input_payload: GovernanceInput,
        attributable_to: str = "system",
        metadata_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[CompartmentDecision, LedgerEntry]:
        """
        Evaluates input using registered evaluator, executes custom registered rules,
        and immediately writes an immutable ALCOA-compliant ledger entry.
        """
        if self.evaluator is None:
            raise ValueError(f"No evaluator registered for compartment '{self.compartment}'.")

        decision = self.evaluator(input_payload)

        # Run registered rules and update payload rule_findings if rules exist
        rule_results = []
        rules = self.get_rules()
        if rules:
            for rule in rules:
                try:
                    res = rule(input_payload)
                    if res:
                        rule_results.append(res)
                except Exception as e:
                    rule_results.append({"rule_error": str(e)})

        if rule_results:
            decision.payload["rule_findings"] = rule_results

        # Build ALCOA metadata
        meta = ALCOAMetadata(
            attributable_to=attributable_to,
            system_id=f"{self.compartment}_compartment_registry",
            version="9.1.0",
            is_original=True,
            validation_status="VALID",
            legible_format="json_v1",
        )
        meta_dict = meta.to_dict()
        if metadata_overrides:
            meta_dict.update(metadata_overrides)

        ledger_entry = write_ledger_entry(
            compartment=self.compartment,
            input_payload=input_payload,
            decision=decision,
            metadata=meta_dict,
            store=self.store,
        )

        return decision, ledger_entry


from .safety import evaluate_safety
from .grace import evaluate_grace
from .consensus import evaluate_consensus
from .fallback import evaluate_fallback

SAFETY_REGISTRY = CompartmentRegistry("safety", evaluator=evaluate_safety)
GRACE_REGISTRY = CompartmentRegistry("grace", evaluator=evaluate_grace)
CONSENSUS_REGISTRY = CompartmentRegistry("consensus", evaluator=evaluate_consensus)
FALLBACK_REGISTRY = CompartmentRegistry("fallback", evaluator=evaluate_fallback)
