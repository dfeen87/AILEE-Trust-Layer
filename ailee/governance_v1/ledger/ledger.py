# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Authoritative Ledger System (Python)
Append-only, hash-chained, tamper-evident ledgers with ALCOA metadata requirements.
"""

import json
import hashlib
import os
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..schemas import ALCOAMetadata, CompartmentDecision, GovernanceInput, LedgerEntry

GENESIS_PREVIOUS_HASH = "0" * 64


def canonical_json_dumps(obj: Any) -> str:
    """
    Serializes Python objects to canonical JSON strings:
      - Sorted keys
      - ISO 8601 formatting for datetimes
      - UTF-8 compatible
      - Compact formatting (no whitespace between separators)
    """
    def default_serializer(o: Any):
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(
        obj,
        default=default_serializer,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_entry_hash(
    entry_id: str,
    compartment: str,
    input_snapshot: Dict[str, Any],
    decision: Dict[str, Any],
    metadata: Dict[str, Any],
    previous_hash: str,
    created_at_iso: str,
) -> str:
    """
    Computes SHA-256 hash over canonical JSON representation of the entry fields.
    """
    payload_to_hash = {
        "id": entry_id,
        "compartment": compartment,
        "input_snapshot": input_snapshot,
        "decision": decision,
        "metadata": metadata,
        "previous_hash": previous_hash,
        "created_at": created_at_iso,
    }
    canonical_str = canonical_json_dumps(payload_to_hash)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class LedgerStore(ABC):
    """Abstract base class for authoritative ledger storage backends."""

    @abstractmethod
    def append(self, entry: LedgerEntry) -> LedgerEntry:
        pass

    @abstractmethod
    def get_entries(self, compartment: str) -> List[LedgerEntry]:
        pass

    @abstractmethod
    def get_latest_entry(self, compartment: str) -> Optional[LedgerEntry]:
        pass

    @abstractmethod
    def verify_integrity(self, compartment: str) -> bool:
        pass


class InMemoryLedgerStore(LedgerStore):
    """Thread-safe in-memory append-only ledger store."""

    def __init__(self):
        self._store: Dict[str, List[LedgerEntry]] = {}
        self._lock = threading.Lock()

    def append(self, entry: LedgerEntry) -> LedgerEntry:
        with self._lock:
            comp = entry.compartment
            if comp not in self._store:
                self._store[comp] = []

            # Verify chaining
            latest = self._store[comp][-1] if self._store[comp] else None
            expected_prev = latest.current_hash if latest else GENESIS_PREVIOUS_HASH
            if entry.previous_hash != expected_prev:
                raise ValueError(
                    f"Hash chain broken for compartment '{comp}'. Expected previous hash '{expected_prev}', got '{entry.previous_hash}'."
                )

            self._store[comp].append(entry)
            return entry

    def get_entries(self, compartment: str) -> List[LedgerEntry]:
        with self._lock:
            return list(self._store.get(compartment, []))

    def get_latest_entry(self, compartment: str) -> Optional[LedgerEntry]:
        with self._lock:
            entries = self._store.get(compartment, [])
            return entries[-1] if entries else None

    def verify_integrity(self, compartment: str) -> bool:
        with self._lock:
            entries = self._store.get(compartment, [])
            if not entries:
                return True

            prev_hash = GENESIS_PREVIOUS_HASH
            for entry in entries:
                if entry.previous_hash != prev_hash:
                    return False
                calc_hash = compute_entry_hash(
                    entry.id,
                    entry.compartment,
                    entry.input_snapshot,
                    entry.decision,
                    entry.metadata,
                    entry.previous_hash,
                    entry.created_at.isoformat() if isinstance(entry.created_at, datetime) else str(entry.created_at),
                )
                if calc_hash != entry.current_hash:
                    return False
                prev_hash = entry.current_hash
            return True


class FileLedgerStore(LedgerStore):
    """File-based JSON Lines append-only ledger store."""

    def __init__(self, base_dir: str = "ledger_data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _get_file_path(self, compartment: str) -> str:
        return os.path.join(self.base_dir, f"{compartment}.jsonl")

    def append(self, entry: LedgerEntry) -> LedgerEntry:
        with self._lock:
            path = self._get_file_path(entry.compartment)
            latest = self._get_latest_entry_unlocked(entry.compartment)
            expected_prev = latest.current_hash if latest else GENESIS_PREVIOUS_HASH

            if entry.previous_hash != expected_prev:
                raise ValueError(
                    f"Hash chain broken for compartment '{entry.compartment}'. Expected '{expected_prev}', got '{entry.previous_hash}'."
                )

            created_at_str = (
                entry.created_at.isoformat()
                if isinstance(entry.created_at, datetime)
                else str(entry.created_at)
            )

            record = {
                "id": entry.id,
                "compartment": entry.compartment,
                "input_snapshot": entry.input_snapshot,
                "decision": entry.decision,
                "metadata": entry.metadata,
                "previous_hash": entry.previous_hash,
                "current_hash": entry.current_hash,
                "created_at": created_at_str,
            }

            with open(path, "a", encoding="utf-8") as f:
                f.write(canonical_json_dumps(record) + "\n")

            return entry

    def _get_latest_entry_unlocked(self, compartment: str) -> Optional[LedgerEntry]:
        entries = self._get_entries_unlocked(compartment)
        return entries[-1] if entries else None

    def _get_entries_unlocked(self, compartment: str) -> List[LedgerEntry]:
        path = self._get_file_path(compartment)
        if not os.path.exists(path):
            return []

        entries: List[LedgerEntry] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                created_at = datetime.fromisoformat(data["created_at"])
                entry = LedgerEntry(
                    id=data["id"],
                    compartment=data["compartment"],
                    input_snapshot=data["input_snapshot"],
                    decision=data["decision"],
                    metadata=data["metadata"],
                    previous_hash=data["previous_hash"],
                    current_hash=data["current_hash"],
                    created_at=created_at,
                )
                entries.append(entry)
        return entries

    def get_entries(self, compartment: str) -> List[LedgerEntry]:
        with self._lock:
            return self._get_entries_unlocked(compartment)

    def get_latest_entry(self, compartment: str) -> Optional[LedgerEntry]:
        with self._lock:
            return self._get_latest_entry_unlocked(compartment)

    def verify_integrity(self, compartment: str) -> bool:
        with self._lock:
            entries = self._get_entries_unlocked(compartment)
            if not entries:
                return True

            prev_hash = GENESIS_PREVIOUS_HASH
            for entry in entries:
                if entry.previous_hash != prev_hash:
                    return False
                calc_hash = compute_entry_hash(
                    entry.id,
                    entry.compartment,
                    entry.input_snapshot,
                    entry.decision,
                    entry.metadata,
                    entry.previous_hash,
                    entry.created_at.isoformat(),
                )
                if calc_hash != entry.current_hash:
                    return False
                prev_hash = entry.current_hash
            return True


# Default global ledger store instance (in-memory)
_GLOBAL_STORE: LedgerStore = InMemoryLedgerStore()


def set_default_ledger_store(store: LedgerStore):
    global _GLOBAL_STORE
    _GLOBAL_STORE = store


def get_default_ledger_store() -> LedgerStore:
    return _GLOBAL_STORE


def verify_ledger_integrity(compartment: str, store: Optional[LedgerStore] = None) -> bool:
    """
    Public API function to verify the hash-chain integrity of a specific compartment's ledger.
    """
    target_store = store or _GLOBAL_STORE
    return target_store.verify_integrity(compartment)


def write_ledger_entry(
    compartment: str,
    input_payload: Union[GovernanceInput, Dict[str, Any]],
    decision: Union[CompartmentDecision, Dict[str, Any]],
    metadata: Optional[Union[ALCOAMetadata, Dict[str, Any]]] = None,
    store: Optional[LedgerStore] = None,
) -> LedgerEntry:
    """
    Writes an append-only entry to the specified compartment ledger.
    """
    target_store = store or _GLOBAL_STORE

    # Normalize input snapshot
    if isinstance(input_payload, GovernanceInput):
        input_snapshot = {
            "request_id": input_payload.request_id,
            "raw_input": input_payload.raw_input,
            "user_context": input_payload.user_context,
            "candidate_outputs": input_payload.candidate_outputs,
            "conversation_state": input_payload.conversation_state,
            "error_flags": input_payload.error_flags,
        }
    else:
        input_snapshot = dict(input_payload)

    # Normalize decision
    if isinstance(decision, CompartmentDecision):
        decision_dict = {
            "id": decision.id,
            "compartment": decision.compartment,
            "payload": decision.payload,
            "created_at": decision.created_at.isoformat(),
        }
    else:
        decision_dict = dict(decision)

    # Normalize ALCOA metadata
    if metadata is None:
        alcoa = ALCOAMetadata(
            attributable_to="system_governance_v1",
            system_id="ailee_trust_layer_v1",
            version="9.1.1",
        )
        metadata_dict = alcoa.to_dict()
    elif isinstance(metadata, ALCOAMetadata):
        metadata_dict = metadata.to_dict()
    else:
        metadata_dict = dict(metadata)
        # Ensure ALCOA minimal fields exist
        required_alcoa = [
            "attributable_to",
            "system_id",
            "version",
            "contemporaneous_timestamp",
            "is_original",
            "validation_status",
            "legible_format",
        ]
        for key in required_alcoa:
            if key not in metadata_dict:
                if key == "attributable_to":
                    metadata_dict["attributable_to"] = "system_governance_v1"
                elif key == "system_id":
                    metadata_dict["system_id"] = "ailee_trust_layer_v1"
                elif key == "version":
                    metadata_dict["version"] = "9.1.1"
                elif key == "contemporaneous_timestamp":
                    metadata_dict["contemporaneous_timestamp"] = datetime.utcnow().isoformat()
                elif key == "is_original":
                    metadata_dict["is_original"] = True
                elif key == "validation_status":
                    metadata_dict["validation_status"] = "VALID"
                elif key == "legible_format":
                    metadata_dict["legible_format"] = "json_v1"

    latest = target_store.get_latest_entry(compartment)
    previous_hash = latest.current_hash if latest else GENESIS_PREVIOUS_HASH

    entry_id = f"entry-{uuid.uuid4().hex[:12]}"
    created_at = datetime.utcnow()
    created_at_iso = created_at.isoformat()

    current_hash = compute_entry_hash(
        entry_id=entry_id,
        compartment=compartment,
        input_snapshot=input_snapshot,
        decision=decision_dict,
        metadata=metadata_dict,
        previous_hash=previous_hash,
        created_at_iso=created_at_iso,
    )

    entry = LedgerEntry(
        id=entry_id,
        compartment=compartment,
        input_snapshot=input_snapshot,
        decision=decision_dict,
        metadata=metadata_dict,
        previous_hash=previous_hash,
        current_hash=current_hash,
        created_at=created_at,
    )

    return target_store.append(entry)
