# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Vendor-neutral, supervisory/read-only industrial evidence governance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Mapping, Optional, Protocol, Sequence, Tuple


class ProcessState(str, Enum):
    SETUP = "SETUP"
    READY = "READY"
    RUNNING = "RUNNING"
    PLANNED_HOLD = "PLANNED_HOLD"
    UNPLANNED_STOP = "UNPLANNED_STOP"
    FAULT = "FAULT"
    COMPLETE = "COMPLETE"
    IDLE = "IDLE"


class TelemetryValidity(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    MALFORMED = "MALFORMED"
    STALE = "STALE"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    INVALID_VALUE = "INVALID_VALUE"
    CONTRADICTORY_STATE = "CONTRADICTORY_STATE"
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"
    ACTIVE_FAULT = "ACTIVE_FAULT"


class ThroughputReason(str, Enum):
    CALCULATED = "CALCULATED"
    MISSING_TELEMETRY = "MISSING_TELEMETRY"
    MALFORMED_TELEMETRY = "MALFORMED_TELEMETRY"
    STALE_TELEMETRY = "STALE_TELEMETRY"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    INVALID_MATERIAL_VALUE = "INVALID_MATERIAL_VALUE"
    MATERIAL_EVIDENCE_MISSING = "MATERIAL_EVIDENCE_MISSING"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"
    CONTRADICTORY_STATE = "CONTRADICTORY_STATE"
    FAULT_EXCLUDED = "FAULT_EXCLUDED"
    NON_PRODUCTIVE_STATE = "NON_PRODUCTIVE_STATE"


@dataclass(frozen=True)
class TelemetryObservation:
    observation_id: str
    machine_id: str
    signal_id: str
    timestamp: datetime
    value: object
    unit: str
    source: str
    received_at: datetime
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterialObservation:
    observation_id: str
    machine_id: str
    timestamp: datetime
    quantity: float
    unit: str
    source: str
    received_at: datetime


@dataclass(frozen=True)
class ProcessInterval:
    interval_id: str
    machine_id: str
    start: datetime
    end: datetime
    start_state: ProcessState
    end_state: ProcessState
    source: str
    state_observations: Tuple[TelemetryObservation, ...]
    material_start: Optional[MaterialObservation] = None
    material_end: Optional[MaterialObservation] = None


@dataclass(frozen=True)
class IndustrialAuditEvidence:
    interval_id: str
    machine_id: str
    observation_ids: Tuple[str, ...]
    sources: Tuple[str, ...]
    evaluated_at: datetime
    validity: TelemetryValidity


@dataclass(frozen=True)
class ThroughputResult:
    available: bool
    reason: ThroughputReason
    telemetry_validity: TelemetryValidity
    elapsed_seconds: float
    productive_seconds: float
    material_delta: Optional[float]
    material_unit: Optional[str]
    quantity_per_hour: Optional[float]
    decision_id: str
    audit: IndustrialAuditEvidence


@dataclass(frozen=True)
class IndustrialEvent:
    event_id: str
    machine_id: str
    timestamp: datetime
    event_type: str
    source: str
    affected_subsystem: Optional[str] = None
    severity: Optional[str] = None
    category: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


class ReadOnlyTelemetryAdapter(Protocol):
    """Ingestion-only adapter boundary; intentionally exposes no control API."""

    def observations(self, machine_id: str) -> Iterable[TelemetryObservation]: ...


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None


class TelemetryValidator:
    def __init__(self, max_age: timedelta = timedelta(minutes=5)):
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive")
        self.max_age = max_age

    def validate(self, observation: Optional[TelemetryObservation], evaluated_at: datetime) -> TelemetryValidity:
        if observation is None:
            return TelemetryValidity.MISSING
        if not all((observation.observation_id, observation.machine_id, observation.signal_id, observation.unit)):
            return TelemetryValidity.MALFORMED
        if not observation.source:
            return TelemetryValidity.UNKNOWN_SOURCE
        if not _aware(observation.timestamp) or not _aware(observation.received_at) or not _aware(evaluated_at):
            return TelemetryValidity.INVALID_TIMESTAMP
        observed = observation.timestamp.astimezone(timezone.utc)
        received = observation.received_at.astimezone(timezone.utc)
        now = evaluated_at.astimezone(timezone.utc)
        if observed > received or received > now:
            return TelemetryValidity.INVALID_TIMESTAMP
        if now - observed > self.max_age:
            return TelemetryValidity.STALE
        if observation.value is None:
            return TelemetryValidity.MISSING
        if observation.minimum is not None or observation.maximum is not None:
            if isinstance(observation.value, bool) or not isinstance(observation.value, (int, float)):
                return TelemetryValidity.INVALID_VALUE
            if observation.minimum is not None and observation.value < observation.minimum:
                return TelemetryValidity.INVALID_VALUE
            if observation.maximum is not None and observation.value > observation.maximum:
                return TelemetryValidity.INVALID_VALUE
        return TelemetryValidity.VALID


class ThroughputGovernor:
    """Derive throughput only from quantity change over validated RUNNING time."""

    def __init__(self, validator: Optional[TelemetryValidator] = None):
        self.validator = validator or TelemetryValidator()

    def evaluate(self, interval: ProcessInterval, evaluated_at: datetime) -> ThroughputResult:
        reason, validity = self._validate(interval, evaluated_at)
        elapsed = 0.0
        if _aware(interval.start) and _aware(interval.end) and interval.end >= interval.start:
            elapsed = (interval.end - interval.start).total_seconds()
        productive = elapsed if reason is ThroughputReason.CALCULATED else 0.0
        delta = None
        unit = None
        rate = None
        if reason is ThroughputReason.CALCULATED:
            delta = interval.material_end.quantity - interval.material_start.quantity  # type: ignore[union-attr]
            unit = interval.material_start.unit  # type: ignore[union-attr]
            rate = delta / (productive / 3600.0)
        observation_ids = tuple(o.observation_id for o in interval.state_observations)
        sources = [interval.source]
        sources.extend(o.source for o in interval.state_observations)
        for material in (interval.material_start, interval.material_end):
            if material:
                observation_ids += (material.observation_id,)
                sources.append(material.source)
        audit = IndustrialAuditEvidence(
            interval.interval_id, interval.machine_id, observation_ids,
            tuple(sorted(set(sources))), evaluated_at, validity,
        )
        decision_id = self._decision_id(interval, evaluated_at, reason)
        return ThroughputResult(
            reason is ThroughputReason.CALCULATED, reason, validity, elapsed,
            productive, delta, unit, rate, decision_id, audit,
        )

    def _validate(self, interval: ProcessInterval, evaluated_at: datetime):
        if not interval.interval_id or not interval.machine_id or not interval.source:
            return ThroughputReason.MALFORMED_TELEMETRY, TelemetryValidity.MALFORMED
        if not _aware(interval.start) or not _aware(interval.end) or not _aware(evaluated_at) or interval.end <= interval.start:
            return ThroughputReason.INVALID_TIMESTAMP, TelemetryValidity.INVALID_TIMESTAMP
        if not interval.state_observations:
            return ThroughputReason.MISSING_TELEMETRY, TelemetryValidity.MISSING
        for observation in interval.state_observations:
            status = self.validator.validate(observation, evaluated_at)
            if status is not TelemetryValidity.VALID:
                reasons = {
                    TelemetryValidity.MISSING: ThroughputReason.MISSING_TELEMETRY,
                    TelemetryValidity.STALE: ThroughputReason.STALE_TELEMETRY,
                    TelemetryValidity.INVALID_TIMESTAMP: ThroughputReason.INVALID_TIMESTAMP,
                    TelemetryValidity.UNKNOWN_SOURCE: ThroughputReason.UNKNOWN_SOURCE,
                }
                return reasons.get(status, ThroughputReason.MALFORMED_TELEMETRY), status
            if observation.machine_id != interval.machine_id:
                return ThroughputReason.CONTRADICTORY_STATE, TelemetryValidity.CONTRADICTORY_STATE
        if interval.start_state != interval.end_state:
            return ThroughputReason.CONTRADICTORY_STATE, TelemetryValidity.CONTRADICTORY_STATE
        if interval.start_state is ProcessState.FAULT:
            return ThroughputReason.FAULT_EXCLUDED, TelemetryValidity.ACTIVE_FAULT
        if interval.start_state is not ProcessState.RUNNING:
            return ThroughputReason.NON_PRODUCTIVE_STATE, TelemetryValidity.VALID
        if interval.material_start is None or interval.material_end is None:
            return ThroughputReason.MATERIAL_EVIDENCE_MISSING, TelemetryValidity.MISSING
        start, end = interval.material_start, interval.material_end
        if not start.source or not end.source:
            return ThroughputReason.UNKNOWN_SOURCE, TelemetryValidity.UNKNOWN_SOURCE
        if start.machine_id != interval.machine_id or end.machine_id != interval.machine_id:
            return ThroughputReason.CONTRADICTORY_STATE, TelemetryValidity.CONTRADICTORY_STATE
        if not _aware(start.timestamp) or not _aware(end.timestamp) or start.timestamp > end.timestamp:
            return ThroughputReason.INVALID_TIMESTAMP, TelemetryValidity.INVALID_TIMESTAMP
        if start.timestamp < interval.start or end.timestamp > interval.end:
            return ThroughputReason.INVALID_TIMESTAMP, TelemetryValidity.INVALID_TIMESTAMP
        if start.unit != end.unit:
            return ThroughputReason.UNIT_MISMATCH, TelemetryValidity.INVALID_VALUE
        if isinstance(start.quantity, bool) or isinstance(end.quantity, bool) or not isinstance(start.quantity, (int, float)) or not isinstance(end.quantity, (int, float)):
            return ThroughputReason.INVALID_MATERIAL_VALUE, TelemetryValidity.INVALID_VALUE
        if end.quantity < start.quantity:
            return ThroughputReason.INVALID_MATERIAL_VALUE, TelemetryValidity.INVALID_VALUE
        return ThroughputReason.CALCULATED, TelemetryValidity.VALID

    @staticmethod
    def _decision_id(interval: ProcessInterval, evaluated_at: datetime, reason: ThroughputReason) -> str:
        payload = {
            "evaluated_at": evaluated_at.isoformat() if isinstance(evaluated_at, datetime) else repr(evaluated_at),
            "interval_id": interval.interval_id, "machine_id": interval.machine_id,
            "reason": reason.value,
            "observations": [o.observation_id for o in interval.state_observations],
            "material": [interval.material_start.observation_id if interval.material_start else None,
                         interval.material_end.observation_id if interval.material_end else None],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class EventLedger:
    """Append-only event collection with deterministic chronological projection."""

    def __init__(self):
        self._events = []

    def append(self, event: IndustrialEvent) -> IndustrialEvent:
        if not event.event_id or not event.machine_id or not event.event_type or not event.source or not _aware(event.timestamp):
            raise ValueError("event requires identity, aware timestamp, type, machine, and source")
        if any(existing.event_id == event.event_id for existing in self._events):
            raise ValueError("duplicate event_id")
        self._events.append(event)
        return event

    def chronological(self) -> Tuple[IndustrialEvent, ...]:
        return tuple(sorted(self._events, key=lambda event: (event.timestamp, event.event_id)))
