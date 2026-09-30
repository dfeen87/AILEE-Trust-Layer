# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Vendor-neutral, supervisory/read-only industrial evidence governance."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Mapping, Optional, Protocol, Tuple, cast

SCHEMA_VERSION = "1.0"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")


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
    DUPLICATE = "DUPLICATE"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    INVALID_UNIT = "INVALID_UNIT"
    OUT_OF_ORDER = "OUT_OF_ORDER"


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
    DUPLICATE_EVIDENCE = "DUPLICATE_EVIDENCE"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"


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
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class MaterialObservation:
    observation_id: str
    machine_id: str
    timestamp: datetime
    quantity: float
    unit: str
    source: str
    received_at: datetime
    schema_version: str = SCHEMA_VERSION
    run_id: Optional[str] = None


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
    schema_version: str = SCHEMA_VERSION
    recovery_evidence_id: Optional[str] = None


@dataclass(frozen=True)
class IndustrialAuditEvidence:
    interval_id: str
    machine_id: str
    observation_ids: Tuple[str, ...]
    sources: Tuple[str, ...]
    evaluated_at: datetime
    validity: TelemetryValidity
    reason_codes: Tuple[str, ...] = ()
    domain: str = "INDUSTRIAL_PROCESS"
    schema_version: str = SCHEMA_VERSION


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
    received_at: Optional[datetime] = None
    schema_version: str = SCHEMA_VERSION


class ReadOnlyTelemetryAdapter(Protocol):
    """Ingestion-only adapter boundary; intentionally exposes no control API."""

    def observations(self, machine_id: str) -> Iterable[TelemetryObservation]: ...


class ProcessTransitionPolicy:
    """Configurable evidence gate; recovery from terminal/fault states is never implicit."""

    DEFAULT_ALLOWED = frozenset(
        {
            (ProcessState.IDLE, ProcessState.READY),
            (ProcessState.READY, ProcessState.RUNNING),
            (ProcessState.RUNNING, ProcessState.FAULT),
            (ProcessState.RUNNING, ProcessState.UNPLANNED_STOP),
            (ProcessState.RUNNING, ProcessState.PLANNED_HOLD),
            (ProcessState.RUNNING, ProcessState.COMPLETE),
            (ProcessState.PLANNED_HOLD, ProcessState.RUNNING),
        }
    )
    RECOVERY_TRANSITIONS = frozenset(
        {
            (ProcessState.FAULT, ProcessState.RUNNING),
            (ProcessState.UNPLANNED_STOP, ProcessState.RUNNING),
            (ProcessState.COMPLETE, ProcessState.RUNNING),
        }
    )

    def __init__(self, allowed=None):
        self.allowed = (
            frozenset(allowed) if allowed is not None else self.DEFAULT_ALLOWED
        )

    def permits(
        self,
        start: ProcessState,
        end: ProcessState,
        recovery_evidence_id: Optional[str] = None,
    ) -> bool:
        if not isinstance(start, ProcessState) or not isinstance(end, ProcessState):
            return False
        if start is end:
            return True
        transition = (start, end)
        if transition in self.RECOVERY_TRANSITIONS:
            return transition in self.allowed and _identifier(recovery_evidence_id)
        return transition in self.allowed


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None


def _identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


class TelemetryValidator:
    def __init__(
        self,
        max_age: timedelta = timedelta(minutes=5),
        *,
        allowed_signals=None,
        allowed_sources=None,
        allowed_units=None,
    ):
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive")
        self.max_age = max_age
        self.allowed_signals = (
            frozenset(allowed_signals) if allowed_signals is not None else None
        )
        self.allowed_sources = (
            frozenset(allowed_sources) if allowed_sources is not None else None
        )
        self.allowed_units = (
            frozenset(allowed_units) if allowed_units is not None else None
        )

    def validate(
        self, observation: Optional[TelemetryObservation], evaluated_at: datetime
    ) -> TelemetryValidity:
        if observation is None:
            return TelemetryValidity.MISSING
        if observation.schema_version != SCHEMA_VERSION:
            return TelemetryValidity.UNSUPPORTED_SCHEMA
        if not all(
            _identifier(value)
            for value in (
                observation.observation_id,
                observation.machine_id,
                observation.signal_id,
                observation.unit,
                observation.source,
            )
        ):
            return TelemetryValidity.MALFORMED
        if (
            self.allowed_sources is not None
            and observation.source not in self.allowed_sources
        ):
            return TelemetryValidity.UNKNOWN_SOURCE
        if (
            self.allowed_signals is not None
            and observation.signal_id not in self.allowed_signals
        ):
            return TelemetryValidity.MALFORMED
        if (
            self.allowed_units is not None
            and observation.unit not in self.allowed_units
        ):
            return TelemetryValidity.INVALID_UNIT
        if (
            not _aware(observation.timestamp)
            or not _aware(observation.received_at)
            or not _aware(evaluated_at)
        ):
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
        if isinstance(observation.value, (int, float)) and not _finite_number(
            observation.value
        ):
            return TelemetryValidity.INVALID_VALUE
        if observation.minimum is not None or observation.maximum is not None:
            if not _finite_number(observation.value):
                return TelemetryValidity.INVALID_VALUE
            numeric_value = cast(float, observation.value)
            if (
                (
                    observation.minimum is not None
                    and not _finite_number(observation.minimum)
                )
                or (
                    observation.maximum is not None
                    and not _finite_number(observation.maximum)
                )
                or (
                    observation.minimum is not None
                    and observation.maximum is not None
                    and observation.minimum > observation.maximum
                )
            ):
                return TelemetryValidity.INVALID_VALUE
            if observation.minimum is not None and numeric_value < observation.minimum:
                return TelemetryValidity.INVALID_VALUE
            if observation.maximum is not None and numeric_value > observation.maximum:
                return TelemetryValidity.INVALID_VALUE
        return TelemetryValidity.VALID


class ThroughputGovernor:
    """Derive throughput only from quantity change over validated RUNNING time."""

    def __init__(self, validator: Optional[TelemetryValidator] = None):
        self.validator = validator or TelemetryValidator()

    def evaluate(
        self, interval: ProcessInterval, evaluated_at: datetime
    ) -> ThroughputResult:
        reason, validity = self._validate(interval, evaluated_at)
        elapsed = 0.0
        if (
            _aware(interval.start)
            and _aware(interval.end)
            and interval.end >= interval.start
        ):
            elapsed = (interval.end - interval.start).total_seconds()
        productive = elapsed if reason is ThroughputReason.CALCULATED else 0.0
        delta = None
        unit = None
        rate = None
        if reason is ThroughputReason.CALCULATED:
            delta = interval.material_end.quantity - interval.material_start.quantity  # type: ignore[union-attr]
            unit = interval.material_start.unit  # type: ignore[union-attr]
            rate = delta / (productive / 3600.0)
        observations = (
            interval.state_observations
            if isinstance(interval.state_observations, tuple)
            else ()
        )
        observation_ids = tuple(
            o.observation_id
            for o in observations
            if isinstance(o, TelemetryObservation)
        )
        sources = [interval.source]
        sources.extend(
            o.source for o in observations if isinstance(o, TelemetryObservation)
        )
        for material in (interval.material_start, interval.material_end):
            if isinstance(material, MaterialObservation):
                observation_ids += (material.observation_id,)
                sources.append(material.source)
        audit = IndustrialAuditEvidence(
            interval.interval_id,
            interval.machine_id,
            observation_ids,
            tuple(sorted(set(sources))),
            evaluated_at,
            validity,
            (reason.value,),
        )
        decision_id = self._decision_id(interval, evaluated_at, reason)
        return ThroughputResult(
            reason is ThroughputReason.CALCULATED,
            reason,
            validity,
            elapsed,
            productive,
            delta,
            unit,
            rate,
            decision_id,
            audit,
        )

    def _validate(self, interval: ProcessInterval, evaluated_at: datetime):
        if interval.schema_version != SCHEMA_VERSION:
            return (
                ThroughputReason.UNSUPPORTED_SCHEMA,
                TelemetryValidity.UNSUPPORTED_SCHEMA,
            )
        if not all(
            _identifier(v)
            for v in (interval.interval_id, interval.machine_id, interval.source)
        ):
            return ThroughputReason.MALFORMED_TELEMETRY, TelemetryValidity.MALFORMED
        if (
            not _aware(interval.start)
            or not _aware(interval.end)
            or not _aware(evaluated_at)
            or interval.end <= interval.start
            or interval.end > evaluated_at
        ):
            return (
                ThroughputReason.INVALID_TIMESTAMP,
                TelemetryValidity.INVALID_TIMESTAMP,
            )
        if (
            not isinstance(interval.state_observations, tuple)
            or not interval.state_observations
        ):
            return ThroughputReason.MISSING_TELEMETRY, TelemetryValidity.MISSING
        if not all(
            isinstance(o, TelemetryObservation) for o in interval.state_observations
        ):
            return ThroughputReason.MALFORMED_TELEMETRY, TelemetryValidity.MALFORMED
        observation_ids = [o.observation_id for o in interval.state_observations]
        if len(set(observation_ids)) != len(observation_ids):
            return ThroughputReason.DUPLICATE_EVIDENCE, TelemetryValidity.DUPLICATE
        previous_timestamp = None
        process_state_observed = False
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
                return (
                    ThroughputReason.CONTRADICTORY_STATE,
                    TelemetryValidity.CONTRADICTORY_STATE,
                )
            if (
                observation.timestamp < interval.start
                or observation.timestamp > interval.end
            ):
                return (
                    ThroughputReason.INVALID_TIMESTAMP,
                    TelemetryValidity.INVALID_TIMESTAMP,
                )
            if (
                previous_timestamp is not None
                and observation.timestamp < previous_timestamp
            ):
                return (
                    ThroughputReason.MALFORMED_TELEMETRY,
                    TelemetryValidity.OUT_OF_ORDER,
                )
            previous_timestamp = observation.timestamp
            if observation.signal_id == "process_state":
                process_state_observed = True
                if (
                    not isinstance(interval.start_state, ProcessState)
                    or observation.value != interval.start_state.value
                ):
                    return (
                        ThroughputReason.CONTRADICTORY_STATE,
                        TelemetryValidity.CONTRADICTORY_STATE,
                    )
        if not process_state_observed:
            return ThroughputReason.MISSING_TELEMETRY, TelemetryValidity.MISSING
        if not isinstance(interval.start_state, ProcessState) or not isinstance(
            interval.end_state, ProcessState
        ):
            return (
                ThroughputReason.CONTRADICTORY_STATE,
                TelemetryValidity.CONTRADICTORY_STATE,
            )
        if interval.start_state != interval.end_state:
            return (
                ThroughputReason.CONTRADICTORY_STATE,
                TelemetryValidity.CONTRADICTORY_STATE,
            )
        if interval.start_state is ProcessState.FAULT:
            return ThroughputReason.FAULT_EXCLUDED, TelemetryValidity.ACTIVE_FAULT
        if interval.start_state is not ProcessState.RUNNING:
            return ThroughputReason.NON_PRODUCTIVE_STATE, TelemetryValidity.VALID
        if interval.material_start is None or interval.material_end is None:
            return ThroughputReason.MATERIAL_EVIDENCE_MISSING, TelemetryValidity.MISSING
        start, end = interval.material_start, interval.material_end
        if not isinstance(start, MaterialObservation) or not isinstance(
            end, MaterialObservation
        ):
            return ThroughputReason.MALFORMED_TELEMETRY, TelemetryValidity.MALFORMED
        if (
            start.schema_version != SCHEMA_VERSION
            or end.schema_version != SCHEMA_VERSION
        ):
            return (
                ThroughputReason.UNSUPPORTED_SCHEMA,
                TelemetryValidity.UNSUPPORTED_SCHEMA,
            )
        if not all(
            _identifier(v)
            for v in (
                start.observation_id,
                end.observation_id,
                start.machine_id,
                end.machine_id,
                start.source,
                end.source,
                start.unit,
                end.unit,
            )
        ):
            return ThroughputReason.UNKNOWN_SOURCE, TelemetryValidity.UNKNOWN_SOURCE
        if (
            start.observation_id == end.observation_id
            or start.observation_id in observation_ids
            or end.observation_id in observation_ids
        ):
            return ThroughputReason.DUPLICATE_EVIDENCE, TelemetryValidity.DUPLICATE
        if (
            start.machine_id != interval.machine_id
            or end.machine_id != interval.machine_id
        ):
            return (
                ThroughputReason.CONTRADICTORY_STATE,
                TelemetryValidity.CONTRADICTORY_STATE,
            )
        if (
            not _aware(start.timestamp)
            or not _aware(end.timestamp)
            or not _aware(start.received_at)
            or not _aware(end.received_at)
            or start.timestamp > end.timestamp
            or start.timestamp > start.received_at
            or end.timestamp > end.received_at
            or start.received_at > evaluated_at
            or end.received_at > evaluated_at
        ):
            return (
                ThroughputReason.INVALID_TIMESTAMP,
                TelemetryValidity.INVALID_TIMESTAMP,
            )
        if start.timestamp < interval.start or end.timestamp > interval.end:
            return (
                ThroughputReason.INVALID_TIMESTAMP,
                TelemetryValidity.INVALID_TIMESTAMP,
            )
        if start.unit != end.unit:
            return ThroughputReason.UNIT_MISMATCH, TelemetryValidity.INVALID_VALUE
        if start.source != end.source or start.run_id != end.run_id:
            return (
                ThroughputReason.CONTRADICTORY_STATE,
                TelemetryValidity.CONTRADICTORY_STATE,
            )
        if not _finite_number(start.quantity) or not _finite_number(end.quantity):
            return (
                ThroughputReason.INVALID_MATERIAL_VALUE,
                TelemetryValidity.INVALID_VALUE,
            )
        if end.quantity < start.quantity:
            return (
                ThroughputReason.INVALID_MATERIAL_VALUE,
                TelemetryValidity.INVALID_VALUE,
            )
        return ThroughputReason.CALCULATED, TelemetryValidity.VALID

    @staticmethod
    def _decision_id(
        interval: ProcessInterval, evaluated_at: datetime, reason: ThroughputReason
    ) -> str:
        observations = (
            interval.state_observations
            if isinstance(interval.state_observations, tuple)
            else ()
        )
        payload = {
            "evaluated_at": (
                evaluated_at.isoformat()
                if isinstance(evaluated_at, datetime)
                else repr(evaluated_at)
            ),
            "interval_id": interval.interval_id,
            "machine_id": interval.machine_id,
            "reason": reason.value,
            "observations": [
                o.observation_id
                for o in observations
                if isinstance(o, TelemetryObservation)
            ],
            "material": [
                getattr(interval.material_start, "observation_id", None),
                getattr(interval.material_end, "observation_id", None),
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class EventLedger:
    """Append-only event collection with deterministic chronological projection."""

    def __init__(
        self, *, evaluated_at: Optional[datetime] = None, allowed_categories=None
    ):
        self._events: list[IndustrialEvent] = []
        self._evaluated_at = evaluated_at
        self._allowed_categories = (
            frozenset(allowed_categories) if allowed_categories is not None else None
        )

    def append(self, event: IndustrialEvent) -> IndustrialEvent:
        received_at = event.received_at or event.timestamp
        if event.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported event schema")
        if (
            not all(
                _identifier(v)
                for v in (
                    event.event_id,
                    event.machine_id,
                    event.event_type,
                    event.source,
                )
            )
            or not _aware(event.timestamp)
            or not _aware(received_at)
        ):
            raise ValueError(
                "event requires identity, aware timestamp, type, machine, and source"
            )
        if event.timestamp > received_at or (
            self._evaluated_at is not None and received_at > self._evaluated_at
        ):
            raise ValueError("event chronology is invalid")
        if (
            self._allowed_categories is not None
            and event.category not in self._allowed_categories
        ):
            raise ValueError("unknown event category")
        if any(existing.event_id == event.event_id for existing in self._events):
            raise ValueError("duplicate event_id")
        if any(
            existing.machine_id == event.machine_id
            and existing.timestamp == event.timestamp
            and existing.event_type == event.event_type
            and existing.source == event.source
            and existing.metadata != event.metadata
            for existing in self._events
        ):
            raise ValueError("conflicting event evidence")
        self._events.append(event)
        return event

    def chronological(self) -> Tuple[IndustrialEvent, ...]:
        return tuple(
            sorted(self._events, key=lambda event: (event.timestamp, event.event_id))
        )
