"""Non-authoritative workflow evidence derived from delta-v results."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from .models import DeltaVResult
from .validation import MathValidationError

EVIDENCE_CONTEXT_KEY = "ailee_delta_v"


@dataclass(frozen=True)
class DeltaVEvidence:
    delta_v: float
    formula_version: str
    integration_method: str
    sample_count: int
    start_time: float
    end_time: float


def build_delta_v_evidence(result: DeltaVResult) -> DeltaVEvidence:
    if not isinstance(result, DeltaVResult):
        raise MathValidationError("result must be a DeltaVResult instance")
    return DeltaVEvidence(
        delta_v=result.delta_v,
        formula_version=result.formula_version,
        integration_method=result.integration_method,
        sample_count=result.sample_count,
        start_time=result.start_time,
        end_time=result.end_time,
    )


def attach_delta_v_evidence(
    context: Optional[Mapping[str, Any]], result: DeltaVResult
) -> Dict[str, Any]:
    """Return a deep snapshot with compact evidence; never mutate ``context``."""
    if context is not None and not isinstance(context, Mapping):
        raise MathValidationError("context must be a mapping or None")
    snapshot = deepcopy(dict(context or {}))
    snapshot[EVIDENCE_CONTEXT_KEY] = asdict(build_delta_v_evidence(result))
    return snapshot
