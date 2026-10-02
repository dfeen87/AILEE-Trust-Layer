"""Foundational, deterministic AILEE mathematical evidence engine."""

from .delta_v import compute_delta_v
from .evidence import DeltaVEvidence, attach_delta_v_evidence, build_delta_v_evidence
from .integration import trapezoidal_integral
from .models import (
    FORMULA_VERSION,
    INTEGRATION_METHOD,
    DeltaVParameters,
    DeltaVResult,
    DeltaVSample,
)
from .validation import MathEngineError, MathNumericalError, MathValidationError

__all__ = [
    "FORMULA_VERSION",
    "INTEGRATION_METHOD",
    "DeltaVParameters",
    "DeltaVSample",
    "DeltaVResult",
    "DeltaVEvidence",
    "MathEngineError",
    "MathValidationError",
    "MathNumericalError",
    "compute_delta_v",
    "trapezoidal_integral",
    "build_delta_v_evidence",
    "attach_delta_v_evidence",
]
