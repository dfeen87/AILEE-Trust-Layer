"""Validation and typed errors for untrusted mathematical input."""

import math
from typing import Sequence, Tuple

from .models import DeltaVParameters, DeltaVSample


class MathEngineError(ValueError):
    """Base error for an invalid or unsafe mathematical calculation."""


class MathValidationError(MathEngineError):
    """Input does not satisfy the reference equation's validation rules."""


class MathNumericalError(MathEngineError):
    """A mathematical operation overflowed or produced a non-finite value."""


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MathValidationError(f"{name} must be a real number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise MathValidationError(f"{name} must be finite")
    return number


def validate_parameters(parameters: object) -> DeltaVParameters:
    if not isinstance(parameters, DeltaVParameters):
        raise MathValidationError("parameters must be a DeltaVParameters instance")
    isp = finite_number(parameters.specific_impulse, "specific_impulse (Isp)")
    eta = finite_number(parameters.efficiency, "efficiency (eta)")
    alpha = finite_number(parameters.alpha, "alpha")
    finite_number(parameters.initial_velocity, "initial_velocity (v0)")
    if isp <= 0:
        raise MathValidationError("specific_impulse (Isp) must be > 0")
    if not 0 <= eta <= 1:
        raise MathValidationError("efficiency (eta) must be within [0, 1]")
    if alpha < 0:
        raise MathValidationError("alpha must be >= 0")
    return parameters


def validate_samples(samples: object) -> Tuple[DeltaVSample, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise MathValidationError("samples must be an ordered sequence")
    if len(samples) < 2:
        raise MathValidationError("at least two samples are required")
    previous = None
    validated = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, DeltaVSample):
            raise MathValidationError(f"samples[{index}] must be a DeltaVSample")
        time = finite_number(sample.time, f"samples[{index}].time")
        power = finite_number(sample.input_power, f"samples[{index}].input_power")
        finite_number(sample.workload, f"samples[{index}].workload")
        finite_number(sample.velocity, f"samples[{index}].velocity")
        mass = finite_number(sample.mass, f"samples[{index}].mass")
        if power < 0:
            raise MathValidationError(f"samples[{index}].input_power must be >= 0")
        if mass <= 0:
            raise MathValidationError(f"samples[{index}].mass must be > 0")
        if previous is not None and time <= previous:
            raise MathValidationError("sample times must be strictly increasing")
        previous = time
        validated.append(sample)
    return tuple(validated)
