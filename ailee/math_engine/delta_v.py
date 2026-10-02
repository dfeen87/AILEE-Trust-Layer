"""Executable reference implementation of the canonical AILEE delta-v equation."""

import math
from typing import Sequence

from .integration import trapezoidal_integral
from .models import DeltaVParameters, DeltaVResult, DeltaVSample
from .validation import MathNumericalError, validate_parameters, validate_samples


def _safe_exp(exponent: float, term: str) -> float:
    if not math.isfinite(exponent):
        raise MathNumericalError(f"{term} exponent is non-finite")
    try:
        result = math.exp(exponent)
    except OverflowError as exc:
        raise MathNumericalError(f"{term} exponential overflow") from exc
    if not math.isfinite(result):
        raise MathNumericalError(f"{term} exponential is non-finite")
    return result


def compute_delta_v(
    parameters: DeltaVParameters, samples: Sequence[DeltaVSample]
) -> DeltaVResult:
    """Compute equation-defined AILEE delta-v from validated trajectory samples."""
    parameters = validate_parameters(parameters)
    series = validate_samples(samples)
    isp = float(parameters.specific_impulse)
    eta = float(parameters.efficiency)
    alpha = float(parameters.alpha)
    v0 = float(parameters.initial_velocity)

    initial_damping = _safe_exp(-(alpha * v0 * v0), "initial-velocity damping")
    velocity_factor = _safe_exp(2.0 * alpha * v0, "initial-velocity factor")
    prefactor = isp * eta * initial_damping
    if not math.isfinite(prefactor):
        raise MathNumericalError("prefactor is non-finite")

    points = []
    for index, sample in enumerate(series):
        workload_damping = _safe_exp(
            -(alpha * float(sample.workload) * float(sample.workload)),
            f"samples[{index}] workload damping",
        )
        integrand = (
            float(sample.input_power)
            * workload_damping
            * velocity_factor
            * float(sample.velocity)
        ) / float(sample.mass)
        if not math.isfinite(integrand):
            raise MathNumericalError(f"samples[{index}] integrand is non-finite")
        points.append((float(sample.time), integrand))

    integral = trapezoidal_integral(points)
    delta_v = prefactor * integral
    if not math.isfinite(delta_v):
        raise MathNumericalError("delta_v is non-finite")
    return DeltaVResult(
        delta_v=delta_v,
        integral_value=integral,
        prefactor=prefactor,
        sample_count=len(series),
        start_time=float(series[0].time),
        end_time=float(series[-1].time),
    )
