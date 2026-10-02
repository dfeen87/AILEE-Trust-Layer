"""Immutable models for the AILEE founding delta-v equation."""

from dataclasses import dataclass

FORMULA_VERSION = "ailee-delta-v/v1"
INTEGRATION_METHOD = "composite-trapezoidal"


@dataclass(frozen=True)
class DeltaVParameters:
    """Constant terms: specific impulse (Isp), eta, alpha, and v0."""

    specific_impulse: float
    efficiency: float
    alpha: float
    initial_velocity: float


@dataclass(frozen=True)
class DeltaVSample:
    """A sample of t, P_input(t), w(t), v(t), and M(t), respectively."""

    time: float
    input_power: float
    workload: float
    velocity: float
    mass: float


@dataclass(frozen=True)
class DeltaVResult:
    """Compact reproducibility record; the input trajectory is not retained."""

    delta_v: float
    integral_value: float
    prefactor: float
    sample_count: int
    start_time: float
    end_time: float
    integration_method: str = INTEGRATION_METHOD
    formula_version: str = FORMULA_VERSION
