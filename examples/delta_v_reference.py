"""Illustrative (not physically calibrated) AILEE propulsion-math example."""

from ailee.math_engine import DeltaVParameters, DeltaVSample, compute_delta_v

parameters = DeltaVParameters(
    specific_impulse=300.0,  # Isp
    efficiency=0.82,  # eta
    alpha=0.002,
    initial_velocity=4.0,  # v0
)
trajectory = (
    DeltaVSample(time=0.0, input_power=12.0, workload=1.0, velocity=4.0, mass=100.0),
    DeltaVSample(time=2.0, input_power=13.0, workload=1.2, velocity=4.5, mass=99.0),
    DeltaVSample(time=5.0, input_power=14.0, workload=1.4, velocity=5.0, mass=98.0),
)

result = compute_delta_v(parameters, trajectory)
print(f"Illustrative equation-defined AILEE Δv: {result.delta_v:.6f}")
print(f"Formula: {result.formula_version}; integration: {result.integration_method}")
