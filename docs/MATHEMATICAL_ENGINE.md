# AILEE Mathematical Engine

## Scope and canonical equation

`ailee.math_engine` is the v9.4 deterministic reference implementation of the
founding, propulsion-derived AILEE Δv equation. Its formula identifier is
`ailee-delta-v/v1`. It makes the expression reproducible; it is not experimental
validation or a spacecraft-performance prediction. Without a caller-supplied
calibrated unit convention, the result is equation-defined AILEE Δv / an
optimization-gain metric, not certified m/s.

```text
prefactor = Isp * eta * exp(-alpha * v0**2)
integrand(t) = (P_input(t) * exp(-alpha * w(t)**2)
                * exp(2 * alpha * v0) * v(t)) / M(t)
delta_v = prefactor * integral(integrand(t), t0, tf)
```

The numerator uses `exp(2 * alpha * v0) * v(t)`, **not**
`exp(2 * alpha * v0 * v(t))`. This is not the Tsiolkovsky equation: no `g0` or
`ln(m0/mf)` is inserted.

## Models and variable map

All public models are frozen dataclasses. `DeltaVParameters` contains specific
impulse `Isp`, efficiency `eta`, coefficient `alpha`, and initial velocity
`v0`. Each `DeltaVSample` maps `time`, `input_power`, `workload`, `velocity`,
and `mass` directly to `t`, `P_input(t)`, `w(t)`, `v(t)`, and `M(t)`.
`DeltaVResult` records Δv, the integral, prefactor, sample count and time bounds,
integration method, and formula version—never the full stream.

## Integration, validation, and numerical limits

Version 1 evaluates the integrand at every ordered sample, then uses a fixed,
O(n) composite trapezoidal sum:

```text
sum(0.5 * (f[i] + f[i+1]) * (t[i+1] - t[i]))
```

There is no adaptation, interpolation, randomness, or runtime heuristic. At
least two samples and finite, strictly increasing times are required. Booleans
are not numbers. `Isp > 0`, `0 <= eta <= 1`, `alpha >= 0`, `P_input >= 0`, and
`M > 0`; `v0`, workload, and signed velocity must be finite. Malformed samples,
NaN, and infinities raise `MathValidationError`. Exponential overflow and
non-finite intermediate, integral, prefactor, or output values raise
`MathNumericalError` rather than being clamped. Floating-point exponential
underflow to zero is legitimate. Normal binary floating-point rounding remains.

## Use

```python
from ailee.math_engine import (
    DeltaVParameters, DeltaVSample, attach_delta_v_evidence, compute_delta_v,
)

parameters = DeltaVParameters(300.0, 0.82, 0.002, 4.0)
samples = (
    DeltaVSample(0.0, 12.0, 1.0, 4.0, 100.0),
    DeltaVSample(2.0, 13.0, 1.2, 4.5, 99.0),
)
result = compute_delta_v(parameters, samples)
context = attach_delta_v_evidence({"mission": "illustrative"}, result)
decision = pipeline.process(raw_value=signal, raw_confidence=.95,
                            peer_values=peers, context=context)
```

See the runnable, explicitly illustrative propulsion example at
[`examples/delta_v_reference.py`](../examples/delta_v_reference.py).

## Evidence and trust boundary

`build_delta_v_evidence` creates frozen, compact evidence with Δv, identifiers,
sample count, and time bounds. `attach_delta_v_evidence` deep-copies caller
context and adds a serializable snapshot under `ailee_delta_v`; it neither
retains the trajectory nor mutates caller data. Inputs are untrusted and
mathematical failures remain explicit.

```text
             AILEE FOUNDATIONAL SERVICES

        +-------------------------------+
        | Math Engine                   |
        | Delta-v / mathematical evidence|
        +---------------+---------------+
                        | evidence
                        v
        +-------------------------------+
        | AILEE Trust Pipeline          |
        | Safety / GRACE / Consensus    |
        | Fallback / Audit              |
        +-------------------------------+

Agentic AI / Tools -> AILEE Local Computing -> OS-native enforcement
```

The pipeline merely preserves this context in audit metadata; no decision rule
consumes the evidence. Δv neither allows nor denies, bypasses pipeline layers,
nor proves trustworthiness. Local Computing separately continues to distinguish
policy decision, platform capability, and enforcement result. The Math Engine
is not in a kernel or OS enforcement layer, and mathematical failure never
grants authorization.
