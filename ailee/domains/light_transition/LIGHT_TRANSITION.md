# AILEE Light Transition Domain

The **light_transition** domain applies AILEE-Trust-Layer governance to data
signals carried by light: fiber optics, free-space optical links, photonic
interconnects, waveguides, laser links, and other light-transition channels.

## Scope

This domain governs whether a light-carried signal is trustworthy enough to act
upon. It evaluates:

- Photonic signal integrity and peer sensor consensus.
- Signal freshness in nanoseconds.
- Bit-error rate, SNR, eye opening, dispersion, and clock offset.
- Time-of-flight claims against a hard speed-of-light boundary.
- AILEE fallback, GRACE, consensus, and audit metadata.

## Non-goals

This domain does **not** implement a transport protocol, modulator, laser driver,
routing plane, or faster-than-light system. It explicitly treats the speed of
light as a physics boundary and flags any measured time-of-flight claim that
implies propagation above `299,792,458 m/s`.

## Quick Start

```python
from ailee.domains.light_transition import (
    create_default_governor,
    create_example_signals,
)

governor = create_default_governor()
decision = governor.evaluate(create_example_signals())

if decision.actionable:
    trusted_signal_score = decision.trusted_value
else:
    safe_fallback_reason = decision.safety_flags or decision.reasons
```

## Trust Levels

| Level | Name | Meaning |
| --- | --- | --- |
| 0 | `NO_ACTION` | Do not act on the light-transition payload. |
| 1 | `ADVISORY` | Log, monitor, or alert only. |
| 2 | `GUARDED` | Act with constraints and operator-visible audit trail. |
| 3 | `AUTONOMOUS` | Full authority for trusted light-transition action. |

## Governed Actions

- `ACCEPT_FRAME`
- `FORWARD_SIGNAL`
- `SWITCH_PATH`
- `RESYNCHRONIZE_CLOCK`
- `REDUCE_RATE`
- `ENTER_SAFE_MODE`
- `NO_ACTION`

## Design Principle

Light may carry information at the fastest physically permitted propagation
speed, but trust is still structural. AILEE governs the decision boundary between
receiving a light-transition payload and acting on it.
