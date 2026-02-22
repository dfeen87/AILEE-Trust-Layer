# AILEE Trust Layer — Crypto Mining Domain

> *High-throughput AI governance for hash rate, thermal, power, and pool decisions in production mining environments.*

---

## Why This Domain Exists

Crypto mining rigs operate at the intersection of **continuous AI-driven optimization** and **real hardware risk**.  
Modern mining software uses AI and statistical models to make rapid decisions:

- Tune hash rate for maximum throughput
- Throttle or cool GPUs before damage occurs
- Switch mining pools for improved profitability
- Adjust power caps to balance performance and efficiency

These decisions happen **hundreds of times per hour**, driven by sensor data, market signals, and model outputs. When a model is wrong — or when its confidence is not justified — the consequences are immediate and physical:

- Thermal runaway → hardware damage
- Premature pool switches → orphaned shares, lost revenue
- Unchecked power draw → tripped circuit breakers, infrastructure failure
- Hash rate instability → reduced block discovery rates

**AILEE does not optimize mining. AILEE governs whether a proposed optimization is safe enough to act on.**

---

## The Problem AILEE Solves

Without a trust layer, a mining AI pipeline looks like this:

```
Model Output  →  Hardware Actuator
```

With AILEE, every AI-proposed action passes through a multi-stage governance pipeline:

```
Model Output
    │
    ▼
Safety Layer       ← Is the value within safe bounds? Is confidence sufficient?
    │
    ▼
Grace Layer        ← Is a borderline decision consistent with recent trend & peer sensors?
    │
    ▼
Consensus Layer    ← Do multiple hardware sensors agree within tolerance?
    │
    ▼
Fallback           ← If consensus fails, use the last known-good value
    │
    ▼
Trusted Decision   ← Only then: act on the value
```

This architecture enforces **structural restraint**: an AI system cannot override hardware unless it has earned sufficient trust across all four stages.

---

## Core Concepts

### Trust Levels

Every governance decision is assigned one of four trust levels:

| Level | Value | Meaning |
|-------|-------|---------|
| `NO_ACTION` | 0 | Insufficient confidence or consensus — do not change hardware state |
| `ADVISORY` | 1 | Log or alert only — no autonomous hardware change |
| `SUPERVISED` | 2 | Act, but flag for operator review |
| `AUTONOMOUS` | 3 | Fully authorized autonomous action |

A decision is **actionable** only when its trust level meets or exceeds the policy's `min_trust_for_action` (default: `SUPERVISED`).

### Mining Domains

AILEE governs five distinct subsystems within a mining operation:

| Domain | What is Governed | Config |
|--------|-----------------|--------|
| `HASH_RATE` | MH/s tuning and overclock decisions | `HASH_RATE_OPTIMIZATION` |
| `THERMAL` | Temperature thresholds and throttle commands | `THERMAL_PROTECTION` |
| `POWER` | Per-rig wattage limits and power capping | `POWER_MANAGEMENT` |
| `POOL_MANAGEMENT` | Pool selection, switching, and profitability scoring | `POOL_SWITCHING` |
| `HARDWARE` | Restart, reset, and maintenance gating | `THERMAL_PROTECTION` |

### Domain-Specific Configurations

Each domain ships with purpose-built `AileeConfig` presets that encode real mining constraints:

**`THERMAL_PROTECTION`** (strictest)
- Accept threshold: **0.95** — hardware damage risk demands near-certainty
- Hard max: **95 °C** — enforced absolute ceiling
- Consensus quorum: **4 sensors** — multi-GPU agreement required
- Stability weight: **0.55** — thermal inertia must be respected

**`HASH_RATE_OPTIMIZATION`** (moderate)
- Accept threshold: **0.90** — manageable change risk
- Consensus quorum: **3 rig sensors**
- Consensus delta: **2 MH/s** — tight agreement window
- Stability weight: **0.40**

**`POWER_MANAGEMENT`** (high confidence required)
- Accept threshold: **0.93** — power changes carry infrastructure risk
- Consensus delta: **10 W**
- Stability weight: **0.50**

**`POOL_SWITCHING`** (business-level, moderate)
- Accept threshold: **0.88**
- Pool latency scores in `[0, 1]`
- Pool-switch rate limiting enforced via policy

---

## Safety Mechanisms

### Thermal Override

The governor enforces a **hard thermal override** independent of the AI pipeline. If any observed hardware sensor reading reaches or exceeds `policy.thermal_throttle_temp_c` (default: **85 °C**), the decision is immediately set to `NO_ACTION` with a `CRITICAL` status — without consulting the model or pipeline.

```python
# Always fires before the pipeline runs
if max_observed_temp >= policy.thermal_throttle_temp_c:
    return NO_ACTION(reason="Thermal override: 87.3°C exceeds 85.0°C threshold")
```

This guarantees that **no confidence level, no matter how high, can authorize a hardware action at unsafe temperatures**.

### Pool-Switch Rate Limiting

Rapid pool switching ("pool thrashing") wastes shares and degrades profitability. The governor enforces a hard limit:

```python
# Default: 5 pool switches per hour
if pool_switches_this_hour >= policy.max_pool_switches_per_hour:
    return NO_ACTION(reason="Rate limit reached: too many pool switches this hour")
```

### Fallback Stability

When consensus fails or confidence drops below thresholds, the pipeline falls back to the **last known-good value** rather than passing through a suspect value. This prevents instability from propagating into hardware.

---

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         MiningGovernor            │
                    │                                   │
                    │  policy: MiningPolicy             │
                    │  ┌───────────────────────────┐   │
                    │  │  AILEE Pipelines (×5)     │   │
                    │  │  HASH_RATE / THERMAL /    │   │
                    │  │  POWER / POOL / HARDWARE  │   │
                    │  └───────────────────────────┘   │
                    │  monitor: TrustMonitor            │
                    │  events: List[MiningEvent]        │
                    │  history: List[MiningDecision]    │
                    └──────────────────────────────────┘
                                   │
             ┌─────────────────────┼─────────────────────┐
             │                     │                     │
    ┌────────┴───────┐   ┌─────────┴──────┐   ┌─────────┴──────┐
    │  MiningSignals  │   │ MiningDecision │   │  MiningEvent   │
    │  mining_domain  │   │ authorized_lv  │   │  event_type    │
    │  proposed_action│   │ actionable     │   │  timestamp     │
    │  ai_value       │   │ trusted_value  │   │  decision      │
    │  ai_confidence  │   │ operation_stat │   │  details       │
    │  hw_readings    │   │ fallback_reason│   └────────────────┘
    └─────────────────┘   │ decision_id   │
                          │ audit trail   │
                          └───────────────┘
```

---

## Quick Start

```python
import time
from ailee.domains.crypto_mining import (
    MiningGovernor,
    MiningPolicy,
    MiningSignals,
    MiningDomain,
    MiningAction,
    HardwareReading,
    create_mining_governor,
)

# Create a governor with default policy
governor = create_mining_governor()

# Build signals from your monitoring system
signals = MiningSignals(
    mining_domain=MiningDomain.HASH_RATE,
    proposed_action=MiningAction.TUNE_HASH_RATE,
    ai_value=95.0,            # MH/s proposed by the AI optimizer
    ai_confidence=0.88,       # Model's reported confidence
    hardware_readings=[
        HardwareReading(93.5, time.time(), "rig_01"),
        HardwareReading(94.2, time.time(), "rig_02"),
        HardwareReading(93.8, time.time(), "rig_03"),
    ],
    rig_id="rig_01",
)

# Evaluate the proposed action
decision = governor.evaluate(signals)

if decision.actionable:
    miner.set_hash_rate(decision.trusted_value)
elif decision.used_fallback:
    logger.warning("Fallback active: %s", decision.fallback_reason)
else:
    logger.info(
        "Action not yet authorized: level=%s",
        decision.authorized_level.name,
    )
```

### Factory Functions

| Function | Policy Preset | Min Trust | Consensus | Notes |
|----------|--------------|-----------|-----------|-------|
| `create_mining_governor()` | Standard | `SUPERVISED` | Required | Accepts a pre-built `MiningPolicy` or keyword overrides |
| `create_default_governor()` | Balanced | `SUPERVISED` | Required | Accepts keyword overrides only (no pre-built policy) |
| `create_strict_governor()` | Safety-first | `AUTONOMOUS` | Required, max 2 pool switches/hr, thermal throttle at 80 °C | |
| `create_permissive_governor()` | Advisory | `ADVISORY` | Not required | |

---

## Governance Policy Reference

```python
@dataclass
class MiningPolicy:
    min_trust_for_action: CryptoMiningTrustLevel = SUPERVISED
    require_consensus: bool = True
    require_hardware_validation: bool = True
    max_hash_rate_change_pct: float = 10.0      # % per governance cycle
    max_pool_switches_per_hour: int = 5
    thermal_throttle_temp_c: float = 85.0       # hard override threshold
    max_power_watts: Optional[float] = None
    enable_audit_events: bool = True
    track_decision_history: bool = True
```

---

## Audit Trail

Every `evaluate()` call produces a `MiningDecision` with a complete audit trail:

```python
decision.decision_id        # SHA-256 fingerprint of the evaluation
decision.timestamp          # Unix epoch of the decision
decision.authorized_level   # Trust level granted
decision.actionable         # Whether policy allows action
decision.trusted_value      # Value it is safe to act on
decision.used_fallback      # Whether a fallback value was substituted
decision.fallback_reason    # Why fallback was used
decision.safety_flags       # List of active safety conditions
decision.reasons            # Human-readable explanation
decision.metadata           # Pipeline internals (confidence_score, safety_status, …)
```

Events are also emitted for:
- `"decision"` — normal governed evaluation
- `"fallback"` — pipeline used a fallback value
- `"thermal_override"` — hard thermal limit was triggered
- `"rate_limit"` — pool-switch rate limit was reached

---

## Design Principles

1. **Hardware-first safety** — Thermal overrides are unconditional and fire before any model evaluation.
2. **Determinism** — Given the same inputs and history, the governor always produces the same decision.
3. **Auditability** — Every decision carries a unique ID, timestamp, and human-readable rationale.
4. **Graduated trust** — Actions require earned confidence, not just a single model score.
5. **Separation of concerns** — AILEE does not optimize mining. Optimization is the AI model's job. AILEE decides whether that optimization is trustworthy enough to act on.

---

## Related Files

| File | Description |
|------|-------------|
| `ailee/domains/crypto_mining/ailee_crypto_mining_domain.py` | Full domain implementation |
| `ailee/domains/crypto_mining/__init__.py` | Public API exports |
| `tests/test_crypto_mining_domain.py` | Test suite (26 tests) |
| `BENCHMARKS.md` | Simulated performance and governance benchmarks |
| `docs/GRACE_LAYER.md` | Grace Layer specification |
| `docs/AUDIT_SCHEMA.md` | Full audit schema reference |
