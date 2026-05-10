# AILEE Memory Domain — Benchmarks

Performance and control quality metrics for memory governance systems.

**Test Environment:**
- **Runtime:** Python 3.12 (CPython)
- **Iterations:** 10,000 per performance test; 200 decisions per quality scenario
- **Warm-up:** 200 iterations before quality measurement; 100 iterations before latency measurement
- **Hardware:** Commodity control-plane hardware (representative)
- **Date:** May 2026
- **Version:** AILEE Trust Layer v4.7.0

---

## Performance Benchmarks

Latency and throughput measurements for governance decision evaluation.
These benchmarks measure **decision arbitration only** — not physical memory operations, kernel
allocations, or OS-level swap activity.

| Benchmark | Iterations | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Throughput (Hz) |
|-----------|------------|-----------|-------------|----------|----------|------------------|
| RAM Allocation Decision | 10,000 | 0.198 | 0.194 | 0.215 | 0.233 | 5,044 |
| Heap Monitoring Decision | 10,000 | 0.208 | 0.198 | 0.216 | 0.240 | 4,797 |
| Swap Management Decision | 10,000 | 0.141 | 0.126 | 0.136 | 0.160 | 7,090 |
| Process Memory Decision | 10,000 | 0.131 | 0.117 | 0.128 | 0.150 | 7,640 |
| Signal Validation | 10,000 | <0.001 | <0.001 | <0.001 | <0.001 | 1,598,016 |
| OOM Override (hard path) | 10,000 | 0.008 | 0.007 | 0.009 | 0.019 | 132,282 |

### Performance Analysis

- **Fastest decision path:** Process Memory (`~7,640 Hz`)
- **Slowest decision path:** Heap Monitoring (`~4,797 Hz`)
- **OOM override (pre-pipeline hard path):** `~132,000 Hz` — 17× faster than full evaluation
- **Signal validation (standalone):** `~1.6 MHz` — negligible overhead for pre-call checks
- **Real-time compliance:** ✅ All scenarios exceed OS memory management requirements (1–100 Hz typical)
- **Latency budget:** ✅ All P99 values are well below 10 ms control-loop budgets

**OOM Override Fast-Path:**  
The OOM override executes before any pipeline evaluation. Its latency (`~0.008 ms`) is over 20× lower
than a full pipeline evaluation, ensuring that imminent out-of-memory conditions are handled
immediately without incurring normal governance overhead.

---

## Control Quality Metrics

Evaluation of governance effectiveness across the four memory subsystem domains.
Each scenario uses domain-appropriate signal patterns that reflect real workload behavior:

- **RAM Allocation:** sinusoidal server-load pattern (gradual oscillation)
- **Heap Monitoring:** sawtooth pattern (JVM-style allocation/GC cycles)
- **Swap Management:** low-amplitude sinusoidal (swap is typically stable)
- **Process Memory:** random-walk pattern (individual process footprint drift)

Mixed-confidence evaluation: 80% high-confidence, 10% borderline, 10% low-confidence signals.

| Control System | Decisions | Quality Score | Fallback Rate | Avg Confidence | Actionable Rate | Stability |
|----------------|-----------|---------------|---------------|----------------|-----------------|------------|
| RAM Allocation | 200 | 0.959 | 0.020 | 0.897 | 0.980 | 0.980 |
| Heap Monitoring | 200 | 0.959 | 0.025 | 0.909 | 0.975 | 0.975 |
| Swap Management | 200 | 0.976 | 0.005 | 0.921 | 0.995 | 0.995 |
| Process Memory | 200 | 0.940 | 0.040 | 0.881 | 0.960 | 0.960 |

**Quality Score Definition:**  
`Quality = (1 − Fallback Rate) × 0.40 + Actionable Rate × 0.35 + Avg Confidence × 0.25`  
Aggregates stability, actionability, and sensor confidence on a 0–1 scale.

### Control Quality Analysis

- **Average quality score:** 0.959
- **Average fallback rate:** 0.023 — consistent governance with well-established history
- **Average actionable rate:** 0.978 — governance decisions are acted upon in ~98% of cycles
- **Swap Management:** highest stability (0.995) — conservative threshold prevents unnecessary swap events
- **Process Memory:** most conservative actionable rate (0.960) — appropriate for OOM-kill gating
- **Safety violations:** Zero safety violations across all evaluated scenarios

> **Note:** HEAP and SWAP domains require domain-appropriate signal patterns with meaningful variance to
> reach peak performance. Flat or zero-variance inputs will trigger the conservative `likelihood` scorer,
> yielding borderline status and Grace-layer evaluation. This is intentional: these domains are calibrated
> for real workload signals (JVM cycles, swappiness curves), not synthetic constants.

---

## OOM Override Behavior

The OOM override is a pre-pipeline hard-safety mechanism. When **any** sensor reading in a
`RAM_ALLOCATION` evaluation exceeds `oom_emergency_threshold` (default: 0.97), the decision is
immediately rejected without pipeline evaluation.

| Scenario | Max Peer Utilization | Override Triggered | Decision | Latency |
|----------|---------------------|-------------------|----------|---------|
| Normal operation | 0.75 | ❌ | Full pipeline evaluation | ~0.198 ms |
| Near-threshold | 0.95 | ❌ | Full pipeline evaluation | ~0.198 ms |
| OOM threshold crossed | 0.99 | ✅ | Immediate NO_ACTION + CRITICAL | ~0.008 ms |

---

## OOM Kill Rate-Limiting

The governor tracks `KILL_PROCESS` authorizations per rolling hour. Once `max_oom_kills_per_hour`
(default: 5) is reached, further kill authorizations are blocked until the hour resets.

| Configuration | Kills Allowed/Hour | Behavior After Limit |
|---------------|-------------------|----------------------|
| Default (`max_oom_kills_per_hour=5`) | 5 | Block with `rate_limit` flag, WARNING health |
| Strict (`create_strict_governor`) | 2 | Block with `rate_limit` flag, WARNING health |
| Permissive (`create_permissive_governor`) | 5 (default) | Block with `rate_limit` flag |

---

## System Requirements

### Minimum Requirements

**Hardware:**
- **CPU:** Single-core 1.5+ GHz (x86-64 or ARM)
- **RAM:** 64 MB dedicated to governance pipeline
- **Storage:** 256 MB for decision history and audit logs

**Performance:**
- **Operating Rate:** 1–100 Hz (typical OS memory management loops)
- **Latency Budget:** <10 ms per decision (P99 comfortably met)
- **Sensor Polling:** 100 ms–10 second intervals

### Recommended Configuration

**Hardware:**
- **CPU:** Dual-core 2.0+ GHz
- **RAM:** 512 MB+ for extended history windows and audit trails
- **Storage:** 2 GB+ for long-term telemetry retention

**Performance:**
- **Operating Rate:** 10–50 Hz for responsive memory governance
- **History Window:** 120+ samples (configurable)
- **Consensus:** 3–4 peer sensors per decision (default: 3)

---

## Validated Use Cases

### 1. RAM Allocation Governance

**Configuration:** `RAM_ALLOCATION` preset

- **Utilization Range:** [0.0, 0.98] (hard safety ceiling)
- **Accept Threshold:** 0.90 confidence
- **Consensus:** ≥3 memory sensors
- **Stability Weight:** 0.50
- **Fallback:** `last_good` — preserves last successful allocation setpoint
- **Typical Frequency:** 1–50 Hz
- **Use Cases:** Hypervisor memory balloon tuning, container memory limits, kernel memory advisory

---

### 2. Heap Monitoring

**Configuration:** `HEAP_MONITORING` preset

- **Utilization Range:** [0.0, 0.95] (protects headroom for GC)
- **Accept Threshold:** 0.92 confidence (tighter — heap exhaustion is catastrophic)
- **Consensus:** ≥3 heap probes
- **Stability Weight:** 0.50
- **Fallback:** `last_good`
- **Typical Frequency:** 1–20 Hz (JVM GC cycles)
- **Use Cases:** JVM heap governance, .NET GC tuning, garbage-collector trigger gating

---

### 3. Swap Management

**Configuration:** `SWAP_MANAGEMENT` preset

- **Utilization Range:** [0.0, 1.0]
- **Accept Threshold:** 0.95 confidence (highest — swap overuse degrades system performance)
- **Stability Weight:** 0.55 (heaviest — swap thrashing must be prevented)
- **Fallback:** `last_good`
- **Typical Frequency:** 0.1–5 Hz (swap changes are slow)
- **Use Cases:** Swap partition governance, swappiness tuning, SSD swap endurance management

---

### 4. Process Memory Governance

**Configuration:** `PROCESS_MEMORY` preset

- **Utilization Range:** [0.0, 1.0]
- **Accept Threshold:** 0.88 confidence
- **Fallback:** `median` — robust against single-process spikes
- **Typical Frequency:** 1–10 Hz
- **Use Cases:** Per-process RSS limit enforcement, OOM-kill gating, container cgroup governance

---

## Integration Example

```python
import time
from ailee.domains.memory import (
    MemoryGovernor,
    MemoryPolicy,
    MemorySignals,
    MemoryReading,
    MemoryDomain,
    MemoryAction,
    create_memory_governor,
)

governor = create_memory_governor()

# Build signals from OS-level probes
readings = [
    MemoryReading(os.get_mem_util("node_01"), time.time(), "node_01"),
    MemoryReading(os.get_mem_util("node_02"), time.time(), "node_02"),
    MemoryReading(os.get_mem_util("node_03"), time.time(), "node_03"),
]

signals = MemorySignals(
    memory_domain=MemoryDomain.RAM_ALLOCATION,
    proposed_action=MemoryAction.ENFORCE_ALLOCATION_LIMIT,
    ai_value=ai_model.predict_utilization(),
    ai_confidence=ai_model.confidence(),
    memory_readings=readings,
    node_id="prod_host_01",
)

decision = governor.evaluate(signals)

if decision.actionable:
    memory_manager.enforce_limit(decision.trusted_value)
else:
    log.warning(f"Memory action blocked: {decision.reasons}")
```
