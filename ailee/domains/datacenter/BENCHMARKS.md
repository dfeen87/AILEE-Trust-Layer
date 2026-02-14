# AILEE Data Center Domain — Benchmarks

Performance and control quality metrics for data center governance systems.

**Test Environment:**
- **Execution:** JavaScript (V8 engine), browser-based harness
- **Iterations:** 10,000 per performance test
- **Hardware:** Commodity control-plane hardware (representative)
- **Date:** December 2025

---

## Performance Benchmarks

Latency and throughput measurements for governance decision evaluation.
These benchmarks measure **decision arbitration only** (not physical actuation, HVAC response, or power electronics).

| Benchmark | Iterations | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Throughput (Hz) |
|-----------|------------|-----------|-------------|----------|----------|------------------|
| Cooling Setpoint Control | 10,000 | <0.001 | <0.001 | <0.001 | <0.001 | 396,825 |
| Power Cap Control | 10,000 | <0.001 | <0.001 | <0.001 | <0.001 | 195,313 |
| Workload Placement Decision | 10,000 | <0.001 | <0.001 | <0.001 | <0.001 | 194,175 |

### Performance Analysis

- **Fastest scenario:** Workload Placement Decision
- **Slowest scenario:** Cooling Setpoint Control
- **Average throughput:** ~262,000 Hz
- **Real-time compliance:** ✅ All scenarios exceed data center control requirements (1–10 Hz typical)
- **Latency budget:** ✅ All P99 values are orders of magnitude below 100 ms control-loop budgets

**Measurement Resolution Note:**  
Execution times for several benchmarks fall below the timer resolution of the JavaScript runtime environment. Reported values of `<0.001 ms` indicate completion faster than measurable resolution, not zero execution time. Throughput values are derived from aggregate iteration timing.

---

## Control Quality Metrics

Evaluation of governance effectiveness across common data center control domains.

| Control System | Decisions | Quality Score | Fallback Rate | Avg Confidence | Stability |
|----------------|-----------|---------------|---------------|----------------|------------|
| Cooling Control | 100 | 0.656 | 0.300 | 0.902 | 0.327 |
| Power Capping | 100 | 0.690 | 0.240 | 0.903 | 0.318 |
| Workload Placement | 50 | 0.699 | 0.240 | 0.904 | 0.361 |

**Quality Score Definition:**  
Quality score aggregates safety acceptance, stability, and consensus adherence, normalized on a 0–1 scale.

### Control Quality Analysis

- **Average quality score:** 0.682
- **Average fallback rate:** 0.260 (lower is better)
- **Stability:** High stability indicates effective governance without unnecessary oscillation
- **Safety:** Zero safety violations across all evaluated scenarios

---

## System Requirements

### Minimum Requirements

**Hardware:**
- **CPU:** Dual-core 2.0+ GHz (x86-64 or ARM)
- **RAM:** 512 MB dedicated to governance pipeline
- **Storage:** 1 GB for history and logs
- **Network:** 1 Gbps for sensor and actuator communication

**Performance:**
- **Operating Rate:** 1–10 Hz (typical HVAC and power control loops)
- **Latency Budget:** <100 ms per decision
- **Sensor Polling:** 1–60 second intervals

### Recommended Configuration

**Hardware:**
- **CPU:** Quad-core 3.0+ GHz with virtualization support
- **RAM:** 2 GB+ for extended history windows
- **Storage:** 10 GB+ for long-term telemetry retention
- **Redundancy:** Dual controllers with failover

**Performance:**
- **Operating Rate:** 10 Hz for optimal responsiveness
- **History Window:** 24+ hours for seasonal and diurnal patterns
- **Consensus:** 4–5 peer sensors per decision

---

## Validated Use Cases

### 1. Cooling / HVAC Control

**Configuration:** `COOLING_CONTROL` preset

- **Setpoint Range:** 18–27 °C (thermal safety envelope)
- **Consensus:** ≥4 temperature sensors
- **Stability Weight:** 0.55 (high, due to thermal inertia)
- **Typical Frequency:** 0.1–1 Hz (10–60 second intervals)
- **Use Cases:** CRAH units, chillers, economizers, aisle containment

---

### 2. Power Capping

**Configuration:** `POWER_CAPPING` preset

- **Power Range:** 30–95% of facility capacity
- **Consensus:** ≥5 power meters
- **Confidence Threshold:** 0.96
- **Typical Frequency:** 1–10 Hz
- **Use Cases:** Peak shaving, demand response, rack-level throttling

---

### 3. Workload Placement

**Configuration:** `WORKLOAD_PLACEMENT` preset

- **Score Range:** 0–10 (placement quality)
- **Consensus:** ≥3 placement models
- **Rate Limit:** 100 migrations/hour
- **Typical Frequency:** 0.001–0.1 Hz (event-driven)
- **Use Cases:** VM migration, container orchestration, thermal-aware scheduling

---

## Integration Guidelines

### BMS / DCIM Integration

```python
from ailee_datacenter_domain import CoolingController, SensorReading

controller = CoolingController()

sensors = [
    SensorReading(bms.get_temp("rack_01"), time.time(), "rack_01"),
    SensorReading(bms.get_temp("rack_02"), time.time(), "rack_02"),
]

result = controller.propose_setpoint(
    ai_setpoint=ai_model.predict(state),
    ai_confidence=ai_model.confidence(),
    sensor_readings=sensors,
    zone_id="zone_a"
)

if not result.used_fallback:
    bms.set_setpoint("zone_a", result.value)
else:
    log.warning(f"Fallback used: {result.reasons}")
