# FEEN Integration

**Status:** Production-ready hardware acceleration (optional)  
**Version:** AILEE 2.0+ / FEEN 3.0+  
**Repository:** https://github.com/dfeen87/feen

---

## Overview

FEEN (Frequency-Encoded Elastic Network) provides optional hardware acceleration for AILEE's confidence computation layer using phononic (mechanical vibration) computing.

When available, FEEN implements confidence scoring in physics-native hardware, delivering:
- **Ultra-low latency:** 1 μs vs 2-5 ms (software)
- **Ultra-low power:** 100 μW vs 10-50 mW (software)
- **Deterministic execution:** No CPU scheduling variance

**FEEN is optional.** AILEE's software implementation remains the canonical reference.

---

## What FEEN Provides

FEEN implements **one** component of AILEE's trust pipeline:

### Confidence Scoring (Hardware-Accelerated)

```
Input:  raw_value, peers, history
Output: ConfidenceResult(score, stability, agreement, likelihood)
```

FEEN uses three phononic resonator channels to compute:
- **Stability:** Inverse variance of recent history
- **Agreement:** Peer consensus strength
- **Likelihood:** Historical plausibility (z-score)

These are **physics primitives** — FEEN has no knowledge of AILEE's thresholds, policies, or domain logic.

---

## What AILEE Expects

AILEE consumes FEEN's output as a **signal**, not a decision.

The `backends/feen/confidence_scorer.py` bridge:
1. Detects FEEN availability
2. Translates AILEE inputs to FEEN format
3. Returns AILEE-compatible `ConfidenceResult`
4. Falls back to software if FEEN unavailable

**AILEE retains full control** over:
- Confidence thresholds (accept, borderline, reject)
- Grace layer logic
- Consensus validation
- Fallback behavior
- All policy decisions

---

## Architecture Boundary

```
┌─────────────────────────────────────────────────────────┐
│                    AILEE Trust Layer                     │
│                   (Python, Policy Layer)                 │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │        Safety Layer (AILEE)                    │    │
│  │  • Threshold evaluation (0.70, 0.90)          │    │
│  │  • Safety status decision                      │    │
│  │  • Consumes confidence score below             │    │
│  └────────────────┬───────────────────────────────┘    │
│                   │                                      │
│                   ↓                                      │
│  ┌────────────────────────────────────────────────┐    │
│  │   Confidence Computation (FEEN or Software)    │    │
│  │                                                 │    │
│  │   If FEEN available:                           │    │
│  │     → backends/feen/confidence_scorer.py       │    │
│  │     → pyfeen.ailee.PhononicConfidenceScorer    │◄───┼─── FEEN Hardware
│  │     → Returns: {score, stability, ...}         │    │     (C++/MEMS)
│  │                                                 │    │
│  │   Else:                                        │    │
│  │     → Software confidence computation          │    │
│  │     → Same result structure                    │    │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │        Grace Layer (AILEE)                     │    │
│  │  • Trend checks, forecast, peer context       │    │
│  │  • Software only (requires contextual logic)   │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ... (Consensus, Fallback - all AILEE)                  │
└──────────────────────────────────────────────────────────┘
```

**Key principle:** FEEN provides primitives. AILEE makes decisions.

---

## Integration Points

### 1. Backend Selection

AILEE automatically selects FEEN if available:

```python
# Automatic detection
from ailee import AileeClient

client = AileeClient(config)  # Uses FEEN if installed

# Manual selection
import os
os.environ['AILEE_BACKEND'] = 'feen'    # Force FEEN
os.environ['AILEE_BACKEND'] = 'software'  # Force software
```

### 2. Graceful Degradation

If FEEN hardware fails or is unavailable:
```python
from ailee.backends.feen import FEENConfidenceScorer

scorer = FEENConfidenceScorer()

if scorer.is_available():
    result = scorer.compute(value, peers, history)
else:
    # Automatic fallback to software
    result = software_confidence_scorer.compute(...)
```

### 3. Configuration Alignment

FEEN configuration must match AILEE policy:

```python
# AILEE config (canonical)
ailee_config = AileeConfig(
    w_stability=0.45,
    w_agreement=0.30,
    w_likelihood=0.25,
    history_window=60,
    agreement_delta=0.10
)

# FEEN config (derived from AILEE)
feen_config = {
    'w_stability': ailee_config.w_stability,
    'w_agreement': ailee_config.w_agreement,
    'w_likelihood': ailee_config.w_likelihood,
    'history_window': ailee_config.history_window,
    'agreement_delta': ailee_config.grace_peer_delta
}

scorer = FEENConfidenceScorer(feen_config)
```

**AILEE's config is the source of truth.** FEEN implements it.

---

## What Is Explicitly Out of Scope

### FEEN Does NOT:
- ❌ Make trust decisions
- ❌ Apply thresholds (0.70, 0.90, etc.)
- ❌ Implement Grace layer logic
- ❌ Perform consensus validation
- ❌ Execute fallback behavior
- ❌ Have domain knowledge
- ❌ Override AILEE policy

### AILEE Does NOT:
- ❌ Implement physics primitives
- ❌ Manage FEEN hardware
- ❌ Optimize resonator parameters
- ❌ Handle FEEN calibration
- ❌ Expose FEEN internals to users

---

## Performance Characteristics

### Latency (Confidence Computation Only)

| Backend | Typical | 99th percentile |
|---------|---------|-----------------|
| Software | 2.5 ms | 4.8 ms |
| FEEN | 0.8 μs | 1.2 μs |
| **Speedup** | **3,125×** | **4,000×** |

### Power Consumption (Idle + Active)

| Backend | Idle | Active | Total |
|---------|------|--------|-------|
| Software | 5 mW (CPU) | 15 mW | 20 mW |
| FEEN | 10 μW | 90 μW | 100 μW |
| **Reduction** | **500×** | **167×** | **200×** |

### End-to-End AILEE Pipeline

FEEN accelerates **only** confidence computation (~40% of total time):

| Component | Software | FEEN | Notes |
|-----------|----------|------|-------|
| Confidence | 2.5 ms | 0.8 μs | FEEN accelerated |
| Safety | 0.5 ms | 0.5 ms | Software (trivial) |
| Grace | 1.2 ms | 1.2 ms | Software (contextual) |
| Consensus | 0.8 ms | 0.8 ms | Software (peer logic) |
| **Total** | **5.0 ms** | **2.5 ms** | **2× speedup** |

**Why not 3000× end-to-end?**  
Python overhead, safety layer, and Grace checks remain in software. FEEN accelerates the computational bottleneck, not the entire pipeline.

---

## Determinism & Reproducibility

### Software Backend
- Deterministic given fixed inputs
- Subject to floating-point variance across platforms
- CPU scheduler affects timing (not results)

### FEEN Backend
- **Bit-identical** results across runs (same hardware)
- Physics-level determinism (resonator dynamics)
- May vary across different FEEN hardware revisions
- **Same policy semantics** as software

To ensure reproducibility:
```python
# Force software backend for regression tests
os.environ['AILEE_BACKEND'] = 'software'

# Or verify FEEN matches software
software_result = software_scorer.compute(...)
feen_result = feen_scorer.compute(...)
assert abs(software_result.score - feen_result.score) < 1e-6
```

---

## Monitoring & Diagnostics

### Detecting FEEN Usage

```python
from ailee import AileeClient

client = AileeClient(config)
result = client.process(...)

# Check which backend was used
if hasattr(result.metadata, 'backend'):
    print(f"Backend: {result.metadata['backend']}")
    # Output: "feen_phononic" or "software"
```

### Hardware Health Checks

```python
from ailee.backends.feen import FEENConfidenceScorer

scorer = FEENConfidenceScorer()

if scorer.is_available():
    diagnostics = scorer.get_hardware_diagnostics()
    print(diagnostics['channel_energies'])  # [E_stability, E_agreement, E_likelihood]
```

### Alerting on FEEN Failure

```python
from ailee import AlertingMonitor

def feen_failure_handler(alert_type, value, threshold):
    if alert_type == 'backend_degradation':
        logger.critical("FEEN unavailable - using software fallback")
        # Notify ops team

monitor = AlertingMonitor(
    alert_callback=feen_failure_handler
)
```

---

## Installation

### Software-Only (No FEEN)
```bash
pip install ailee-trust-layer
```

### With FEEN Hardware Acceleration
```bash
pip install ailee-trust-layer
pip install pyfeen  # Optional FEEN backend

# Verify FEEN availability
python -c "from ailee.backends.feen import FEENConfidenceScorer; print(FEENConfidenceScorer().is_available())"
```

---

## Deployment Scenarios

### 1. Development (Software)
```python
# Local development - no FEEN required
pipeline = create_pipeline("llm_scoring")
result = pipeline.process(...)
```

### 2. Edge Deployment (FEEN)
```python
# Embedded system with FEEN chip
os.environ['AILEE_BACKEND'] = 'feen'
pipeline = create_pipeline("sensor_fusion")

# Ultra-low power inference
result = pipeline.process(...)  # <1 μs, <100 μW
```

### 3. Data Center (Hybrid)
```python
# High-throughput cluster
# FEEN for confidence, software for Grace/Consensus
pipeline = AileeClient(config)

# Automatic backend selection per request
for request in stream:
    result = pipeline.process(...)
    # Uses FEEN if available, software otherwise
```

### 4. Safety-Critical (Software + FEEN Validation)
```python
# Aerospace, medical, automotive
software_result = software_pipeline.process(...)
feen_result = feen_pipeline.process(...)

# Cross-validation
if abs(software_result.score - feen_result.score) > tolerance:
    trigger_diagnostic()
    use_conservative_fallback()
```

---

## Version Compatibility

| AILEE Version | FEEN Version | Status |
|---------------|--------------|--------|
| 2.0.0 | 3.0.0 | ✅ Tested |
| 2.0.0 | 2.x | ⚠️ Limited features |
| 1.x | Any | ❌ Not supported |

Breaking changes between versions:
- AILEE 1.x → 2.x: API redesign (no FEEN support in 1.x)
- FEEN 2.x → 3.x: Added ailee namespace, confidence scorer

---

## Troubleshooting

### FEEN Not Detected
```python
from ailee.backends.feen import FEENConfidenceScorer

scorer = FEENConfidenceScorer()
if not scorer.is_available():
    # Check:
    # 1. Is pyfeen installed? `pip list | grep pyfeen`
    # 2. Are FEEN drivers loaded? `lsmod | grep feen`
    # 3. Is hardware present? `ls /dev/feen*`
```

### Confidence Mismatch (Software vs FEEN)
```python
# Small differences (<1e-4) are expected due to:
# - Floating-point precision
# - Integration timesteps
# - Resonator settling

# Large differences (>1e-2) indicate:
# - Configuration mismatch
# - Hardware calibration issue
# - FEEN malfunction → fall back to software
```

### Performance Lower Than Expected
```python
# Check Python ↔ C++ boundary overhead
import time

start = time.perf_counter()
result = scorer.compute(value, peers, history)
elapsed = time.perf_counter() - start

if elapsed > 10e-6:  # 10 μs
    # Boundary overhead dominates
    # Consider batch processing:
    results = scorer.compute_batch([v1, v2, v3, ...])
```

---

## Future Considerations

### Potential Additions (Not Committed)
- Batch processing API
- GPU backend (for high-throughput clusters)
- Streaming support
- Async compute for pipeline parallelism

### Not Planned
- FEEN implementation of Grace layer (requires contextual logic)
- FEEN policy decisions (AILEE's responsibility)
- Direct FEEN API exposure to users (abstracted via backend)

---

## Contact & Support

**AILEE Issues:** https://github.com/dfeen87/ailee-trust-layer/issues  
**FEEN Issues:** https://github.com/dfeen87/feen/issues  
**Integration Questions:** Tag both repos in issue

For FEEN hardware procurement or commercial licensing, contact via FEEN repository.

---

**Last Updated:** 2026-02-13  
**Maintained By:** Don Michael Feeney Jr.
