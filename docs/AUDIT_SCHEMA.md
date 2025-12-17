# AILEE Audit & Decision Schema
## Traceability, Transparency, and Explainability

Version: 1.0.0  
Status: Stable (Reference Specification)

---

## Purpose

This document defines the **audit schema** used by the AILEE Trust Pipeline to ensure:
- Full decision traceability
- Deterministic explainability
- Enterprise and regulatory compatibility
- Post-hoc analysis and debugging

Every pipeline execution produces a **DecisionResult** object that can be logged,
stored, or transmitted without loss of meaning.

---

## DecisionResult (Top-Level Object)

```python
DecisionResult(
    value: float,
    safety_status: SafetyStatus,
    grace_status: GraceStatus,
    consensus_status: ConsensusStatus,
    used_fallback: bool,
    confidence_score: float,
    reasons: List[str],
    metadata: Dict[str, Any]
)
```

---

## Field Definitions

### value

The final trusted output of the pipeline.

This may be:
- The raw model output
- A consensus-approved value
- A fallback value

Downstream systems should only consume this value

---

### safety_status

Enum:
- `ACCEPTED`
- `BORDERLINE`
- `OUTRIGHT_REJECTED`

Represents the initial safety classification based on confidence scoring.

---

### grace_status

Enum:
- `PASS`
- `FAIL`
- `SKIPPED`

Indicates whether the GRACE Layer:
- Approved the borderline input
- Rejected it
- Was not invoked

---

### consensus_status

Enum:
- `PASS`
- `FAIL`
- `SKIPPED`

Indicates the outcome of peer agreement checks.

---

### used_fallback

Boolean flag indicating whether the final value came from the fallback mechanism.

This is a critical operational signal for:
- Monitoring instability
- Alerting
- Model retraining triggers

---

### confidence_score

A normalized score ∈ [0.0, 1.0] representing internal confidence.

Derived from:
- Historical stability
- Peer agreement
- Likelihood / plausibility
- Optional model-provided confidence

---

### reasons

Ordered list of human-readable explanations describing:
- Safety decisions
- Grace outcomes
- Consensus results
- Fallback triggers

Example:
```
[
  "Safety: BORDERLINE (0.73 < 0.90).",
  "Grace: PASS (trend_ok, peer_ok).",
  "Consensus: PASS (ratio=0.75)."
]
```

This list is designed for:
- Logs
- Debugging
- Incident reports
- Compliance review

---

## Metadata Object

The metadata field contains structured diagnostic data.
Its presence is configurable but strongly recommended.

### Standard Sections

#### timestamp

Unix timestamp (float) of the decision.

---

#### context

User- or system-supplied contextual information.

Examples:
- feature name
- units
- environment
- request ID

---

#### confidence_components

Breakdown of internal confidence computation:

```json
{
  "stability": 0.82,
  "variance": 0.04,
  "agreement": 0.67,
  "likelihood": 0.91,
  "raw_confidence_used": true
}
```

---

#### grace

Present only if GRACE was evaluated:

```json
{
  "passed_checks": ["trend_ok", "forecast_ok"],
  "forecast": 10.27
}
```

---

#### consensus

Present only if consensus was evaluated:

```json
{
  "peer_mean": 10.25,
  "within_ratio": 0.75,
  "raw_within_peer_mean": true,
  "delta": 0.1,
  "quorum": 3
}
```

---

#### input

Captures the original input state:

```json
{
  "raw_value": 10.3,
  "raw_confidence": 0.78,
  "peer_count": 4,
  "peer_values_sample": [10.2, 10.4, 10.3]
}
```

---

#### state

Internal pipeline state summary:

```json
{
  "history_len": 57,
  "last_good_value": 10.25
}
```

---

## Audit Guarantees

The schema guarantees:
- No silent overrides
- No hidden state transitions
- Deterministic replay
- Human-readable explanations

This enables:
- Root cause analysis
- Compliance audits
- Model comparison
- Incident review

---

## Best Practices

- Always log DecisionResult objects for borderline or fallback cases
- Monitor `used_fallback` frequency
- Track `confidence_score` trends over time
- Treat repeated GRACE failures as signals, not noise

---

## Summary

The AILEE audit schema transforms AI decisions from opaque outputs into
inspectable, accountable system actions.

Trust is not asserted.  
Trust is demonstrated.
