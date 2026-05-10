# Light Transition Benchmarks

The benchmark posture for the **light_transition** domain is deterministic and
simulation-friendly. It is designed to measure governance behavior rather than
optical hardware throughput.

## Recommended Scenarios

1. **Clean fiber frame acceptance**
   - BER ≤ `1e-12`
   - SNR ≥ `24 dB`
   - Eye opening ≥ `0.75`
   - Three or more peer optical readings within consensus delta

2. **Free-space degradation**
   - Rising BER and falling SNR
   - Clock offset above policy
   - Expected outcome: `ADVISORY` or `NO_ACTION`

3. **Physics-bound rejection**
   - Distance and time-of-flight imply propagation above `1.0c`
   - Expected outcome: safety flag `physics_bound_violation`

4. **Peer disagreement**
   - Sensor readings split outside consensus delta
   - Expected outcome: fallback or non-actionable decision

## Metrics to Track

- Decisions per channel.
- Fallback rate.
- Actionable decision rate.
- Average AILEE confidence score.
- Count of physics-bound, staleness, BER, SNR, and clock-offset flags.

## Example Check

```bash
pytest tests/test_light_transition_domain.py
```
