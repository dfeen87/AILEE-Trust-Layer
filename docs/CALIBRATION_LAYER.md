# AILEE V8.1 Calibration Layer

## Purpose and compatibility

The V8.1 Calibration Layer is an **opt-in, deterministic refinement** for confidence values supplied to the existing Brooks V8 trust pipeline. It is deliberately a wrapper: it does not change V8 telemetry scoring, guard decisions, consensus engine behavior, fallback selection, or hard bounds. With its default `enabled: false` configuration, the Brooks domain supplies precisely the V8 baseline confidence to `AileeTrustPipeline`.

When enabled, calibration runs after the Brooks rule checks produce baseline snapshot confidence and before the unchanged trust pipeline evaluates that confidence. Invalid calibration inputs always retain the V8 baseline; they never create a permissive result.

## Configuration and metadata

Pass calibration configuration as the fifth `BrooksDomain` constructor argument. All score-like values are normalized to `[0, 1]`; invalid configuration fails safe.

```ts
const domain = new BrooksDomain("mfc_line_a", undefined, undefined, 1000, {
  enabled: true,
  acceptanceThreshold: 0.95,
  uncertaintyBand: 0.05,
  maxGraceMargin: 0.02,
  consensusThreshold: 0.8,
  minimumPeerCount: 2,
});
```

Per-decision metadata is supplied through the existing `trustContext` object:

```ts
await domain.evaluateState(snapshot, {
  calibration: {
    graceMargin: 0.01,
    peerConsensus: { agreement: 0.9, peerCount: 2 },
  },
});
```

- `acceptanceThreshold` is the threshold calibration may reach, not exceed.
- `uncertaintyBand` defines `[acceptanceThreshold - uncertaintyBand, acceptanceThreshold)`.
- `graceMargin` is capped by `maxGraceMargin` and by the distance to the acceptance threshold.
- `peerConsensus` must meet both the configured agreement and peer-count requirements. The Brooks V8 pipeline has no peer values of its own, so this is an explicit external signal rather than a replacement for V8 consensus.

## Decision behavior and audit data

| Baseline confidence | Peer consensus | Result |
| --- | --- | --- |
| At/above threshold | Any | Baseline retained. |
| In uncertainty band | Meets requirements | Add bounded grace, up to the threshold. |
| In uncertainty band | Missing or insufficient | Baseline retained. |
| Below uncertainty band | Any | Baseline retained. |
| Invalid config/metadata | Any | Baseline retained and fail-safe event recorded. |

For example, a baseline of `0.94` with a `0.95` threshold, `0.02` maximum grace, and qualifying `0.90` peer agreement becomes `0.95` (grace is capped at the `0.01` distance to threshold). The same `0.94` input remains `0.94` when calibration is disabled, consensus is absent, or metadata is invalid.

The returned `DecisionResult.context.calibration` contains deterministic audit fields: `event`, `thresholdDecision`, `consensusChecked`, `fallbackUsed`, `applied`, `confidence`, and human-readable `reasons`. These fields make calibration events, threshold decisions, fallback triggers, and consensus checks available to existing structured decision logging without adding a logging dependency.

## Degraded operation and V8.2 extension

Malformed metadata (non-finite/out-of-range agreement, negative grace, invalid peer counts), invalid calibration configuration, or a disabled layer leaves the V8 baseline unchanged. Standard V8 safety rejection and hardware fallback continue to operate afterwards.

V8.2 can add domain-specific, versioned consensus evidence or calibration profiles while keeping this boundary: normalize external evidence first, invoke the pure `CalibrationLayer`, and continue to delegate safety/fallback decisions to the established V8 pipeline.
