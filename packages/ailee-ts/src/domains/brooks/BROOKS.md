# Brooks Domain Technical Manual v8.3 (`@ailee/trust-layer`)

## 1) Executive Overview

`BrooksDomain` (exported from `packages/ailee-ts/src/domains/brooks/index.ts`) is the deterministic physical safety gateway between AI actuation intent and field hardware execution for:

- Brooks SLA5800 Mass Flow Controllers (MFCs)
- Pressure Controllers
- BCU Ultrasonic Flow Meters

The domain converts raw fieldbus telemetry into trusted, policy-bounded decisions before any actuator-facing command is allowed to proceed. It is designed to fail closed and block dangerous control outputs before they become physical motion or valve transitions.

### Accident Prevention Envelope

AILEE blocks or degrades unsafe outputs prior to hardware actuation for conditions including:

- Chemical gas line breaches (hazardous gas misrouting, unsafe gas switching)
- Runaway thermal reactions from abrupt flow/setpoint jumps
- Pressure spikes and unsafe differential pressure events across valves
- Zero-drift corruption at nominal zero-flow baseline
- Telemetry staleness/heartbeat loss

---

## 2) System Architecture & Data Flow

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                    AILEE Brooks Deterministic Safety Path                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Raw Fieldbus Frame Ingress                                                  │
│   - EtherNet/IP (CIP) binary payload                                        │
│   - EtherCAT (CoE PDO) binary payload                                       │
│                            │                                                 │
│                            ▼                                                 │
│ Manifest-Driven Adapter Parsing                                             │
│   - `EtherNetIPAdapter.parseCIPFrame(...)`                                  │
│   - `EtherCATAdapter.parsePDOFrame(...)`                                    │
│   - Offsets/endian rules from `configs/sla5800_manifest.json`               │
│                            │                                                 │
│                            ▼                                                 │
│ Snapshot + Rule Engine Evaluation                                           │
│   - PressureGuard / PressureDeltaGuard                                      │
│   - RampRateGuard / ZeroDriftGuard                                          │
│   - GasSafetyGuard                                                          │
│                            │                                                 │
│                            ▼                                                 │
│ Trust Pipeline Arbitration (`AileeTrustPipeline.process`)                   │
│                            │                                                 │
│                            ▼                                                 │
│ Deterministic Decision Egress                                               │
│   - APPROVED  -> pass-through                                                │
│   - DEGRADED  -> bounded/held behavior                                      │
│   - REJECTED  -> `VALVE_CLOSE` or `VALVE_HOLD`                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Determinism and Runtime Guarantees

- **Execution budget:** `evaluateState(...)` is explicitly documented as a **<2ms synchronous budget**.
- **Hot-path allocation posture:** adapter byte extraction and guard checks are implemented with `DataView` + native TypeScript checks and avoid external runtime validators.
- **Manifest-driven binary decoding:** no reflection/dynamic schema engines in the safety loop.

---

## 3) Fieldbus Protocol & Byte Layout Specifications

Source of truth: `packages/ailee-ts/src/domains/brooks/configs/sla5800_manifest.json` (24-byte frame).

> Note: repository docs refer to EtherNet/IP CIP as Big-Endian profile for integration docs, while the current adapter/manifest implementation decodes with little-endian flags. The table below includes required protocol expectations and implementation handling.

### Frame Map (Offsets 0..23)

| Field | Byte Range | Width | Type | Units | EtherNet/IP (CIP) | EtherCAT (CoE PDO) | Notes |
|---|---:|---:|---|---|---|---|---|
| Flow Rate | 0..3 | 4 | Float32 | SLPM | Big-Endian profile / current impl LE | Little-Endian | Live measured flow |
| Setpoint | 4..7 | 4 | Float32 | SLPM | Big-Endian profile / current impl LE | Little-Endian | Active target |
| Valve Position | 8..11 | 4 | Float32 | % open | Big-Endian profile / current impl LE | Little-Endian | 0..100 |
| Temperature | 12..15 | 4 | Float32 | °C | Big-Endian profile / current impl LE | Little-Endian | Device thermal telemetry |
| Zero Offset | 16..19 | 4 | Float32 | %FS | Big-Endian profile / current impl LE | Little-Endian | Baseline drift signal |
| Gas ID | 20..21 | 2 | UINT16 | Catalog ID | Big-Endian profile / current impl LE | Little-Endian | Canonical gas selector |
| Status Flags | 22..23 | 2 | UINT16 | Bitmask | Big-Endian profile / current impl LE | Little-Endian | `0x8000` fault, `0x4000` warn |

### Endianness Handling

- Adapter methods parse with `DataView.getFloat32/getUint16`.
- `true` little-endian flag is currently applied in both adapters (`ethernet_ip.ts`, `ethercat.ts`).
- Serializer methods mirror the same byte-order contract for deterministic round-trip behavior.

---

## 4) Gas Safety Classification & K-Factor Matrix

Source: `models/gas_database.ts` and `configs/gas_db.json`.

| Gas ID | Formula | Common Name | GCF / Thermal K-Factor (vs N₂=1.0) | Safety Tier(s) | Max Flow Ceiling (SLPM) | Purge Required on Change |
|---:|---|---|---:|---|---:|---|
| 1 | N2 | Nitrogen | 1.000 | INERT | 1000 | No |
| 2 | Air | Air | 1.000 | INERT | 1000 | No |
| 3 | Ar | Argon | 1.415 | INERT | 1000 | No |
| 10 | H2 | Hydrogen | 1.010 | FLAMMABLE | 100 | Yes |
| 11 | NH3 | Ammonia | 0.730 | TOXIC, CORROSIVE | 50 | Yes |
| 15 | O2 | Oxygen | 0.993 | OXIDIZER | 200 | Yes |
| 28 | SiH4 | Silane | 0.598 | PYROPHORIC | 20 | Yes |
| 42 | Cl2 | Chlorine | 0.860 | CORROSIVE, TOXIC | 30 | Yes |

Hazardous classifications used by deterministic fallback logic: `FLAMMABLE`, `TOXIC`, `CORROSIVE`, `OXIDIZER`, `PYROPHORIC`.

---

## 5) Deterministic Safety Policy Rules

Policy source: `types/policy.ts` (`DEFAULT_BROOKS_POLICY`).

### RampRateGuard

- Enforces bounded setpoint acceleration against full-scale flow.
- Default ceiling: **20% FS per 100ms** (time-normalized).
- Breach result: `OUTRIGHT_REJECTED` with `VALVE_HOLD` recommendation.
- Safety objective: prevent thermal shock, surge fronts, and unstable transients.

### ZeroDriftGuard

- Evaluates zero baseline when setpoint is exactly zero.
- Threshold: **±0.5% FS**.
- Breach result: `BORDERLINE` (`DEGRADED` semantic) warning path.
- Safety objective: detect sensor baseline corruption before it contaminates actuation quality.

### GasSafetyGuard

- Validates requested gas ID against registered catalog.
- Caps setpoint by per-gas manifold limit (`defaultMaxFlowSlpm`).
- Rejects gas switching when active flow is present (`>0.1 SLPM`).
- Enforces purge prerequisite for hazardous transitions when policy flag enabled.

### PressureDeltaGuard

- Computes `|upstream - downstream|` and blocks valve-open requests above limit.
- Default max differential: **50 PSI**.
- Safety objective: prevent valve damage, downstream shock loading, and containment risk.

---

## 6) Fallback & Emergency Override State Machine

Fallback behavior is deterministic and safety-tier aware.

### Override Matrix

- **`VALVE_CLOSE`** (hard isolation, FAULT mode)
  - Trigger classes:
    - Telemetry timeout / heartbeat stale breach
    - Absolute overpressure or overpressure trip
    - Hazardous gas setpoint/switch safety breach
    - Hazardous delta-pressure breach
  - Default for hazardous/reactive lines (`TOXIC`, `PYROPHORIC`, `CORROSIVE`, `FLAMMABLE`, `OXIDIZER`) when strict hazardous mode is enabled.

- **`VALVE_HOLD`** (freeze position, HOLD mode)
  - Trigger classes:
    - Inert-line fallback conditions
    - Non-critical degraded conditions (e.g., warning-level drift path)
  - Objective: avoid abrupt pressure collapse, vacuum upset, or unnecessary transient disturbance.

State machine integration is handled by `DeviceStateMachine` and `writeActuators(...)` in `index.ts`.

---

## 7) TypeScript API & Usage Examples

## 7.0 V8.1 Optional Calibration Layer

V8.1 adds an opt-in confidence calibration wrapper. It refines only an uncertainty-zone confidence when explicit peer-consensus metadata qualifies; the V8 guard and pipeline logic remain unchanged. See [`docs/CALIBRATION_LAYER.md`](../../../../../docs/CALIBRATION_LAYER.md) for configuration, failure semantics, and audit fields.

## 7.1 Initialize `BrooksDomain`

```ts
import { BrooksDomain } from "@ailee/trust-layer";

const brooks = new BrooksDomain("mfc_sla5800_line_a");
```

## 7.2 Ingest a raw 24-byte fieldbus frame

```ts
import { EtherNetIPAdapter } from "@ailee/trust-layer";

// Raw frame from PLC/fieldbus bridge
const raw = new ArrayBuffer(24);
const view = new DataView(raw);

// Example payload write (using current LE implementation)
view.setFloat32(0, 12.5, true);   // flowRate
view.setFloat32(4, 10.0, true);   // setpoint
view.setFloat32(8, 45.0, true);   // valvePosition
view.setFloat32(12, 23.2, true);  // temperature
view.setFloat32(16, 0.1, true);   // zeroOffset
view.setUint16(20, 1, true);      // gasId (N2)
view.setUint16(22, 0x0000, true); // statusFlags

const telemetry = EtherNetIPAdapter.parseCIPFrame(raw);
```

## 7.3 Evaluate AI actuation intent against live telemetry

```ts
import { BrooksDomain } from "@ailee/trust-layer";

const domain = new BrooksDomain("mfc_sla5800_line_a");

// Pull current baseline snapshot
const snapshot = await domain.readSensors();

// Inject parsed telemetry + AI-requested target into evaluation snapshot
snapshot.readings.flowRate = telemetry.flowRate;
snapshot.readings.setpoint = 75.0; // AI requested flow command
snapshot.readings.gasId = telemetry.gasId;
snapshot.readings.zeroOffset = telemetry.zeroOffset;
snapshot.readings.pressure = 32.0;
snapshot.readings.upstreamPressure = 60.0;
snapshot.readings.downstreamPressure = 20.0;
snapshot.readings.overpressureTrip = false;
snapshot.readings.telemetryTimestamp = Date.now();
snapshot.readings.previousTelemetryTimestamp = Date.now() - 100;
snapshot.readings.previousSetpoint = 50.0;

const decision = await domain.evaluateState(snapshot, {
  aiRequestId: "req-2026-09-13-001",
  agentId: "agent_alpha",
});
```

## 7.4 Handle `APPROVED`, `DEGRADED`, `REJECTED`

```ts
type Outcome = "APPROVED" | "DEGRADED" | "REJECTED";

function mapOutcome(safetyStatus: "ACCEPTED" | "BORDERLINE" | "OUTRIGHT_REJECTED"): Outcome {
  if (safetyStatus === "ACCEPTED") return "APPROVED";
  if (safetyStatus === "BORDERLINE") return "DEGRADED";
  return "REJECTED";
}

const outcome = mapOutcome(decision.safetyStatus);

switch (outcome) {
  case "APPROVED":
    // Forward bounded value to actuator orchestration layer
    console.log("Actuation approved", decision.value);
    break;
  case "DEGRADED":
    // Continue with caution, alert operator/observability
    console.warn("Degraded decision", decision.reasons);
    break;
  case "REJECTED":
    // `BrooksDomain` fallback path already enforces VALVE_CLOSE/VALVE_HOLD
    console.error("Rejected; safe fallback applied", decision.reasons);
    break;
}
```
