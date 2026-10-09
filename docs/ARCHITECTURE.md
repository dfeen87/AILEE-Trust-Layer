# AILEE Trust Layer — Architecture Overview

## Repository Layout

### Library (`ailee/`)
The installable Python package containing the core trust pipeline, optional modules,
domain governance layers, and backend abstractions.

### Local Computing foundation (`ailee/local_computing/`)
Domain-independent governance for consequential operating-system actions. The
`common/` package owns immutable requests, deterministic policy, capability,
enforcement, audit, and failure contracts. The `linux/`, `windows/`, and
`macos/` packages implement distinct user-space adapters selected only for the
running OS. They use the hosting process's authority and do not install kernel
code, intercept unrelated computation, create an AILEE agent framework, or
require any application domain. Policy decision, platform capability,
enforcement result, operation completion, and audit delivery remain separate
evidence dimensions.

### Rust Core (`src/`, `Cargo.toml`)
Production-grade Rust implementation providing generative AI trust scoring,
consensus engines, and cryptographic lineage verification.

### Deployment Application (`ailee/web/` + static root assets)
- `ailee/web/app.py` — FastAPI web application serving the AILEE demo
- `ailee/web/models.py` — Multi-model generation and search orchestration
- `ailee/web/formatters.py` — Output formatting (JSON, text, markdown, HTML)
- `index.html`, `script.js`, `styles.css` — Frontend chat interface
- `render.yaml` — Render.com deployment configuration

### Documentation (`docs/`)
Specification documents for the GRACE Layer, Audit Schema, Versioning Policy,
AI Integration Guide, and Rust implementation details.

### Tests (`tests/`)
Unit, integration, domain, and runtime-specific tests are distributed across the
repository. The `tests/` tree includes Python domain and pipeline suites, Rust
integration tests, and native C++ coverage; Rust modules also contain unit tests,
while the TypeScript package has its own Vitest suites.

## Trust Decision State and Routing

The following diagram describes the principal routes in the Python trust
pipeline. Consensus and GRACE are configurable: a disabled or evidence-limited
stage is represented by `SKIPPED`, while a borderline decision with GRACE
disabled follows the fallback route.

```mermaid
flowchart TD
    A[Model / System Output] --> B{Within hard envelope?}
    B -->|No: hard envelope violation| F[Fallback]
    B -->|Yes| C{Safety classification}

    C -->|ACCEPTED| D{Consensus enabled?}
    D -->|No: SKIPPED| T[Trusted Output]
    D -->|Yes| E{Consensus result}
    E -->|PASS| T
    E -->|SKIPPED| T
    E -->|FAIL| F

    C -->|BORDERLINE| G{GRACE enabled?}
    G -->|No: SKIPPED| F
    G -->|Yes| H{GRACE result}
    H -->|FAIL| F
    H -->|PASS| I{Consensus enabled?}
    I -->|No: SKIPPED| T
    I -->|Yes| J{Consensus result}
    J -->|PASS| T
    J -->|SKIPPED| T
    J -->|FAIL| F

    C -->|OUTRIGHT_REJECTED| F
    F --> S[Bounded / stable fallback value]
    S --> O[Final Output]
    T --> O
```

Here, a consensus `SKIPPED` result can mean that consensus was enabled but could
not be evaluated with the available peer evidence. The output remains auditable
through explicit status, reasons, and metadata.

## Separation of Concerns

The core `ailee/` package is **independently installable**. The deployable web
application lives in `ailee/web/`, while static assets and platform configuration
remain at the repository root. The deployment application imports the core
package as a consumer.

Local Computing is likewise standalone and composable: consumers submit an
explicit `CapabilityRequest` to `LocalComputingTrust`; domain packages are not
imported by that path. **AILEE governs agency, not computation.** The OS kernel
and its native permissions remain authoritative below AILEE's user-space
decision and adapter boundary.

The v9.4.0 placement is explicitly:

```text
User / Applications
        ↓
Agentic AI / Tools / Automation
        ↓
AILEE Local Computing Trust Governance
        ↓
Supported OS Interfaces
        ↓
OS / Kernel
        ↓
Hardware
```

AILEE is kernel-aware, not kernel-invasive. Each installation governs only its
own host and only requests submitted through the public boundary; Local
Computing requires no AILEE-native agent and establishes no cross-machine
federation. Platform capability, policy authorization, enforcement, operation
completion, and audit success are separate evidence claims. The final operator
manuals and acceptance matrix are in the [Local Computing operator
manual](local_computing/README.md), with the [Linux](local_computing/LINUX.md),
[Windows](local_computing/WINDOWS.md), [macOS](local_computing/MACOS.md), and
[acceptance evidence](local_computing/EVIDENCE.md) documents alongside it.

The Python pipeline's governing decision logic is deterministic given identical
inputs, configuration, and relevant per-instance history. That guarantee applies
to decision semantics; operational metadata such as a default current timestamp
or a generated identifier can vary between executions unless the caller supplies
or controls it.

## Governance evidence and Rust consensus corrections (v10.0.1)

The governance domain validates temporal numbers and delegation cardinality
before creating a decision or updating history. It preserves exact numeric
values and treats zero time bounds as evidence. Invalid numeric policy settings
are rejected at governor construction and rechecked before evaluation because
configuration remains mutable. When scope enforcement is enabled, unknown
jurisdiction evidence produces `NO_TRUST` and `actionable=False`; an explicitly
optional jurisdiction or disabled enforcement retains its existing policy.
Denial can short-circuit later stages, so downstream validation should not be
inferred solely from a default status in the denied decision.

Rust consensus requires at least the configured minimum number of outputs that
meet its trust threshold before reporting achieved consensus. Insufficient
evidence still returns the documented degraded best-available result, with
`consensus_achieved=false` and a reason identifying the quorum shortfall. A high
degraded score is neither achieved consensus nor governance authorization.

The [post-BEDROCK review](POST_BEDROCK_V10_0_1.md) records the verified corrections
and unresolved risks. The core Python pipeline, TypeScript domain contracts,
Local Computing platform limitations, and mathematical-evidence boundary were
reviewed without changing their governing behavior in this patch.
