# AILEE Trust Layer — Architecture Overview

## Repository Layout

### Library (`ailee/`)
The installable Python package containing the core trust pipeline, optional modules,
domain governance layers, and backend abstractions.

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

The Python pipeline's governing decision logic is deterministic given identical
inputs, configuration, and relevant per-instance history. That guarantee applies
to decision semantics; operational metadata such as a default current timestamp
or a generated identifier can vary between executions unless the caller supplies
or controls it.
