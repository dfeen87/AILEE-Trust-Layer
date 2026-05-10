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
Unit and integration test suite (currently empty — see Roadmap).

## Separation of Concerns

The core `ailee/` package is **independently installable**. The deployable web
application lives in `ailee/web/`, while static assets and platform configuration
remain at the repository root. The deployment application imports the core
package as a consumer.
