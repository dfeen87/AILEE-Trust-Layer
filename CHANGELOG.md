## Validation & Assurance Roadmap (v1.8.0)

The **Governance** and **Cross-Ecosystem** domains intentionally do not include
traditional performance benchmarks.

These domains are **normative and safety-critical**, and are evaluated by
**deterministic invariants, guarantees, and restraint** rather than throughput,
latency, or optimization metrics.

- The **Governance** domain is validated through authority enforcement,
  jurisdictional scope containment, temporal correctness, delegation safety,
  and deterministic decision outcomes.
- The **Cross-Ecosystem** domain is validated through semantic fidelity,
  consent preservation, capability alignment, and safe continuity across
  incompatible platforms.

Formal documentation files (e.g., `ASSURANCE.md`, `INVARIANTS.md`) for both
domains are **actively being developed** and will be introduced once real-world
usage patterns and adversarial scenarios meaningfully inform their structure.

This approach is intentional and preserves architectural correctness while
avoiding premature or misleading evaluation artifacts.

---

## AILEE Trust Layer — v1.4.0

**Release Type:** Minor (Domain Expansion & Packaging)  
**Status:** Production / Stable

### Added
- **IMAGING domain** governance layer  
  - `domains/imaging/imaging.py` — production-grade imaging trust and QA governance  
  - `domains/imaging/__init__.py` — domain exports  
  - `domains/imaging/IMAGING.md` — imaging domain conceptual framework
- **Python packaging support**
  - `setup.py` — minimal, production-safe package configuration for installation and distribution

### Updated
- `README.md` — added IMAGING domain overview
- Root `__init__.py` — exposed IMAGING domain (non-invasive, optional)

### Notes
- No changes to the AILEE core trust pipeline
- No breaking API changes
- Existing deployments remain fully compatible

## Documentation Note (v1.3.0)

Documentation clarification:
Automotive and Power Grid governance domains were conceptually part of the AILEE architecture in v1.3.0 but were not yet accompanied by standalone domain documentation at release time.

Additionally, the Data Center documentation file was relocated to a domain-consistent path (domains/datacenter/) to align with the evolving domain structure.

These were documentation-only oversights. No behavioral, API, or governance logic changes were introduced.
