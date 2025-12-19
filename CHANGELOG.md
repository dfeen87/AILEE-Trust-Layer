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
