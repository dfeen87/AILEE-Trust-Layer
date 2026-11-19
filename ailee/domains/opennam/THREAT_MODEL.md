# OpenNAM research boundary and limitations

This package is experimental synthetic-only metadata engineering. Its threat
model concerns untrusted metadata submitted to an honest offline harness,
not biological risk evaluation or a general-purpose security boundary.

| Boundary or failure | Intended behavior / residual condition |
| --- | --- |
| Unknown schema, extra fields, duplicates, malformed JSON, unsupported numbers, huge request, or invalid UTC clock | Closed bounded ingress and explicit invalid result; no permissive normalization/fallback. |
| Typed caller object with nested malformed values | Revalidate and reconstruct exact typed fields before evaluation. Independently assembled catalog and policy are copied, not aliased. |
| Submitted scope, source/quality annotations or catalog authority | Only independently selected catalog and policy establish trusted fixture facts; impossible to authenticate those facts from local type checks and digest syntax. |
| Omitted, silently substituted or altered reviewed metadata | Exact scope and roster coverage, independently pinned catalog reference, per-record metadata SHA-256. Coverage is limited to that catalog, not all literature. |
| Different contexts, incomplete or contradictory quality metadata, stale/future observations, abstract categorical conflict/uncertainty | Explicit abstention, no score averaging, no scientific extrapolation or action. |
| Duplicated artifact and repeated source-group labels | Duplicates disqualify; group labels count once. Distinct labels are not proof of independence, provenance, or adequate study design. |
| Input content in decision/audit | Controlled reason codes and explicit time, plus selected snapshot and fully catalog-bound request fingerprints. No raw IDs, observations, artifacts or rejected-input fingerprint; summary can still indicate category and must not be treated as anonymization. |
| Reproduction artifact or code drift | Checked-in manifest hashes selected source, method, fixture and report files; the runner checks before replay and requires exact frozen output. This is integrity comparison against a known trusted checkout, not signed attestation. |

## Assumptions and controls outside this package

- The host selects an appropriate catalog/policy, verifies any upstream
  provenance/quality facts, supplies an explicit trustworthy clock and
  enforces the synthetic-only boundary. There are no authority/network adapters.
- The gate compares declared digest-bound metadata, not raw artifact bytes,
  actual provenance, actual method quality/application domain, quantitative
  uncertainty, biological mechanism, chemical hazard, effectiveness or safety.
  It is not validated for a real study, substance, animal replacement or a
  regulatory workflow.
- A malicious in-process Python operator, simultaneous arbitrary object
  mutation, replaced module, malicious platform or rewritten complete checkout
  is out of scope. Fingerprints are not encryption, access control, a digital
  signature, proof of scientific independence, trust authority or a complete
  audit ledger. A modified manifest can bless modified contents; an external
  trusted commit or reviewed distribution is needed to establish provenance.
- A `Decision`, including one created outside the governor, is not an approval
  token. No external operation, protocol selection, scientific result,
  automated release, treatment, classification or regulatory acceptance is
  enabled by any output.
