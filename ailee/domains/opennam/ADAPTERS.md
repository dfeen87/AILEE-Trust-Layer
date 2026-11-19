# OpenNAM synthetic metadata adapters — experimental

This optional module (`ailee.domains.opennam.adapters`) provides a bounded
**transport and metadata-normalization foundation**. It accepts only already
categorized, original fictional records. The adapters do not perform,
interpret, combine, select, calibrate, or validate a scientific test; they do
not transform numerical readings, convert units, transfer inferences across
endpoints or methods, or derive toxicity, hazard, safety, applicability,
reliability, regulatory acceptance, or animal replacement conclusions. No
real assay adapters or laboratory data are in scope.

## Explicit profile, two fixed transports

The host independently selects an `AdapterProfile` with an allowlisted
`AdapterFormat`, exact adapter revision `1`, a synthetic profile ID and
positive integer profile revision, exact `Scope`, and catalog reference.
The profile is an immutable descriptor, not an authority or certification.
Profile revision is a host-declared identity; parser revision is closed to
`1` until reviewed code explicitly implements another revision. A source
cannot select a module, import a plugin, execute a callback or change the
selected profile. No automatic format guessing or unknown fallback exists.

Both parser families are fixed in code and use only abstract tokens:

| Selected format | Supported source dialect | Abstract output token mapping |
| --- | --- | --- |
| `JSON_ENVELOPE` (`syn-json-envelope`) | one closed UTF-8 object with `record` | `positive`, `negative`, `inconclusive` unchanged |
| `CSV_ROW` (`syn-csv-row`) | fixed ASCII header and exactly one nonempty, unquoted LF-terminated row | `ABSTRACT_POS` → positive; `ABSTRACT_NEG` → negative; `ABSTRACT_UNRESOLVED` → inconclusive |

Two fictional example profiles could pin `syn-method-a` / `syn-endpoint-a` /
`syn-unit-a` with the JSON format and `syn-method-b` / `syn-endpoint-b` /
`syn-unit-b` with CSV. A profile's scope also pins material, protocol version,
and context. The formats can normalize identical already-categorical
metadata under identical scope with different descriptors, but this is not
evidence of scientific equivalence. A method/endpoint/unit/protocol/context
change is incompatible with the existing profile. No adapters match by
similarity, unit conversions, partial scope, or permissive fallback.

### Input contract

Each `SourcePacket` contains two immutable byte strings:

- **Envelope**: ≤8 KiB, nonempty. The source declares exact adapter ID/version,
  profile ID/version, catalog ID/version, all six scope fields, one record
  ID, observed UTC whole-second timestamp, exact categorical token and a
  lowercase SHA-256 declaration for the supplied artifact.
- **Opaque artifact**: ≤64 KiB, nonempty. The adapter computes SHA-256 over
  the actual bytes and requires equality to the envelope declaration.
  Binary artifact bytes are allowed; no artifact payload parsing, path IO,
  URL fetch, semantic inspection or interpretation occurs. They must be
  fictional under the host-enforced scope. The adapter checks self-consistency
  with the supplied digest, **not** whether the metadata describes the
  artifact, whether its origin is authentic, whether a complete source was
  supplied, or whether the artifact is scientifically valid.

One batch contains 1–32 `SourcePacket`s from one exact profile and totals
at most 1 MiB of envelope plus artifact bytes (bounds checked before
decoding). One packet is one record with one artifact. No multirow source
file is split into multiple evidentiary contributions. Duplicate evidence
IDs or actual artifact byte hashes anywhere in the batch reject the entire
batch rather than counting them as independent evidence. Distinct bytes,
including a re-encoded duplicate, do not prove different studies or source
independence. The independently supplied catalog's source groups remain
necessary. There is no partial result on rejection.

JSON admits precisely the following keys, including exact nested scope and
catalog keys. The example below creates original fictional bytes — not a
real study, test method, result, or external evidence source:

```python
from hashlib import sha256
import json
from ailee.domains.opennam import CatalogRef, Scope
from ailee.domains.opennam.adapters import (
    AdapterFormat, AdapterProfile, EvidenceAdapter, SourcePacket,
)

profile = AdapterProfile(
    AdapterFormat.JSON_ENVELOPE, 1, "syn-profile-a", 1,
    Scope("syn-material", "syn-endpoint-a", "syn-method-a",
          "syn-protocol-a1", "syn-unit-a", "syn-context-a"),
    CatalogRef("syn-catalog", 1),
)
artifact = b"fictional record A; opaque fixture bytes"
envelope = json.dumps({
    **profile.to_dict(),
    "record": {
        "record_id": "syn-row-a",
        "observed_at": "2025-01-01T00:00:00Z",
        "result_token": "negative",
        "artifact_sha256": sha256(artifact).hexdigest(),
    },
}).encode("utf-8")
result = EvidenceAdapter(profile=profile).adapt((SourcePacket(envelope, artifact),))
request = result.request  # untrusted metadata, not a decision or catalog
receipt = result.receipts[0]
```

CSV uses the exported `CSV_HEADER` exactly, with columns in this order:

```text
adapter_id,adapter_version,profile_id,profile_version,catalog_id,catalog_version,material_id,endpoint_id,method_id,protocol_version,unit_id,context_id,record_id,observed_at,result_token,artifact_sha256
```

It permits only a literal header, a single unquoted row and terminal LF;
general CSV dialects (including CRLF, quoting, BOM, reordered or repeated
headers, multiline, or extra columns/rows) are intentionally not admitted.
For both formats, headers and declarations must match the explicitly selected
profile. No malformed, truncated syntax, unknown tokens, `N/A`, raw
measurements, status annotations, authority, or missing fields are silently
normalized. JSON duplicates, non-UTF-8, floats, NaN, infinite values, bool or
huge revisions are rejected; CSV revisions must be canonical positive
decimals. Controlled `AdaptationError.reason` values never echo rejected
content. Receipts and packet representations do not include raw envelope or
artifact bytes. The host must supply complete packets and enforce the synthetic
scope; a checksum does not detect an authored or consistently rehashed
truncation. The declared artifact digest comes from the untrusted envelope.

## Receipts and preserved governor authority

The result contains a canonical immutable `EvidenceRequest` plus one ordered
receipt for each record. A receipt binds adapter identity/revision, profile
identity/revision and canonical profile SHA-256 (including exact scope and
catalog), envelope **byte** SHA-256, actual artifact byte SHA-256 and the
canonical output record metadata SHA-256. `receipt_sha256` fingerprints the
receipt fields deterministically. Equivalent source encodings can produce
the same record but different receipts; changing selected profile/context or
raw source bytes changes the corresponding fingerprints. This is local
consistency linkage, not authenticity, signature, privacy/anonymization,
scientific correspondence or process attestation. A caller can construct a
self-consistent receipt or modify an entire document and declared digest.

The core governor is unchanged and does **not** consume receipts. Pass only
`result.request` to its existing evaluation method with the independent
policy, complete catalog and explicit time. The adapter never constructs a
trusted `CatalogEntry`, `EvidenceCatalog`, quality/control/applicability flag,
source group, `ReviewPolicy`, clock or `Decision`. Neither source envelope,
profile, successful adaptation, byte equality nor receipt authorizes a ready
outcome. `EvidenceGovernor` still has to independently bind every record,
check roster coverage, reporting, both controls, applicability, source
groups, chronology and categorical agreement. An unknown/not-applicable
control is **not** converted to a pass; these untrusted fields are rejected
at ingestion, and independent `ReviewStatus.UNKNOWN` remains an abstention.
No fallback for methods lacking required reviewed controls exists.

Approved-adapter enforcement, if ever wanted, would need an independently
trusted consumer policy and provenance boundary; it is not implemented by
passing a caller-created receipt around this existing gate. No dynamic
registration, root package export, hard dependency, network, persistence,
external actions, or workflow integration is introduced.

## Staged opportunities before any real adapter or release

1. **Transport adapters only (current scope):** explicit versioned schemas,
   independently selected synthetic exact profiles, opaque byte consistency,
   deterministic adapters, governed metadata, frozen contract and adversarial
   tests. The existing 40-case engineering corpus remains unchanged.
2. **Candidate new test-family metadata schema:** define its reporting,
   provenance, version, exact scope, controls and applicability contracts
   with qualified domain experts. Require supported parser choices and
   separate reviewed catalog facts; resolve any applicable-control difference
   explicitly with conservative policy review rather than silently treating
   missing/unknown/not-applicable controls as pass. Introduce schema changes
   and dedicated negative tests in separate review. No arbitrary adapter
   callbacks or numerical inference in the current domain.
3. **Scientific qualification (not attempted):** a distinct authorized project
   would need applicable expert requirements and independent scientific evidence,
   privacy/provenance review, representative qualification data, prespecified
   scientific performance criteria, external assessment, and appropriate legal
   or regulatory review. A transport adapter or this synthetic suite does not
   establish any such qualification or determine suitability for a real test.

Future endpoint/method-specific work must not weaken existing readiness
semantics, claim validation from synthetic software metrics, use actual data
without proper review, or turn a version number into a readiness claim.
