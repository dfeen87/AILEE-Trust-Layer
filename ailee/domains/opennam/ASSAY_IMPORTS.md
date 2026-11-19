# OpenNAM assay workbook staging — research candidate

This opt-in module (ailee.domains.opennam.assay_imports) accepts offline XLSX bytes
and returns descriptive StagedAssayRow objects for **DPRA, KeratinoSens, and h-CLAT**.
It deliberately does NOT convert them to the synthetic OpenNAM EvidenceRequest,
construct reviewed catalogs, evaluate OECD-defined approaches, determine toxicity,
interpret assay calls, claim safety, or recommend omitting animal tests.
Original adapters, governor, corpus and release version remain unchanged.

## Source and current verification boundary

The intended source is the NIH/NICEATM in-vitro workbook:
https://ntp.niehs.nih.gov/sites/default/files/2025-05/NTP-InVitro-Data_10Aug2023.xlsx

Source page:
https://ntp.niehs.nih.gov/whatwestudy/niceatm/test-method-evaluations/skin-sens/da

**The original published XLSX bytes were not available to this implementation
session.** Sheet names, column names and authentic rows have not yet been
verified. All current tests construct fictional XLSX files. This is a tested
staging parser, NOT a source-qualified NIH workbook importer or a scientifically
validated NAM analysis.

## Explicit, reviewed input mapping

The caller must select one exact assay, sheet, header row, identifier column,
assay-specific metric names, actual units and optional independent column names
for original reported call, borderline-aware call and interference.

Example with **illustrative synthetic headers, not actual NIH headers**:

~~~python
from ailee.domains.opennam.assay_imports import (
    Assay, AssayImportSpec, MetricColumn, import_assay_xlsx,
)
spec = AssayImportSpec(
    assay=Assay.DPRA, sheet_name="SyntheticData", header_row=1,
    substance_header="Substance",
    metrics=(MetricColumn("mean_depletion", "Mean depletion", "%"),),
    reported_call_header="Reported call",
    borderline_call_header="Borderline-aware call",
    interference_headers=("Interference note",),
)
with open("reviewed-source-workbook.xlsx", "rb") as stream:
    staged = import_assay_xlsx(stream.read(), spec)
~~~

The adapter retains raw numeric strings and units, greater-than/less-than
qualifiers, missing markers, uninterpreted text, reported and borderline-aware
calls as separate strings, and original interference text. The workbook SHA-256,
sheet name, and Excel row number are attached for traceability. All QC status
is explicitly UNASSESSED; it is never derived from calls or reported results.
No implicit unit conversion, call arbitration or scientific inference occurs.

The parser is offline and standard-library only. It bounds workbook bytes at
8 MiB, total uncompressed member bytes at 16 MiB, archive members at 100,
rows at 10,000, and columns at 128. Macros, external links, formulas, missing
mapped headers, unsupported metric names, and malformed inputs fail closed.
Checksum consistency is not scientific authenticity or qualified provenance.

## Required next gates before science claims

1. Obtain original NIH workbook and independently record its SHA-256/source version.
2. Review and pin exact sheet/header/metric/unit maps for the three assays.
3. Add real-workbook fixtures and compare authentic rows against primary source,
   including bound values, interference, missingness and borderline-adjusted calls.
4. Seek toxicology/domain review and prespecify any scientific validation.
5. Keep these staged data separate from the synthetic governance engine unless
   a separately reviewed contract and evidence policy authorize integration.

Testing (after installing the project test dependencies):
~~~sh
python -m pytest -q tests/test_opennam_assay_imports.py
~~~

No external network activity, dataset download, release, scientific
interpretation, animal study avoidance or regulatory conclusion is included.
