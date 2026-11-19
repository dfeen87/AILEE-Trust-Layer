# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Opt-in, offline staging import for skin sensitization assay workbook rows.

NOT an interpretation of OECD 497, a safety prediction, or a path to the
synthetic OpenNAM EvidenceGovernor. Published numerical data stay descriptive.
"""

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from io import BytesIO
import posixpath
import re
from typing import Optional, Tuple
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

SOURCE_URL = "https://ntp.niehs.nih.gov/sites/default/files/2025-05/NTP-InVitro-Data_10Aug2023.xlsx"
MAX_XLSX_BYTES = 8 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100
MAX_ROWS = 10000
MAX_COLUMNS = 128
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_CELL = re.compile(r"([A-Z]{1,3})([1-9][0-9]{0,5})\Z")
_NUM = re.compile(r"([<>]=?|[≤≥])?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\Z")
_ALLOWED = {
    "DPRA": frozenset(("cysteine_depletion", "lysine_depletion", "mean_depletion")),
    "KeratinoSens": frozenset(("ec1_5", "ec3", "ic50", "max_induction")),
    "h-CLAT": frozenset(("ec150", "ec200", "cv75")),
}
_MISSING = frozenset(("", "na", "n/a", "nd", "not determined", "not tested", "not available", "missing", "-"))


class AssayImportError(ValueError):
    """No raw worksheet data in exceptions: source data may be sensitive."""


class Assay(str, Enum):
    DPRA = "DPRA"
    KERATINOSENS = "KeratinoSens"
    HCLAT = "h-CLAT"


@dataclass(frozen=True)
class MetricColumn:
    metric: str
    header: str
    unit: str

    def __post_init__(self) -> None:
        if not all(type(v) is str and 0 < len(v) <= 96 for v in
                   (self.metric, self.header, self.unit)):
            raise AssayImportError("invalid metric declaration")


@dataclass(frozen=True)
class AssayImportSpec:
    """Caller-reviewed exact sheet/column mapping; no heuristic alias matching."""

    assay: Assay
    sheet_name: str
    header_row: int
    substance_header: str
    metrics: Tuple[MetricColumn, ...]
    reported_call_header: Optional[str] = None
    borderline_call_header: Optional[str] = None
    interference_headers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.assay) is not Assay or type(self.sheet_name) is not str or not self.sheet_name:
            raise AssayImportError("invalid assay or sheet")
        if type(self.header_row) is not int or not 1 <= self.header_row <= MAX_ROWS:
            raise AssayImportError("invalid header row")
        if type(self.substance_header) is not str or not self.substance_header:
            raise AssayImportError("invalid substance column")
        if type(self.metrics) is not tuple or not self.metrics or len(self.metrics) > 8:
            raise AssayImportError("invalid metrics")
        if any(type(m) is not MetricColumn or m.metric not in _ALLOWED[self.assay.value]
               for m in self.metrics):
            raise AssayImportError("unsupported assay metric")
        if len({m.metric for m in self.metrics}) != len(self.metrics):
            raise AssayImportError("duplicate metric")
        optional = (self.reported_call_header, self.borderline_call_header)
        if any(x is not None and (type(x) is not str or not x) for x in optional):
            raise AssayImportError("invalid call column")
        if type(self.interference_headers) is not tuple or len(self.interference_headers) > 8 or any(
                type(x) is not str or not x for x in self.interference_headers):
            raise AssayImportError("invalid interference columns")
        columns = (self.substance_header, *(m.header for m in self.metrics),
                   *(x for x in optional if x is not None), *self.interference_headers)
        if len(columns) != len(set(columns)):
            raise AssayImportError("duplicate mapped header")


@dataclass(frozen=True)
class ReportedMeasurement:
    metric: str
    unit: str
    raw: Optional[str]
    qualifier: str
    magnitude: Optional[str]


@dataclass(frozen=True)
class StagedAssayRow:
    assay: Assay
    substance_raw: str
    measurements: Tuple[ReportedMeasurement, ...]
    reported_call_raw: Optional[str]
    borderline_call_raw: Optional[str]
    interference_raw: Tuple[Tuple[str, Optional[str]], ...]
    qc_status: str
    source_url: str
    source_sha256: str
    sheet_name: str
    excel_row: int


def _measurement(metric: MetricColumn, raw: Optional[str]) -> ReportedMeasurement:
    if raw is None or raw.strip().lower() in _MISSING:
        return ReportedMeasurement(metric.metric, metric.unit, raw, "missing", None)
    match = _NUM.fullmatch(raw.strip())
    if match is None:
        return ReportedMeasurement(metric.metric, metric.unit, raw, "uninterpreted", None)
    op, magnitude = match.groups()
    qualifiers = {None: "exact", ">": "greater_than", ">=": "greater_or_equal",
                  "≥": "greater_or_equal", "<": "less_than", "<=": "less_or_equal",
                  "≤": "less_or_equal"}
    return ReportedMeasurement(metric.metric, metric.unit, raw, qualifiers[op], magnitude)


def _column_index(label: str) -> int:
    result = 0
    for char in label:
        result = result * 26 + ord(char) - ord("A") + 1
    return result


def _text(cell: ET.Element, shared: Tuple[str, ...]) -> str:
    if cell.find(_NS + "f") is not None:
        raise AssayImportError("formula cells not supported")
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        node = cell.find(_NS + "is")
        return "" if node is None else "".join(x.text or "" for x in node.iter(_NS + "t"))
    value = cell.find(_NS + "v")
    raw = "" if value is None or value.text is None else value.text
    if kind == "s":
        try:
            return shared[int(raw)]
        except (ValueError, IndexError):
            raise AssayImportError("invalid shared-string reference") from None
    if kind not in ("n", "str", "b"):
        raise AssayImportError("unsupported cell type")
    return raw


def _safe_zip(blob: bytes) -> ZipFile:
    if type(blob) is not bytes or not 0 < len(blob) <= MAX_XLSX_BYTES:
        raise AssayImportError("workbook exceeds size limits or is empty")
    try:
        archive = ZipFile(BytesIO(blob))
        entries = archive.infolist()
        if len(entries) > MAX_ARCHIVE_MEMBERS or sum(e.file_size for e in entries) > MAX_UNCOMPRESSED_BYTES:
            raise AssayImportError("workbook archive exceeds limits")
        for entry in entries:
            if (entry.file_size > MAX_XLSX_BYTES or entry.filename.startswith("/")
                    or ".." in entry.filename.split("/") or "\\" in entry.filename):
                raise AssayImportError("invalid workbook member")
        if "xl/workbook.xml" not in archive.namelist() or "xl/_rels/workbook.xml.rels" not in archive.namelist():
            raise AssayImportError("not a supported XLSX workbook")
        if any("vbaProject" in x.filename or "externalLinks/" in x.filename for x in entries):
            raise AssayImportError("macros or external links are not supported")
        return archive
    except (BadZipFile, OSError):
        raise AssayImportError("invalid XLSX archive") from None


def _worksheet(archive: ZipFile, name: str) -> bytes:
    try:
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        sheet_ids = {s.get("name"): s.get(_REL + "id") for s in book.iter(_NS + "sheet")}
        if name not in sheet_ids or not sheet_ids[name]:
            raise AssayImportError("requested sheet does not exist")
        targets = {r.get("Id"): r.get("Target") for r in rels.iter(_PKG + "Relationship")}
        target = targets.get(sheet_ids[name])
        if not target:
            raise AssayImportError("worksheet relationship missing")
        path = posixpath.normpath(target.lstrip("/") if target.startswith("/")
                                else posixpath.join("xl", target))
        if not path.startswith("xl/worksheets/") or not path.endswith(".xml"):
            raise AssayImportError("unsafe worksheet relationship")
        return archive.read(path)
    except (KeyError, ET.ParseError, UnicodeError):
        raise AssayImportError("malformed workbook XML") from None


def _shared_strings(archive: ZipFile) -> Tuple[str, ...]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return ()
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        return tuple("".join(t.text or "" for t in si.iter(_NS + "t"))
                     for si in root.iter(_NS + "si"))
    except (ET.ParseError, UnicodeError, KeyError):
        raise AssayImportError("malformed shared strings") from None


def _cells(root: ET.Element, shared: Tuple[str, ...]) -> dict:
    rows = {}
    total = 0
    for row in root.iter(_NS + "row"):
        rownum = row.get("r")
        if not rownum or not rownum.isascii() or not rownum.isdecimal():
            raise AssayImportError("invalid worksheet row")
        number = int(rownum)
        if number < 1 or number > MAX_ROWS or number in rows:
            raise AssayImportError("duplicate or oversized worksheet row")
        cells = {}
        for cell in row.findall(_NS + "c"):
            coord = cell.get("r", "")
            match = _CELL.fullmatch(coord)
            if match is None or int(match.group(2)) != number:
                raise AssayImportError("invalid cell address")
            column = _column_index(match.group(1))
            if column > MAX_COLUMNS or column in cells:
                raise AssayImportError("duplicate or oversized worksheet cell")
            cells[column] = _text(cell, shared)
            total += 1
            if total > MAX_ROWS * MAX_COLUMNS:
                raise AssayImportError("worksheet too large")
        rows[number] = cells
    return rows


def import_assay_xlsx(blob: bytes, spec: AssayImportSpec) -> Tuple[StagedAssayRow, ...]:
    """Stage only caller-mapped values; NEVER feed output to the synthetic governor.

    The official workbook is not fetched. No data conversion, thresholding,
    call arbitration, QC determination, or animal-testing recommendation.
    """
    if type(spec) is not AssayImportSpec:
        raise AssayImportError("invalid import specification")
    spec = AssayImportSpec(spec.assay, spec.sheet_name, spec.header_row,
                           spec.substance_header, tuple(spec.metrics),
                           spec.reported_call_header, spec.borderline_call_header,
                           tuple(spec.interference_headers))
    archive = _safe_zip(blob)
    with archive:
        try:
            root = ET.fromstring(_worksheet(archive, spec.sheet_name))
            rows = _cells(root, _shared_strings(archive))
        except ET.ParseError:
            raise AssayImportError("malformed worksheet XML") from None
    header = rows.get(spec.header_row)
    if header is None:
        raise AssayImportError("header row missing")
    headers = list(header.values())
    if len(headers) != len(set(headers)):
        raise AssayImportError("duplicate source headers")
    lookup = {name: column for column, name in header.items()}
    columns = (spec.substance_header, *(m.header for m in spec.metrics),
               *(x for x in (spec.reported_call_header, spec.borderline_call_header) if x),
               *spec.interference_headers)
    if any(k not in lookup for k in columns):
        raise AssayImportError("declared source columns not present")
    digest = sha256(blob).hexdigest()
    staged = []
    for rownum in sorted(rows):
        if rownum <= spec.header_row:
            continue
        row = rows[rownum]
        if not any(value != "" for value in row.values()):
            continue
        substance = row.get(lookup[spec.substance_header], "")
        if not substance:
            raise AssayImportError("row lacks substance identifier")
        def optional(header: Optional[str]) -> Optional[str]:
            return None if header is None else row.get(lookup[header])
        staged.append(StagedAssayRow(
            spec.assay, substance,
            tuple(_measurement(m, row.get(lookup[m.header])) for m in spec.metrics),
            optional(spec.reported_call_header), optional(spec.borderline_call_header),
            tuple((name, optional(name)) for name in spec.interference_headers),
            "unassessed", SOURCE_URL, digest, spec.sheet_name, rownum))
    if not staged:
        raise AssayImportError("no assay rows")
    return tuple(staged)


__all__ = ["Assay", "AssayImportError", "AssayImportSpec", "MetricColumn",
           "ReportedMeasurement", "StagedAssayRow", "import_assay_xlsx", "SOURCE_URL"]
