# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.
"""Synthetic-only checks: these do not validate toxicology or NIH sheet layout."""

import io
import zipfile
from xml.sax.saxutils import escape

import pytest

from ailee.domains.opennam.assay_imports import (
    Assay, AssayImportError, AssayImportSpec, MetricColumn, import_assay_xlsx,
)


def xlsx(sheet, headers, values, *, formula=False, extra_files=None):
    cells = []
    for row, items in enumerate((headers, *values), 1):
        parts = []
        for col, value in enumerate(items):
            letter = chr(65 + col)
            if value is None:
                continue
            if formula and row == 2 and col == 2:
                parts.append(f'<c r="{letter}{row}"><f>2+2</f><v>4</v></c>')
            else:
                parts.append(f'<c r="{letter}{row}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        cells.append(f'<row r="{row}">{"".join(parts)}</row>')
    xml = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
           '<sheetData>' + ''.join(cells) + '</sheetData></worksheet>')
    book = ('<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="{escape(sheet)}" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('xl/workbook.xml', book)
        archive.writestr('xl/_rels/workbook.xml.rels', rels)
        archive.writestr('xl/worksheets/sheet1.xml', xml)
        for key, val in (extra_files or {}).items():
            archive.writestr(key, val)
    return output.getvalue()


@pytest.mark.parametrize(('assay','metric','unit','value'), [
    (Assay.DPRA, 'cysteine_depletion', '%', '>2000'),
    (Assay.KERATINOSENS, 'ec1_5', 'uM', '<=0.5'),
    (Assay.HCLAT, 'ec150', 'ug/mL', '≥2.2e3'),
])
def test_three_assays_preserve_bounds_calls_and_source(assay, metric, unit, value):
    workbook = xlsx('Data', ['Compound', 'Measured', 'Reported', 'Borderline', 'Interference'],
                    [['SYN-CHEM-1', value, 'POS', 'Borderline', 'interference possible']])
    spec = AssayImportSpec(assay, 'Data', 1, 'Compound',
                           (MetricColumn(metric, 'Measured', unit),),
                           'Reported', 'Borderline', ('Interference',))
    rows = import_assay_xlsx(workbook, spec)
    assert len(rows) == 1
    row = rows[0]
    assert row.assay == assay and row.substance_raw == 'SYN-CHEM-1'
    assert row.measurements[0].raw == value and row.measurements[0].unit == unit
    assert row.measurements[0].qualifier != 'exact'
    assert row.reported_call_raw == 'POS'
    assert row.borderline_call_raw == 'Borderline'
    assert row.interference_raw == (('Interference', 'interference possible'),)
    assert row.qc_status == 'unassessed'
    assert row.source_sha256 and row.sheet_name == 'Data' and row.excel_row == 2
    assert not hasattr(row, 'observation')


def test_missing_values_and_non_numeric_are_not_inferred():
    workbook = xlsx('Data', ['Compound', 'Measured', 'Reported'],
                    [['SYN-CHEM-1', 'N/A', 'Negative'], ['SYN-CHEM-2', 'co-elution', 'Positive']])
    spec = AssayImportSpec(Assay.DPRA, 'Data', 1, 'Compound',
                           (MetricColumn('mean_depletion', 'Measured', '%'),), 'Reported')
    a, b = import_assay_xlsx(workbook, spec)
    assert (a.measurements[0].qualifier, a.measurements[0].magnitude) == ('missing', None)
    assert (b.measurements[0].qualifier, b.measurements[0].magnitude) == ('uninterpreted', None)


def test_exact_decimal_stays_string_and_no_borderline_is_invented():
    workbook = xlsx('Data', ['Substance', 'Measure'], [['SYN-CHEM-1', '1.20']])
    spec = AssayImportSpec(Assay.HCLAT, 'Data', 1, 'Substance',
                           (MetricColumn('cv75', 'Measure', 'µg/mL'),))
    row, = import_assay_xlsx(workbook, spec)
    assert row.measurements[0].magnitude == '1.20'
    assert row.measurements[0].qualifier == 'exact'
    assert row.reported_call_raw is None and row.borderline_call_raw is None


@pytest.mark.parametrize('tamper', ['invalid_sheet', 'missing_header', 'duplicate_header', 'formula', 'extra_rows'])
def test_reject_incorrect_or_unreviewed_input(tamper):
    headers = ['Substance', 'Measure', 'Reported']
    values = [['SYN-CHEM-1', '1.0', 'POS']]
    if tamper == 'duplicate_header':
        headers = ['Substance', 'Measure', 'Measure']
    if tamper == 'extra_rows':
        values.append(['', '3.0', 'NEG'])
    workbook = xlsx('Data', headers, values, formula=tamper == 'formula')
    spec = AssayImportSpec(Assay.DPRA,
                           'Missing' if tamper == 'invalid_sheet' else 'Data', 1, 'Substance',
                           (MetricColumn('mean_depletion', 'Absent' if tamper == 'missing_header' else 'Measure', '%'),),
                           'Reported')
    with pytest.raises(AssayImportError):
        import_assay_xlsx(workbook, spec)


def test_invalid_archive_and_external_links_rejected():
    spec = AssayImportSpec(Assay.DPRA, 'Data', 1, 'Substance',
                           (MetricColumn('mean_depletion', 'Measure', '%'),))
    with pytest.raises(AssayImportError):
        import_assay_xlsx(b'junk', spec)
    workbook = xlsx('Data', ['Substance', 'Measure'], [['SYN-CHEM-1', '3']],
                    extra_files={'xl/externalLinks/externalLink1.xml':'inert'})
    with pytest.raises(AssayImportError):
        import_assay_xlsx(workbook, spec)


def test_closed_metric_allowlists():
    with pytest.raises(AssayImportError):
        AssayImportSpec(Assay.DPRA, 'Data', 1, 'Substance',
                        (MetricColumn('ec150', 'Measure', '%'),))
