# ============================================================
# src/tubular_extractor.py
# Extracts casing scheme and tubular data directly from
# PDF tables using pdfplumber — no LLM needed for this step.
# Used for TUBULAR intent queries.
# ============================================================

import re
from pathlib import Path
from src.well_mapping import WELL_MAPPING, ID_TO_FOLDER

# Base path for wells
WELLS_BASE = Path("GeoHackathon_2025/Wells")

# Regex: matches pipe sizes expressed in inches.
# Deliberately excludes decimal form (e.g. 9.625") because degree-minute-second
# coordinates like 21.656" would cause false positives.
# Fraction form (9 5/8", 13-3/8") and whole-inch sizes cover all report variants.
_PIPE_SIZE_RE = re.compile(
    r'\b\d+[-\s]\d+/\d+["\']'                     # fraction: 13-3/8" or 13 3/8"
    r'|\b(?:28|24|20|18|14|13|10|9|7|5|4)["\']'   # whole-inch: 7"  13"  24"  etc.
)

# Named casing strings (used as fallback when no pipe-size pattern matches)
_CASING_NAMED = [
    "conductor", "surface casing", "production casing",
    "liner", "tubing string", "casing string",
    "in conductor", "in casing", "in liner",
]

# Keywords that identify a casing TABLE HEADER row
CASING_HEADER_KEYWORDS = [
    "item", "string", "diameter", "top", "bottom",
    "start md", "end md", "depth", "weight", "grade",
    "connection", "interval", "od", "id", "size", "from"
]

# Encoding fix for garbled fraction characters
ENCODING_FIXES = {
    r'\(cid:\d+\)': '',  # Remove (cid:XX) sequences
    r'½': '1/2',
    r'⅜': '3/8',
    r'⅝': '5/8',
    r'¾': '3/4',
    r'¼': '1/4',
}


def _clean_text(text: str) -> str:
    """Fix encoding issues in extracted text."""
    for pattern, replacement in ENCODING_FIXES.items():
        text = re.sub(pattern, replacement, text)
    return text.strip()


def _is_casing_row(row: list) -> bool:
    """Check if a table row is a casing/tubular data row."""
    row_text = ' '.join(str(c or '') for c in row).lower()
    # Pipe-size pattern takes priority (handles 13-3/8", 10-3/4", 7", etc.)
    if _PIPE_SIZE_RE.search(row_text):
        return True
    return any(k in row_text for k in _CASING_NAMED)


def _is_header_row(row: list) -> bool:
    """Check if a table row is a header row."""
    row_text = ' '.join(str(c or '') for c in row).lower()
    matches = sum(1 for k in CASING_HEADER_KEYWORDS
                  if k.lower() in row_text)
    return matches >= 3


def _extract_section_headers(page) -> list:
    """Extract text lines that look like well section headers,
    e.g. 'HAG-GT-01 casing scheme' or 'Table 2: NLW-GT-03-S1 tubular summary'.
    Only lines where the well ID appears at the start, or after 'Table N:',
    are accepted — this excludes prose sentences like 'The following tables
    summarize the casing scheme for wells HAG-GT-01 and HAG-GT-02'."""
    text = page.extract_text() or ''
    lines = text.split('\n')
    headers = []
    for line in lines:
        line = line.strip()
        # Well ID must start the line, or line must begin with "Table N:"
        if not (re.match(r'^[A-Z]{2,}-GT-\d+', line, re.IGNORECASE)
                or re.match(r'^Table\s+\d+', line, re.IGNORECASE)):
            continue
        if re.search(r'[A-Z]{2,}-GT-\d+', line, re.IGNORECASE) and any(
            k in line.lower()
            for k in ['casing', 'tubular', 'string', 'scheme']
        ):
            headers.append(line)
    return headers


def _remove_empty_columns(headers: list, rows: list):
    """Remove columns where header AND all data cells are empty.
    These are artifacts from merged PDF cells."""
    if not rows:
        return headers, rows

    n_cols = max(
        len(headers) if headers else 0,
        max(len(r) for r in rows) if rows else 0,
    )

    keep_cols = []
    for col_idx in range(n_cols):
        header_val = ''
        if headers and col_idx < len(headers):
            header_val = str(headers[col_idx] or '').strip()

        data_vals = [
            str(row[col_idx] or '').strip()
            for row in rows
            if col_idx < len(row) and str(row[col_idx] or '').strip()
        ]

        if header_val or data_vals:
            keep_cols.append(col_idx)

    # Second pass: remove spacer columns — no header and all data is empty or N/A.
    # These appear in PDFs where each real column is flanked by a blank spacer column.
    filtered_cols = []
    for col_idx in keep_cols:
        header_val = ''
        if headers and col_idx < len(headers):
            header_val = str(headers[col_idx] or '').strip()

        if header_val:
            filtered_cols.append(col_idx)
            continue

        # No header — keep only if at least one data cell has meaningful content
        meaningful = [
            str(row[col_idx] or '').strip()
            for row in rows
            if col_idx < len(row)
            and str(row[col_idx] or '').strip()
            and str(row[col_idx] or '').strip().upper() != 'N/A'
        ]
        if meaningful:
            filtered_cols.append(col_idx)
        # else: spacer column — drop it

    keep_cols = filtered_cols

    new_headers = (
        [headers[i] if i < len(headers) else '' for i in keep_cols]
        if headers else []
    )
    new_rows = [
        [row[i] if i < len(row) else '' for i in keep_cols]
        for row in rows
    ]
    return new_headers, new_rows


def _format_table_as_text(headers: list, rows: list,
                           section_label: str = '') -> str:
    """Format extracted table rows as a markdown table."""
    headers, rows = _remove_empty_columns(headers, rows)

    lines = []

    if section_label:
        lines.append(f'**{section_label}**')
        lines.append('')

    if headers:
        clean_headers = []
        for h in headers:
            cell = _clean_text(str(h or ''))
            cell = cell.replace('\n', ' ').strip()
            clean_headers.append(cell)
        if any(clean_headers):
            lines.append('| ' + ' | '.join(clean_headers) + ' |')
            lines.append('|' + '|'.join(['---'] * len(clean_headers)) + '|')

    for row in rows:
        clean_cells = []
        for c in row:
            cell = _clean_text(str(c or ''))
            cell = cell.replace('\n', ' ').strip()
            clean_cells.append(cell)
        if any(clean_cells):
            lines.append('| ' + ' | '.join(clean_cells) + ' |')

    return '\n'.join(lines)


def _extract_casing_from_pdf(pdf_path: Path) -> list:
    """
    Extract casing table rows from a single PDF.
    Returns list of dicts with keys: pdf, page, headers, rows, section_label.
    """
    import pdfplumber

    header_results: list = []   # tables found via proper casing header
    fallback_results: list = [] # tables found via keyword/size fallback only

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages[:15]):
                tables = page.extract_tables()
                page_headers = _extract_section_headers(page)
                page_entries: list = []

                for table in tables:
                    if not table:
                        continue

                    headers = []
                    casing_rows = []

                    for row in table:
                        if not any(row):
                            continue

                        if _is_header_row(row) and not headers:
                            headers = row
                        elif headers:
                            # Header already found — trust all subsequent rows
                            casing_rows.append(row)
                        elif _is_casing_row(row):
                            # No header yet — fall back to size/name matching
                            casing_rows.append(row)

                    if casing_rows:
                        page_entries.append({
                            "pdf": pdf_path.name,
                            "page": page_idx + 1,
                            "headers": headers,
                            "rows": casing_rows,
                            "section_label": "",
                        })

                # Assign section labels to tables in order.
                # If N headers match N tables exactly → pair them up.
                # If only 1 header for any number of tables → assign to all.
                n_h = len(page_headers)
                n_t = len(page_entries)
                for i, entry in enumerate(page_entries):
                    if n_h == n_t:
                        entry["section_label"] = page_headers[i]
                    elif n_h == 1:
                        entry["section_label"] = page_headers[0]

                for entry in page_entries:
                    if entry["headers"]:
                        header_results.append(entry)
                    else:
                        fallback_results.append(entry)

    except Exception as e:
        print(f"[tubular] Failed to read {pdf_path.name}: {e}")

    # Prefer tables that had a recognised casing header over fallback matches.
    # This prevents ops-table rows (e.g. "24\" Hole") from appearing when a
    # proper casing design table exists elsewhere in the same PDF.
    return header_results if header_results else fallback_results


def _get_well_report_pdfs(well_ids: list) -> list:
    """
    Get all PDF paths from Well report subfolders
    for the given well IDs.
    """
    pdf_paths = []

    for well_id in well_ids:
        folder = ID_TO_FOLDER.get(well_id, "")
        if not folder:
            continue

        well_report_dir = WELLS_BASE / folder / "Well report"
        if well_report_dir.exists():
            pdfs = list(well_report_dir.rglob("*.pdf"))
            pdf_paths.extend(pdfs)

    # Prioritise EOWR files (End of Well Reports)
    eowr_files = [p for p in pdf_paths
                  if any(k in p.name.upper()
                         for k in ['EOWR', 'EOJR', 'SODM',
                                   'FINAL WELL', 'COMPLETION'])]
    other_files = [p for p in pdf_paths if p not in eowr_files]

    return eowr_files + other_files


def extract_tubular_data(well_ids: list) -> str:
    """
    Main function — extract casing scheme for given wells.
    Returns formatted text with complete tubular table.

    Args:
        well_ids: List of well IDs e.g. ['ADK-GT-01']

    Returns:
        Formatted string with casing scheme table
    """
    if not well_ids:
        return "No well specified. Please mention a well name."

    pdf_paths = _get_well_report_pdfs(well_ids)

    if not pdf_paths:
        return (f"No Well report PDFs found for wells: "
                f"{', '.join(well_ids)}")

    print(f"[tubular] Searching {len(pdf_paths)} PDFs "
          f"for wells: {well_ids}")

    all_results = []
    for pdf_path in pdf_paths:
        results = _extract_casing_from_pdf(pdf_path)
        if results:
            all_results.extend(results)
            break  # Stop after first PDF that yields casing data

    if not all_results:
        return (f"No casing table found in Well report PDFs "
                f"for {', '.join(well_ids)}. "
                f"The information may be in scanned documents.")

    # When a single well is queried, filter tables by section label if available.
    if len(well_ids) == 1:
        target = well_ids[0].upper()
        labeled = [r for r in all_results
                   if target in r.get("section_label", "").upper()]
        if labeled:
            all_results = labeled

    # Format output
    output_lines = [
        f"Casing Scheme for {', '.join(well_ids)}:",
        f"Source: {all_results[0]['pdf']} (page {all_results[0]['page']})",
        "",
    ]

    for result in all_results[:2]:  # Max 2 tables
        formatted = _format_table_as_text(
            result["headers"],
            result["rows"],
            section_label=result.get("section_label", ""),
        )
        output_lines.append(formatted)
        output_lines.append("")

    return "\n".join(output_lines)


if __name__ == "__main__":
    print("Testing Well 1 (ADK-GT-01):")
    print(extract_tubular_data(["ADK-GT-01"]))
    print()
    print("Testing Well 8 (MSD-GT-01):")
    print(extract_tubular_data(["MSD-GT-01"]))
