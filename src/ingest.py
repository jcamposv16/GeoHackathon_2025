# ============================================================
# src/ingest.py
# Loads PDFs, enriches metadata, builds Chroma vectorstore
# ============================================================

import os
import re
import shutil
import pickle
import hashlib
import concurrent.futures
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.well_mapping import WELL_MAPPING

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = Path(os.getenv("PDF_CACHE_DIR", BASE_DIR / "GeoHackathon_pdf_cache"))


# ============================================================
# PDF EXTRACTION
# ============================================================

def _format_table(table: list) -> str:
    """Convert a pdfplumber table (list of rows) to pipe-delimited text."""
    lines = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        lines.append("TABLE: " + " | ".join(cells))
    return "\n".join(lines)


def pdf_to_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF preserving table structure.
    For each page:
      1. Use pdfplumber to extract tables as structured rows/columns.
      2. Use pymupdf page.get_text() for all remaining prose.
      3. Emit table rows first, then prose, so table data is not mixed
         with surrounding text.
    Falls back to pymupdf-only if pdfplumber is unavailable or fails.
    """
    import pymupdf
    try:
        import pdfplumber
        _has_plumber = True
    except ImportError:
        _has_plumber = False

    page_texts: list[str] = []

    with pymupdf.open(str(pdf_path)) as mupdf_doc:
        plumber_doc = None
        if _has_plumber:
            try:
                plumber_doc = pdfplumber.open(str(pdf_path))
            except Exception:
                plumber_doc = None

        try:
            for page_idx, mu_page in enumerate(mupdf_doc):
                parts: list[str] = []

                # Step 1 — extract tables via pdfplumber
                if plumber_doc is not None:
                    try:
                        pl_page = plumber_doc.pages[page_idx]
                        tables = pl_page.extract_tables()
                        for table in tables:
                            if table:
                                parts.append(_format_table(table))
                    except Exception:
                        pass  # fall through to prose-only for this page

                # Step 2 — extract prose via pymupdf
                parts.append(mu_page.get_text())

                page_texts.append("\n".join(parts))
        finally:
            if plumber_doc is not None:
                plumber_doc.close()

    return "\n".join(page_texts)


# ============================================================
# TABLE DETECTION AND MARKUP
# ============================================================

def _is_table_line(line: str) -> bool:
    """True if a line looks like it belongs to a table."""
    if line.count("|") >= 2:
        return True
    # Lines with at least 3 runs of 2+ whitespace (column-aligned data)
    if len(re.findall(r"\s{2,}", line.strip())) >= 3 and line.strip():
        return True
    return False


def markup_tables(text: str) -> str:
    """
    Detect table blocks and wrap them with <<<TABLE_START>>> / <<<TABLE_END>>>.
    A table block is a run of consecutive lines where the majority look like
    table lines.  Isolated single lines are not wrapped.
    """
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    i = 0
    while i < len(lines):
        # Scan ahead: collect a candidate block of table-ish lines
        j = i
        while j < len(lines) and _is_table_line(lines[j]):
            j += 1

        block_len = j - i
        if block_len >= 2:
            result.append("<<<TABLE_START>>>\n")
            result.extend(lines[i:j])
            result.append("<<<TABLE_END>>>\n")
            i = j
        else:
            result.append(lines[i])
            i += 1

    return "".join(result)


# ============================================================
# TABLE-AWARE CHUNKING
# ============================================================

def table_aware_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into chunks respecting table boundaries.
    Text outside tables is split normally with RecursiveCharacterTextSplitter.
    Text inside <<<TABLE_START>>> / <<<TABLE_END>>> is always kept as one chunk.
    """
    table_re = re.compile(
        r"(<<<TABLE_START>>>\n.*?<<<TABLE_END>>>\n)", re.DOTALL
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks: list[str] = []
    last_end = 0

    for m in table_re.finditer(text):
        # Split the prose before this table
        prose = text[last_end : m.start()]
        if prose.strip():
            chunks.extend(splitter.split_text(prose))

        # Keep the full table as one atomic chunk
        table_chunk = m.group(1).strip()
        if table_chunk:
            chunks.append(table_chunk)

        last_end = m.end()

    # Trailing prose after the last table
    tail = text[last_end:]
    if tail.strip():
        chunks.extend(splitter.split_text(tail))

    return chunks


# ============================================================
# WELL ID EXTRACTION FROM TEXT
# ============================================================

def extract_well_ids_from_text(text: str) -> list[str]:
    """
    Detect well identifiers like ADK-GT-01, HAG-GT-02.
    Returns a sorted, deduplicated list of canonical well IDs.
    """
    wells: set[str] = set()
    norm = text.replace(" -", "-").replace("- ", "-").replace("GT ", "GT-")

    for m in re.finditer(r"\b([A-Z]{2,4}-GT-\d+(?:-S?\d+)?)\b", norm):
        wells.add(m.group(1))

    # Range pattern: HAG-GT-01-02 → HAG-GT-01 + HAG-GT-02
    for m in re.finditer(r"\b([A-Z]{2,4})-GT-(\d+)-(\d+)\b", norm):
        prefix, start, end = m.groups()
        for n in range(int(start), int(end) + 1):
            wells.add(f"{prefix}-GT-{n:02d}")

    return sorted(wells)


# ============================================================
# METADATA HELPERS
# ============================================================

def _well_folder_from_path(path: Path) -> str:
    """
    Walk up the path looking for a part that matches a WELL_MAPPING key
    (e.g. "Well 1", "Well 2").  Returns "" if not found.
    """
    for part in path.parts:
        if part in WELL_MAPPING:
            return part
    return ""


def _doc_type_from_path(path: Path) -> str:
    """
    Detect document type from subfolder name.
    Returns one of: 'Well report', 'Production data',
    'PVT', 'Technical log', 'Well test', 'Unknown'
    """
    doc_types = [
        "Well report", "Production data",
        "PVT", "Technical log", "Well test",
    ]
    for part in path.parts:
        for doc_type in doc_types:
            if doc_type.lower() in part.lower():
                return doc_type
    return "Unknown"


def _well_ids_for_folder(folder: str, text_ids: list[str]) -> str:
    """
    Return a comma-separated list of well IDs for a document.

    Uses ONLY the WELL_MAPPING canonical IDs for the detected folder.
    Text-extracted IDs are intentionally excluded: reports routinely
    reference neighbouring wells ("previously drilled NLW-GT-02-S1..."),
    and adding those IDs cross-contaminates queries for other wells.

    Falls back to text extraction only when no folder mapping exists
    (e.g. PDFs placed outside the standard Well N folder structure).
    """
    mapping_ids: list[str] = WELL_MAPPING.get(folder, [])
    if mapping_ids:
        return ",".join(mapping_ids)
    # No folder mapping — fall back to text extraction
    return ",".join(dict.fromkeys(text_ids).keys())


# ============================================================
# LOAD ALL PDFs (parallel)
# ============================================================

def _content_hash(pdf_path: Path) -> str:
    """MD5 hash of the PDF's raw bytes, used as a content-addressed cache key."""
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()


def _get_cache_path(pdf_path: Path, content_hash: str) -> Path:
    """Generate a unique cache file path based on PDF filename and content hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{pdf_path.stem}_{content_hash}"
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{hash_key}.pkl"


def load_pdf_docs(pdf_dir: Path) -> list[Document]:
    """
    Recursively load all PDFs, detect table blocks, enrich metadata,
    and return one Document per PDF.
    """
    pdf_paths = list(pdf_dir.rglob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
        return []

    # Hash each PDF's content once per run; reused for the cache-hit count
    # below and for the actual cache lookup/write inside convert().
    cache_paths = {p: _get_cache_path(p, _content_hash(p)) for p in pdf_paths}

    cache_hits = sum(1 for p in pdf_paths if cache_paths[p].exists())
    print(f"Found {len(pdf_paths)} PDFs — "
          f"{cache_hits} cached, "
          f"{len(pdf_paths) - cache_hits} to convert...")

    def convert(path: Path) -> Document | None:
        # Check cache first
        cache_path = cache_paths[path]
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass  # cache corrupt — reconvert

        # Not cached — convert fresh
        try:
            raw = pdf_to_text(path)
            text = markup_tables(raw)

            well_folder = _well_folder_from_path(path)
            text_ids = extract_well_ids_from_text(raw + " " + path.stem)
            well_ids_str = _well_ids_for_folder(well_folder, text_ids)

            meta = {
                "title": path.stem,
                "file_path": str(path),
                "well_folder": well_folder,
                "well_ids": well_ids_str,
                "doc_type": _doc_type_from_path(path),
            }
            doc = Document(page_content=text, metadata=meta)

            # Save to cache
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(doc, f)
            except Exception:
                pass  # cache write failed — not critical

            return doc
        except Exception as e:
            print(f"Failed to convert {path.name}: {e}")
            return None

    docs: list[Document] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for doc in ex.map(convert, pdf_paths):
            if doc:
                docs.append(doc)

    print(f"Loaded {len(docs)} documents from {len(pdf_paths)} PDFs.")
    return docs


# ============================================================
# BUILD / RELOAD CHROMA VECTORSTORE
# ============================================================

def build_vectorstore(
    docs: list[Document],
    db_dir: Path,
    embed_model: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    recreate: bool = True,
) -> Chroma:
    """
    Split documents into chunks using table-aware splitting,
    embed them, and persist to Chroma.
    If recreate=False and db_dir exists, reload the existing store.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    # Treat an empty or corrupted db_dir as a forced rebuild
    if db_dir.exists():
        file_count = sum(1 for _ in db_dir.rglob("*") if _.is_file())
        if file_count < 3:
            print(f"db_dir has only {file_count} file(s) — treating as corrupted, rebuilding.")
            recreate = True

    if not recreate and db_dir.exists():
        print(f"Reloading existing vectorstore from {db_dir}")
        return Chroma(
            persist_directory=str(db_dir),
            collection_name="well_reports",
            embedding_function=embeddings,
        )

    # Table-aware split: each doc's text is split preserving table blocks
    chunks: list[Document] = []
    for doc in docs:
        for chunk_text in table_aware_split(doc.page_content, chunk_size, chunk_overlap):
            chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))

    print(f"Split {len(docs)} docs into {len(chunks)} chunks.")

    # Sanitise metadata (Chroma only accepts string/int/float/bool primitives)
    for c in chunks:
        for k, v in list(c.metadata.items()):
            if not isinstance(v, (str, int, float, bool)) or v is None:
                c.metadata[k] = str(v)

    if db_dir.exists():
        shutil.rmtree(db_dir)

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="well_reports",
            persist_directory=str(db_dir),
        )
        print(f"Vectorstore saved to {db_dir}")
    except Exception as e:
        shutil.rmtree(db_dir, ignore_errors=True)
        raise RuntimeError(
            "Vectorstore build failed — directory deleted. "
            "Please run again to rebuild from scratch."
        ) from e

    return vectorstore
