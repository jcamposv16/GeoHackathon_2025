# ============================================================
# src/rag_pipeline.py
# RAG chains for summarisation (Sub-challenge 1)
# and parameter extraction (Sub-challenge 2)
# ============================================================

import re
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from src.well_mapping import normalize_query, get_well_ids_from_query


# ============================================================
# RETRIEVAL HELPERS
# ============================================================

_TABLE_PATTERN = re.compile(
    r"\|"
    r"|(?:\d[\d,.]*\s*(?:m|bar|psi|rpm|°C|°F|kg|mm|in)\b)"
    r"|(?:[NS]\s*\d+°)"
    r"|(?:[EW]\s*\d+°)"
    r"|(?:MD|TVD)\s*[:=]?\s*\d+",
    re.IGNORECASE,
)


def _build_chroma_filter(well_ids: list) -> None:
    # Chroma only supports $eq/$in/$nin on metadata fields; well_ids is a
    # comma-separated string so none of those operators work reliably.
    # Filtering is handled post-retrieval by RRF + metadata boosting instead.
    return None


def _reciprocal_rank_fusion(lists: list, rrf_k: int = 60) -> list:
    scores: dict = {}
    docs: dict = {}
    for ranked in lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content[:120]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            docs[key] = doc
    return [docs[k] for k in sorted(scores, key=scores.__getitem__, reverse=True)]


_SUMMARY_TABLE_KEYWORDS = [
    'well name', 'well type', 'well number',
    'operator', 'spud date', 'total depth',
    'target formation', 'location municipality',
    'days operational', 'well summary',
]


def _boost_table_chunks(docs: list, boost: float = 2.0) -> list:
    boosted = []
    for d in docs:
        content = d.page_content
        score = boost if _TABLE_PATTERN.search(content) else 1.0

        # Extra boost for chunks that contain well summary table data
        content_lower = content.lower()
        summary_hits = sum(
            1 for k in _SUMMARY_TABLE_KEYWORDS if k in content_lower
        )
        if summary_hits >= 2:
            score += 3.0

        boosted.append((score, d))

    boosted.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in boosted]


_BM25_CACHE: dict = {}


def _build_bm25(vectorstore: Chroma, k: int) -> BM25Retriever:
    cache_key = id(vectorstore)
    if cache_key in _BM25_CACHE:
        print("BM25 loaded from cache")
        return _BM25_CACHE[cache_key]

    raw = vectorstore.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    retriever = BM25Retriever.from_documents(chunks, k=k)
    _BM25_CACHE[cache_key] = retriever
    print(f"BM25 index built with {len(chunks)} chunks")
    return retriever


def _filter_by_well_ids(docs: list, well_ids: list) -> list:
    """Keep only docs whose well_ids metadata contains at least one target ID."""
    targets = {wid.lower() for wid in well_ids}
    return [
        d for d in docs
        if targets & {w.strip().lower()
                      for w in d.metadata.get("well_ids", "").split(",")
                      if w.strip()}
    ]


def _filter_by_doc_type(docs: list, doc_types: list) -> list:
    """Keep only docs whose doc_type metadata matches one of the allowed types.
    Returns all docs unchanged if doc_types is empty."""
    if not doc_types:
        return docs
    targets = {dt.lower() for dt in doc_types}
    filtered = [
        d for d in docs
        if d.metadata.get("doc_type", "").lower() in targets
    ]
    return filtered if filtered else docs


def apply_metadata_boost(docs: list, query: str) -> list:
    """
    Re-rank docs by well ID match.
    When the query targets specific wells, documents from those wells
    get +20; documents from other wells get -5 (pushed to the back).
    When no well is specified, only the +20 bonus applies.
    """
    q = query.lower()
    query_well_ids = {wid.lower() for wid in get_well_ids_from_query(query)}
    ranked = []
    for doc in docs:
        score = 1.0
        wells = doc.metadata.get("well_ids", "")
        if isinstance(wells, str):
            doc_well_ids = {w.strip().lower() for w in wells.split(",") if w.strip()}
            if query_well_ids:
                if doc_well_ids & query_well_ids:
                    score += 20.0
                else:
                    score -= 5.0
            elif any(w in q for w in doc_well_ids):
                score += 20.0
        if doc.metadata.get("doc_type") == "Well report":
            score += 10.0
        ranked.append((score, doc))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked]


def _make_retriever(vectorstore: Chroma, bm25: BM25Retriever,
                    k: int, doc_types: list = None):
    """
    Retriever applied to every chain:
      1. Normalize       – expand folder names to well IDs
      2. Hybrid search   – semantic (Chroma) + keyword (BM25)
      3. Metadata filter – post-retrieval filter by detected well IDs
                           (falls back to unfiltered if nothing passes)
      4. RRF merge       – Reciprocal Rank Fusion
      5. Table boost     – surface chunks with measurements / tables
      6. Slice           – return top-k
    """
    def retrieve(q: str) -> list:
        norm_q = normalize_query(q)
        well_ids = get_well_ids_from_query(norm_q)

        # Fetch more candidates so the post-filter still yields k results
        fetch_k = k * 6
        vec_docs = vectorstore.as_retriever(search_kwargs={"k": fetch_k}).invoke(norm_q)
        bm25.k = fetch_k
        bm25_docs = bm25.invoke(norm_q)

        # Post-retrieval metadata filter
        if well_ids:
            vec_filtered  = _filter_by_well_ids(vec_docs,  well_ids) or vec_docs
            bm25_filtered = _filter_by_well_ids(bm25_docs, well_ids) or bm25_docs
        else:
            vec_filtered, bm25_filtered = vec_docs, bm25_docs

        # Doc-type filter — restrict to specific subfolders
        allowed_types = doc_types or []
        if allowed_types:
            vec_filtered  = _filter_by_doc_type(vec_filtered,  allowed_types)
            bm25_filtered = _filter_by_doc_type(bm25_filtered, allowed_types)
            print(f"[retrieve] doc_types={allowed_types} "
                  f"vec after type filter: {len(vec_filtered)}")

        print(f"[retrieve] norm_q={norm_q!r}")
        print(f"[retrieve] well_ids={well_ids}")
        print(f"[retrieve] vec filtered {len(vec_filtered)}/{len(vec_docs)}, "
              f"bm25 filtered {len(bm25_filtered)}/{len(bm25_docs)}")

        merged  = _reciprocal_rank_fusion([vec_filtered, bm25_filtered])
        boosted = apply_metadata_boost(merged, norm_q)
        return _boost_table_chunks(boosted)[:k]

    return (
        RunnableLambda(lambda x: x["input"])
        | RunnableLambda(retrieve)
    )


# ============================================================
# ANSWER CLEANER
# ============================================================
def clean_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?is)end of job report.*", "", text)
    text = re.sub(r"##+.*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"Page \d+.*", "", text)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text.strip()

    first = re.sub(r"^\s*(Answer:|Document:)\s*", "", paragraphs[0], flags=re.IGNORECASE)
    return first.strip()


# ============================================================
# COORDINATE PROTECTION
# ============================================================

# LLM-generated DMS:  51° 57' 55.600"
_DMS_RE = re.compile(r'(\d{1,3})°\s*(\d{1,2})[\'′]\s*([\d.]+)[\"″]')

# Original decimal-minutes in source:  51° 57.869  or  51°57.869'
_DM_CTX_RE = re.compile(r'(\d{1,3})°\s*(\d{1,2}\.\d+)[\'′]?')

# RD easting / northing
_RD_X_RE = re.compile(r'X\s*[=:]\s*([\d,.]+)', re.IGNORECASE)
_RD_Y_RE = re.compile(r'Y\s*[=:]\s*([\d,.]+)', re.IGNORECASE)


def clean_coordinates(answer: str, context_docs: list,
                      query: str = "") -> str:
    """
    Post-process LLM answer to fix coordinate errors.
    Step 0: Remove coordinates from wrong wells written by LLM.
    Step 1: Fix DMS format using correct-well context only.
    Step 2: Append missing RD coordinates from correct well.
    """
    if not context_docs:
        return answer

    # --- Identify which well this answer is about ---
    all_well_ids = [
        "ADK-GT-01-S1", "ADK-GT-01",
        "HAG-GT-01", "HAG-GT-02",
        "MDM-GT-06-S2", "MDM-GT-06-S1", "MDM-GT-06",
        "NLW-GT-02-S1",
        "NLW-GT-03-S1", "NLW-GT-03",
        "LIR-GT-01", "BRI-GT-01", "MSD-GT-01",
    ]

    # Skip coordinate processing for general queries
    # that don't reference a specific well
    general_keywords = [
        'what is', 'define', 'explain', 'how does',
        'what are', 'describe', 'tell me about'
    ]
    query_lower = query.lower()
    has_well = any(w.upper() in (answer + query).upper()
                   for w in all_well_ids)
    is_general = (
        any(q in query_lower for q in general_keywords)
        and not has_well
    )
    if is_general:
        return answer

    # Check answer AND query for well IDs (longer first)
    search_text = (answer + " " + query).upper()
    answer_wells = [w for w in all_well_ids if w.upper() in search_text]

    # Filter context to correct-well chunks only
    if answer_wells:
        correct_docs = [
            d for d in context_docs
            if any(
                w.upper() in d.metadata.get("well_ids", "").upper()
                for w in answer_wells
            )
        ]
        if not correct_docs:
            correct_docs = context_docs
    else:
        correct_docs = context_docs

    correct_context = "\n".join(d.page_content for d in correct_docs)

    # --- Step 0: Remove wrong-well RD coordinates ---
    x_in_answer = _RD_X_RE.search(answer)
    if x_in_answer and answer_wells:
        x_str = x_in_answer.group(1).strip()
        x_clean = x_str.replace(",", "").replace(".", "")[:6]
        if x_clean not in correct_context.replace(",", "").replace(".", ""):
            print(f"[coords] removing wrong RD X: {x_str}")
            answer = re.sub(
                r'-\s*RD[^:\n]*:?\s*X:\s*[\d.,]+\s*/?\s*Y:\s*[\d.,]+[^\n]*\n?',
                '', answer, flags=re.IGNORECASE
            )
            answer = re.sub(
                r'X:\s*[\d.,]+\s*/\s*Y:\s*[\d.,]+',
                '', answer, flags=re.IGNORECASE
            )
            answer = answer.strip()

    # Also remove wrong-well address lines
    wrong_addresses = {
        "ADK-GT-01": ["Nieuwe Dijk", "Andijk"],
        "BRI-GT-01": ["Nieuwe Dijk", "Andijk"],
        "MSD-GT-01": ["Nieuwe Dijk", "Andijk"],
    }
    if answer_wells:
        for well in answer_wells:
            for addr in wrong_addresses.get(well, []):
                if addr in answer:
                    answer = re.sub(
                        rf'-\s*Address:[^\n]*{addr}[^\n]*\n?',
                        '', answer, flags=re.IGNORECASE
                    )
                    print(f"[coords] removed wrong address containing '{addr}'")

    # --- Step 1: DMS → decimal-minutes (correct-well context only) ---
    dms_hits = list(_DMS_RE.finditer(answer))
    if dms_hits:
        ctx_dm: dict[int, str] = {}
        for m in _DM_CTX_RE.finditer(correct_context):
            deg = int(m.group(1))
            if deg not in ctx_dm:
                ctx_dm[deg] = m.group(0)
        for hit in reversed(dms_hits):
            deg = int(hit.group(1))
            if deg in ctx_dm:
                print(f"[coords] DMS fix: '{hit.group(0)}' → '{ctx_dm[deg]}'")
                answer = answer[:hit.start()] + ctx_dm[deg] + answer[hit.end():]

    # --- Step 2: Append RD if still missing ---
    if not _RD_X_RE.search(answer):
        for doc in correct_docs:
            x_m = _RD_X_RE.search(doc.page_content)
            y_m = _RD_Y_RE.search(doc.page_content)
            if x_m and y_m:
                x_val = x_m.group(1).strip()
                y_val = y_m.group(1).strip()
                print(f"[coords] appending RD X: {x_val}, Y: {y_val}")
                answer = answer.rstrip() + f"\n- RD: X: {x_val}, Y: {y_val}"
                break

    return answer.strip()


# ============================================================
# SUB-CHALLENGE 1 — SUMMARISATION CHAIN
# ============================================================
def build_summary_chain(llm, vectorstore: Chroma, k: int = 5):
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.runnables import ConfigurableField
    from typing import List, Any

    class WellFilterRetriever(BaseRetriever):
        base_retriever: Any
        well_ids: List[str] = []

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> List[Document]:

            # Extract clean retrieval query by stripping
            # registry facts prefix injected for LLM context.
            # The embedding search and BM25 should use only
            # the actual user question, not the facts block.
            retrieval_query = query
            skip_prefixes = [
                'VERIFIED WELL DATA',
                'IMPORTANT FACTS',
                'Well ID:',
                'Name:',
                'Field:',
                'Municipality:',
                'Operator:',
                'Total Depth',
                'Spud Date:',
                'Status:',
                'Type: Sidetrack',
            ]
            if any(p in query for p in skip_prefixes):
                lines = query.strip().split(chr(10))
                clean = [
                    l.strip() for l in lines
                    if l.strip() and not any(
                        p in l for p in skip_prefixes)
                ]
                if clean:
                    retrieval_query = clean[-1]

            if not self.well_ids:
                try:
                    return self.base_retriever.invoke(query)
                except Exception:
                    return []

            merged = []
            seen = set()

            def add_docs(docs):
                for d in docs:
                    key = d.page_content[:80]
                    if key not in seen:
                        if d.metadata.get(
                                'doc_type') == 'Well report':
                            if any(
                                wid.upper() in d.metadata
                                .get('well_ids', '').upper()
                                for wid in self.well_ids
                            ):
                                seen.add(key)
                                merged.append(d)

            # Pass 1: BM25-dominant search using well ID
            # BM25 finds exact keyword matches reliably
            # Fixes MDM-GT-06/NLW-GT-03/MSD-GT-01 where
            # embedding search retrieves wrong wells
            try:
                well_str = ' '.join(self.well_ids)
                bm25_docs = self.base_retriever.invoke(well_str)
                add_docs(bm25_docs)
            except Exception:
                pass

            # Pass 2: semantic search with cleaned query
            # Adds narrative context chunks
            try:
                sem_docs = self.base_retriever.invoke(retrieval_query)
                add_docs(sem_docs)
            except Exception:
                pass

            # Pass 3: table-targeted query
            # Surfaces well summary tables specifically
            try:
                table_query = (
                    ' '.join(self.well_ids) +
                    ' well name well type operator '
                    'location municipality total depth '
                    'target formation well summary'
                )
                tbl_docs = self.base_retriever.invoke(table_query)
                add_docs(tbl_docs)
            except Exception:
                pass

            # Fallback if nothing found
            if not merged:
                try:
                    return self.base_retriever.invoke(retrieval_query)
                except Exception:
                    return []

            return merged[:15]

    bm25 = _build_bm25(vectorstore, k)
    base_ret = _make_retriever(vectorstore, bm25, min(k * 3, 20),
                               doc_types=["Well report"])
    retriever = WellFilterRetriever(
        base_retriever=base_ret,
        well_ids=[],
    ).configurable_fields(
        well_ids=ConfigurableField(id="well_ids_filter")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "{registry_context}"
         "You are a technical assistant for geothermal well completion reports.\n\n"
         "When asked to summarise a well, write a clear paragraph of 4-6 sentences "
         "using ONLY information found in the context provided. "
         "Include as many of these details as you can find:\n"
         "- Well name and whether it is a sidetrack (and of which well)\n"
         "- Well type (producer or injector)\n"
         "- Location (field, municipality, country)\n"
         "- Operator\n"
         "- Target formation(s)\n"
         "- Total depth (MD and TVD)\n"
         "- Main operations performed\n"
         "- Key results or outcomes\n\n"
         "Rules:\n"
         "- Only use information explicitly stated in the context\n"
         "- Do NOT say a well is a sidetrack unless the context explicitly states this\n"
         "- Do NOT invent relationships between wells\n"
         "- If you cannot find certain details, skip them\n"
         "- Write in plain text, past tense\n"
         "- Do NOT list coordinates unless asked"),
        ("human",
         "Context:\n{context}\n\n"
         "Summarise this request:\n{input}\n\n"
         "Write 4-6 sentences covering the well details. "
         "If the context contains TABLE: markers, extract "
         "values from them — they contain structured well "
         "data including well name, type, location, operator, "
         "target formation, and total depth. "
         "Use all available information from the context.")
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)


def summarize_report(chain, prompt: str, max_words: int = 180) -> str:
    task = (
        f"{prompt}\n\n"
        f"Write a clear, technically accurate summary within {max_words} words. "
        "Plain text only. If information is missing, state it is not provided."
    )
    result = chain.invoke({"input": task})
    answer = clean_answer(result.get("answer", ""))
    return clean_coordinates(answer, result.get("context", []), query=prompt)


# ============================================================
# SUB-CHALLENGE 2 — PARAMETER EXTRACTION CHAIN
# ============================================================
def build_qa_chain(llm, vectorstore: Chroma, k: int = 6):
    bm25 = _build_bm25(vectorstore, k)
    retriever = _make_retriever(vectorstore, bm25, k,
                                doc_types=["Well report", "Well test"])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise geothermal well data extractor.\n"
         "Answer with the exact value from the document.\n"
         "Format: state the parameter name, then the exact value and unit as written in the source.\n"
         "If multiple coordinate systems exist, list all of them.\n"
         "Copy numbers exactly — never convert, round, or reformat.\n"
         "If not found: respond exactly 'Not found in the reports.'\n"
         "Maximum 2-3 lines. No extra explanation.\n\n"
         "COORDINATE RULE: Only report coordinates that belong to the specific "
         "well being asked about. Never report coordinates from a different well. "
         "If the context contains coordinates labeled as belonging to a different "
         "well ID, ignore them completely. When you see Map Easting / Map Northing "
         "values in the context that match the queried well, report them as RD "
         "coordinates X and Y respectively."),
        ("human",
         "Context:\n{context}\n\n"
         "Question: {input}\n\n"
         "Report ALL values found for this parameter exactly as written in the source. "
         "If multiple coordinate systems exist in the same table row, report all of them.")
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)


def extract_parameter(chain, question: str) -> str:
    result = chain.invoke({"input": question})
    answer = clean_answer(result.get("answer", ""))
    return clean_coordinates(answer, result.get("context", []), query=question)


def _make_production_retriever(vectorstore: Chroma,
                                bm25: BM25Retriever, k: int):
    return _make_retriever(vectorstore, bm25, k,
                           doc_types=["Production data"])


# ============================================================
# NODAL ANALYSIS PARAMETER EXTRACTOR
# ============================================================
NODAL_PARAMETERS = [
    ("well_name",          "What is the name or ID of the well?"),
    ("tubing_id_m",        "What is the tubing inner diameter in metres or inches?"),
    ("measured_depth_m",   "What is the total measured depth (MD) of the well in metres?"),
    ("tvd_m",              "What is the true vertical depth (TVD) of the well in metres?"),
    ("reservoir_pressure", "What is the reservoir pressure in bar or psi?"),
    ("wellhead_pressure",  "What is the wellhead pressure in bar or psi?"),
    ("productivity_index", "What is the productivity index (PI) of the well?"),
    ("casing_scheme",      "What is the casing scheme or casing program of the well?"),
]


def extract_nodal_parameters(qa_chain, well_id: str) -> dict:
    results = {}
    print(f"\nExtracting parameters for well: {well_id}")
    for param_name, question in NODAL_PARAMETERS:
        full_question = f"For well {well_id}: {question}"
        answer = extract_parameter(qa_chain, full_question)
        results[param_name] = answer
        print(f"  {param_name}: {answer}")
    return results
