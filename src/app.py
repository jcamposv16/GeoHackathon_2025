# ============================================================
# src/app.py
# Gradio UI — SPE GeoHackathon 2025
# Unified chat interface with automatic intent routing
# ============================================================

import os
import re
import sys
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Must run before any src.* import: ingest.py reads PDF_CACHE_DIR
# at module import time, so .env has to be loaded first or that
# read silently falls back to the default path.
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")
import gradio as gr

from src.llm_loader import load_llm
from src.ingest import load_pdf_docs, build_vectorstore
from src.rag_pipeline import (
    summarize_report,
    build_summary_chain,
    build_qa_chain,
    clean_answer,
    clean_coordinates,
)
from src.agent import (
    build_agent, classify_intent, SessionContext, route_query,
    needs_clarification, get_clarification_message,
)
from src.well_mapping import get_well_ids_from_query
from src.tubular_extractor import extract_tubular_data
from src.nodal_analysis import run_nodal_analysis, generate_nodal_plot
from src.well_data import format_well_facts
from src.production_analysis import (
    generate_production_chart,
    generate_watercut_chart,
    generate_cumulative_chart,
    generate_gas_trend_chart,
    generate_monthly_comparison_chart,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR      = Path(__file__).parent.parent
PDF_DIR       = BASE_DIR / "GeoHackathon_2025" / "Wells"
DB_DIR        = Path(os.getenv("VECTOR_DB_DIR", BASE_DIR / "vector_db"))

LLM_NAME      = os.getenv("LLM_NAME", "llama3.2")
EMBED_MODEL   = os.getenv("MODEL_EMBED", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K         = int(os.getenv("TOP_K", 6))

# ============================================================
# INITIALISE PIPELINE (runs once at startup)
# ============================================================
print("\nInitialising GeoHackathon RAG Pipeline...\n")

llm           = load_llm(model_name=LLM_NAME)
pdf_docs      = load_pdf_docs(PDF_DIR)
# Set to True only when PDFs change or chunking logic changes.
# The vectorstore is saved to disk and reloads automatically.
vectorstore   = build_vectorstore(
    docs=pdf_docs,
    db_dir=DB_DIR,
    embed_model=EMBED_MODEL,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    recreate=False,
)

summary_chain = build_summary_chain(llm, vectorstore)
qa_chain      = build_qa_chain(llm, vectorstore, k=TOP_K)
agent         = build_agent(llm, vectorstore, k=TOP_K, verbose=False)

print("\nAll systems ready. Launching Gradio UI...\n")


# ============================================================
# SESSION STORE
# ============================================================
session_store: dict = {}


def get_or_create_session(session_id: str) -> SessionContext:
    if session_id not in session_store:
        session_store[session_id] = SessionContext()
    return session_store[session_id]


# ============================================================
# SOURCE COLLECTOR
# ============================================================
def _collect_sources(result: dict) -> list:
    sources = []
    seen: set = set()
    for doc in result.get("context", []):
        title = doc.metadata.get("title", "Unknown")
        page  = doc.metadata.get("page", None)

        if page is None:
            page_match = re.search(r"\[PAGE\s+(\d+)\]", doc.page_content)
            if page_match:
                page = page_match.group(1)

        key = f"{title}_p{page}"
        if key not in seen:
            seen.add(key)
            ref = f"• {title}"
            if page:
                ref += f" (p.{page})"
            sources.append(ref)
    return sources


# ============================================================
# WELL GROUND-TRUTH NOTE
# ============================================================
def _build_well_fact(well_ids: list, message: str) -> str:
    """Build factual notes about well relationships to prevent LLM hallucination."""
    from src.well_mapping import WELL_MAPPING, ID_TO_FOLDER

    facts = []

    # Detect parent-sidetrack relationships from the queried IDs themselves
    parents = [w for w in well_ids
               if not any(w.endswith(s) for s in ['-S1', '-S2', '-S3'])]
    sidetracks = [w for w in well_ids
                  if any(w.endswith(s) for s in ['-S1', '-S2', '-S3'])]
    for st in sidetracks:
        if parents:
            facts.append(f"{st} is a sidetrack of {parents[0]}.")

    # Special case: NLW-GT-02-S1 parent is NLW-GT-02 (not in the same folder)
    for wid in well_ids:
        if wid == 'NLW-GT-02-S1':
            facts.append("NLW-GT-02-S1 is a sidetrack of NLW-GT-02.")

    return ' '.join(facts)


# ============================================================
# CHART FILE HELPER
# ============================================================
def _save_chart_to_file(img_b64: str) -> str:
    """Save base64 PNG to a temp file and
    return the file path for gr.Image."""
    if not img_b64:
        return None
    try:
        import base64
        buf = base64.b64decode(img_b64)
        tmp = tempfile.NamedTemporaryFile(
            suffix='.png', delete=False)
        tmp.write(buf)
        tmp.close()
        return tmp.name
    except Exception:
        return None


# ============================================================
# UNIFIED CHAT HANDLER
# ============================================================
def handle_chat(message: str, history: list, request: gr.Request = None):
    if not message.strip():
        return "", history

    start = time.time()
    chart_path = None

    # Session context
    session_id = str(request.session_hash) if request else "default"
    ctx = get_or_create_session(session_id)

    # Detect well and update context
    well_ids = get_well_ids_from_query(message)
    if well_ids:
        ctx.update_well(well_ids[0])

    # Check if query needs clarification
    if needs_clarification(message) and not well_ids and not ctx.current_well:
        response = get_clarification_message(message)
        response += f"\n\nResponse time: {round(time.time() - start, 1)}s"
        ctx.add_message("user", message)
        ctx.add_message("assistant", response)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return "", history

    # Build enriched query with context
    context_prefix = ctx.get_context_prefix()
    intent = classify_intent(llm, message)
    print(f"[chat] intent={intent} well_ids={well_ids}")

    # For SUMMARY queries, replace "Well N" with all canonical IDs for that
    # folder so the retriever sees the actual IDs (e.g. HAG-GT-01 and HAG-GT-02)
    # rather than a generic label that may not appear in the document text.
    if well_ids and intent == "SUMMARY":
        normalized_message = message
        folder_match = re.search(r'Well\s+\d+', message, re.IGNORECASE)
        if folder_match:
            # Folder query — expand to first 2 IDs
            ids_str = ' and '.join(well_ids[:2])
            normalized_message = re.sub(
                r'Well\s+\d+', ids_str, message,
                flags=re.IGNORECASE
            )
        # else: direct ID query — keep as is
    else:
        normalized_message = message

    # Get verified well data from official registry
    registry_facts = format_well_facts(well_ids) if well_ids else ''

    # Route based on intent
    try:
        if intent == "SUMMARY":
            well_fact = _build_well_fact(well_ids, message)
            # Registry facts go into system context (registry_context).
            # {input} stays as the clean question so the retriever is
            # not polluted by facts-block text.
            registry_context = ''
            if well_fact:
                registry_context += 'KNOWN FACTS: ' + well_fact + '\n\n'
            if registry_facts:
                registry_context += registry_facts + '\n\n'

            result = summary_chain.invoke(
                {
                    "input": normalized_message,
                    "registry_context": registry_context,
                },
                config={
                    "configurable": {
                        "well_ids_filter": well_ids
                    }
                }
            )
            raw     = clean_answer(result.get("answer", ""))
            answer  = clean_coordinates(raw, result.get("context", []), query=message)
            sources = _collect_sources(result)

        elif intent == "TUBULAR":
            tubular_well_ids = well_ids or ([ctx.current_well] if ctx.current_well else [])
            if tubular_well_ids:
                answer  = extract_tubular_data(tubular_well_ids)
                sources = []
            else:
                answer  = get_clarification_message(message)
                sources = []

        elif intent == "PRODUCTION":
            well_ids_prod = get_well_ids_from_query(message)
            if not well_ids_prod and ctx.current_well:
                well_ids_prod = [ctx.current_well]
            if well_ids_prod:
                # Detect chart type from message
                msg_lower = message.lower()
                wc_keywords = ['water cut', 'watercut',
                               'wc', 'fraction']
                cum_keywords = ['cumulative', 'total production',
                                'cum prod', 'cumulative production']
                gas_keywords = ['gas trend', 'gas rate',
                                'gas production trend']
                comp_keywords = ['monthly comparison',
                                 'seasonal', 'by year',
                                 'year comparison',
                                 'monthly pattern']

                if any(k in msg_lower for k in wc_keywords):
                    prod_summary, prod_img = generate_watercut_chart(
                        well_ids_prod)
                elif any(k in msg_lower for k in cum_keywords):
                    prod_summary, prod_img = generate_cumulative_chart(
                        well_ids_prod)
                elif any(k in msg_lower for k in gas_keywords):
                    prod_summary, prod_img = generate_gas_trend_chart(
                        well_ids_prod)
                elif any(k in msg_lower for k in comp_keywords):
                    prod_summary, prod_img = (
                        generate_monthly_comparison_chart(
                            well_ids_prod))
                else:
                    prod_summary, prod_img = generate_production_chart(
                        well_ids_prod)
                answer  = prod_summary
                sources = []
                if prod_img:
                    chart_path = _save_chart_to_file(prod_img)
                    if chart_path:
                        answer += '\n\n'
            else:
                answer  = get_clarification_message(message)
                sources = []

        elif intent == "NODAL":
            try:
                nodal_result = run_nodal_analysis(
                    well_name=', '.join(well_ids) if well_ids
                              else 'Unknown')
                nodal_img = generate_nodal_plot(nodal_result)
                answer = nodal_result['summary']
                if nodal_img:
                    chart_path = _save_chart_to_file(nodal_img)
                    if chart_path:
                        answer += '\n\n'
            except Exception as e:
                answer = f'Nodal analysis error: {e}'
            sources = []

        elif intent == "PARAMETER":
            result = qa_chain.invoke({
                "input": f"{registry_facts}\n\n{message}"
                if registry_facts else message
            })
            raw    = clean_answer(result.get("answer", ""))
            answer = clean_coordinates(
                raw, result.get("context", []), query=message)
            sources = _collect_sources(result)

            # Fallback to registry if not found
            NOT_FOUND = [
                'not found', 'not mentioned',
                'not specified', 'not available',
                'no information', 'cannot find',
                'not in the report', 'not provided',
            ]
            if any(p in answer.lower() for p in NOT_FOUND):
                if registry_facts and well_ids:
                    answer = (
                        "Based on the official NLOG well "
                        "registry:\n\n" +
                        registry_facts.replace(
                            "VERIFIED WELL DATA FROM "
                            "OFFICIAL REGISTRY:\n", ""
                        )
                    )
                    sources = []

        else:  # GENERAL
            if not well_ids:
                # Conceptual question — use LLM directly
                from langchain_core.messages import (
                    SystemMessage, HumanMessage)
                sys_msg = SystemMessage(content=(
                    "You are a geothermal and petroleum "
                    "engineering expert. Answer the "
                    "question clearly in 3-5 sentences "
                    "using your knowledge. Give a direct "
                    "technical definition or explanation. "
                    "Do NOT say 'not found in reports'. "
                    "Do NOT mention specific well names "
                    "unless asked."
                ))
                hum_msg = HumanMessage(content=message)
                resp = llm([sys_msg, hum_msg])
                answer = resp.content.strip()
                sources = []
            else:
                # Has well context — use RAG
                result = qa_chain.invoke({
                    "input": f"{registry_facts}\n\n{message}"
                    if registry_facts else message
                })
                raw    = clean_answer(result.get("answer", ""))
                answer = clean_coordinates(
                    raw, result.get("context", []),
                    query=message)
                sources = _collect_sources(result)

    except Exception as e:
        answer  = f"Error: {e}"
        sources = []

    # Suppress coordinates and sources when answer was not found
    NOT_FOUND_PHRASES = [
        'not found in the reports',
        'not found in the provided',
        'not mentioned in',
        'not specified in',
        'no information found',
        'cannot find',
    ]
    if any(p in answer.lower() for p in NOT_FOUND_PHRASES):
        sources = []
        # Strip any RD coordinate lines appended
        answer = re.sub(
            r'\n?-?\s*RD:\s*X:[\s\S]*?(?=\n|$)',
            '', answer, flags=re.IGNORECASE
        ).strip()

    elapsed = round(time.time() - start, 1)

    # Format response
    response = answer
    if sources:
        response += "\n\nSources:\n" + "\n".join(sources[:4])
    response += f"\n\nResponse time: {elapsed}s"

    # Update session
    ctx.add_message("user", message)
    ctx.add_message("assistant", response)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]

    # Append chart as separate image message if chart was generated
    if chart_path:
        history = history + [
            {"role": "assistant",
             "content": {"path": chart_path,
                         "mime_type": "image/png"}}
        ]

    return "", history


# ============================================================
# GRADIO UI
# ============================================================
with gr.Blocks(title="SPE GeoHackathon 2025 — Well Report AI") as demo:

    gr.Markdown("""
    # Well Report AI Assistant
    Ask anything about the well completion reports. Summaries, parameters, and nodal analysis are handled automatically.
    """)

    chatbot = gr.Chatbot(
        height=600,
        label="Conversation",
        render_markdown=True,
        sanitize_html=False,
        layout="bubble",
        avatar_images=(None, None),
        allow_file_downloads=True,
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Example: Summarise HAG-GT-02 / What is the MD for ADK-GT-01? / Run nodal analysis for Well 2.",
            label="Your message",
            scale=8,
            lines=2,
        )
        send_btn  = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    gr.Examples(
        examples=[
            ["Summarise the completion report for HAG-GT-02."],
            ["What is the casing scheme of ADK-GT-01?"],
            ["What is the reservoir pressure for Well 4?"],
            ["Run nodal analysis for HAG-GT-02."],
            ["What happened during the clean-out of Well 1?"],
        ],
        inputs=msg_input,
    )

    send_btn.click(handle_chat,
        [msg_input, chatbot],
        [msg_input, chatbot])
    msg_input.submit(handle_chat,
        [msg_input, chatbot],
        [msg_input, chatbot])
    clear_btn.click(lambda: [], outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
