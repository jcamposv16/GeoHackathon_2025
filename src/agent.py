# ============================================================
# src/agent.py
# Sub-challenge 3 — ReAct agent with RAG + Nodal Analysis tools
# ============================================================

from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.vectorstores import Chroma

from src.rag_pipeline import (
    build_qa_chain,
    build_summary_chain,
    summarize_report,
    extract_nodal_parameters,
    clean_answer,
)
from src.nodal_analysis import run_nodal_analysis


# ============================================================
# UPGRADE 1 — LLM-BASED INTENT CLASSIFIER
# ============================================================

_INTENT_PROMPT = (
    "You are a query classifier. Read the query and reply "
    "with exactly ONE word from this list:\n"
    "SUMMARY, PARAMETER, NODAL, PRODUCTION, TUBULAR, GENERAL\n\n"
    "Rules (apply in this order — stop at the first match):\n"
    "- NODAL: ANY of these words appear in the query: "
    "'nodal', 'production capacity', 'operating point', "
    "'flow rate', 'IPR', 'VLP', 'TPC', 'inflow', 'outflow', "
    "'wellbore performance'. "
    "Also matches: 'run analysis', 'calculate', 'compute', "
    "'estimate production', 'perform analysis' when combined "
    "with a well reference. "
    "Examples: 'Run nodal analysis for ADK-GT-01' = NODAL, "
    "'Calculate the operating point for Well 2' = NODAL\n"
    "- PRODUCTION: query asks about production data, "
    "flow rates over time, historical production, "
    "monthly or yearly output, production history, "
    "well performance data, data from spreadsheets, "
    "water cut, watercut, or fluid fraction trends. "
    "Examples: 'Show production history for ADK-GT-01' = PRODUCTION, "
    "'Show me the water cut for MDM-GT-06' = PRODUCTION, "
    "'Water cut analysis for Well 3' = PRODUCTION\n"
    "- TUBULAR: query asks specifically about casing scheme, "
    "tubular program, pipe specifications, casing sizes, "
    "liner sizes, conductor, surface casing, production casing, "
    "tubing dimensions, tubular summary, completion string, "
    "wellbore schematic, casing design. "
    "NOTE: TUBULAR is for a complete table of all casings; "
    "PARAMETER is for one specific value like 'tubing inner diameter'. "
    "Examples: 'What is the casing scheme of ADK-GT-01?' = TUBULAR, "
    "'What tubulars were used in HAG-GT-02?' = TUBULAR, "
    "'Show me the casing program for Well 1' = TUBULAR\n"
    "- PARAMETER: query asks 'what is the [property] of [well]' "
    "OR 'what was the [value] for [well]' OR contains "
    "coordinates, depth, pressure, diameter, temperature, "
    "casing, PI, TVD, MD, location, elevation WITH a well "
    "reference. Key signal: 'the' before property + well name. "
    "Also PARAMETER: questions using 'its', 'the well', 'this well' "
    "that ask for a specific value such as location, depth, pressure, "
    "diameter, or casing — even without an explicit well ID, if a well "
    "was previously mentioned. "
    "NOTE: PARAMETER is for requesting specific measured values FROM "
    "a well report (e.g. 'what is the depth OF Well 4', "
    "'what is the pressure IN the reservoir'). "
    "NOT for asking what a term means. "
    "Example: 'What is the depth of HAG-GT-02?' = PARAMETER. "
    "Example: 'What is its surface location?' = PARAMETER. "
    "Example: 'Who operates Well 8?' = PARAMETER. "
    "Counter-example: 'What is geothermal energy?' = GENERAL. "
    "Counter-example: 'What is productivity index?' = GENERAL\n"
    "- SUMMARY: query contains ANY of: summarise, summary, "
    "overview, report on, tell me about, explain how, "
    "give me an overview, what happened, what occurred, "
    "describe what, how did, what was done, what were the operations. "
    "Example: 'What happened during the clean-out of Well 1?' = SUMMARY\n"
    "- GENERAL: conceptual questions 'What is X?' where X is "
    "a technology, engineering term, or concept with no well reference. "
    "This includes definitions, explanations, and how-does-it-work questions. "
    "Examples: 'What is geothermal energy?' = GENERAL, "
    "'What is seismic inversion?' = GENERAL, "
    "'What is permeability?' = GENERAL, "
    "'What is a geothermal doublet?' = GENERAL, "
    "'What is bottom hole pressure?' = GENERAL, "
    "'What is productivity index?' = GENERAL, "
    "'What is a doublet system?' = GENERAL, "
    "'Define IPR' = GENERAL, "
    "'Explain VLP' = GENERAL, "
    "'How does geothermal heating work?' = GENERAL, "
    "'What does PI mean in well testing?' = GENERAL\n\n"
    "Reply with ONE word only. No punctuation. No explanation.\n\n"
    "Query: {query}"
)

_VALID_INTENTS = {"SUMMARY", "PARAMETER", "NODAL", "PRODUCTION", "TUBULAR", "GENERAL"}

_INTENT_HINTS = {
    "SUMMARY":   "[TASK: Generate a report summary] ",
    "PARAMETER": "[TASK: Extract a specific parameter] ",
    "NODAL":     "[TASK: Run nodal analysis calculation] ",
    "PRODUCTION": "[TASK: Query production data] ",
    "TUBULAR":   "[TASK: Extract tubular data from PDF tables] ",
    "GENERAL":   "",
}


def classify_intent(llm, query: str) -> str:
    """Ask the LLM to classify a query into one of the defined intent categories."""
    try:
        response = llm.invoke(_INTENT_PROMPT.format(query=query))
        # Support both string and AIMessage returns
        text = response.content if hasattr(response, "content") else str(response)
        intent = text.strip().upper()
        return intent if intent in _VALID_INTENTS else "GENERAL"
    except Exception:
        return "GENERAL"


def route_query(llm, query: str) -> str:
    """Prepend a routing hint derived from LLM intent classification."""
    intent = classify_intent(llm, query)
    return _INTENT_HINTS.get(intent, "") + query


def needs_clarification(query: str) -> bool:
    """Return True when the query is too vague to route without a well reference."""
    import re as _re
    ambiguous_patterns = [
        r"^extract data\s*$",
        r"^get data\s*$",
        r"^show me data\s*$",
        r"^what is the \w+\s*\??\s*$",
        r"^tell me about\s*$",
        r"^analyse\s*$",
        r"^analyze\s*$",
    ]
    q = query.strip().lower()
    for pattern in ambiguous_patterns:
        if _re.match(pattern, q):
            return True
    return False


def get_clarification_message(query: str) -> str:
    return (
        "Could you please specify which well you are "
        "asking about? For example: Well 1 (ADK-GT-01), "
        "Well 2 (HAG-GT-01 / HAG-GT-02), Well 3 (MDM-GT-06), "
        "Well 4 (NLW-GT-02-S1), Well 5 (NLW-GT-03), "
        "Well 6 (LIR-GT-01), Well 7 (BRI-GT-01), "
        "or Well 8 (MSD-GT-01)?"
    )


# ============================================================
# UPGRADE 2 — SESSION CONTEXT
# ============================================================

class SessionContext:
    """Tracks per-session well context and conversation history."""

    def __init__(self):
        self.current_well: str | None = None
        self.extracted_params: dict = {}
        self.conversation_history: list = []
        self.pending_intent: str | None = None

    def update_well(self, well_id: str):
        if well_id and well_id != self.current_well:
            self.current_well = well_id
            self.extracted_params = {}  # reset params when well changes

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def get_context_prefix(self) -> str:
        parts = []
        if self.current_well:
            parts.append(f"Current well in discussion: {self.current_well}")
        if self.extracted_params:
            params_str = ", ".join(
                f"{k}: {v}"
                for k, v in self.extracted_params.items()
                if v and v != "Not found in the reports."
            )
            if params_str:
                parts.append(f"Already extracted: {params_str}")
        return "\n".join(parts)


# ============================================================
# TOOLS
# ============================================================

def build_tools(llm, vectorstore: Chroma, k: int = 6):
    """Build and return all agent tools."""
    qa_chain   = build_qa_chain(llm, vectorstore, k)
    summ_chain = build_summary_chain(llm, vectorstore, k=5)

    @tool("well_report_retriever")
    def well_report_retriever(query: str) -> str:
        """
        Use this tool to answer any question about well completion reports,
        casing schemes, equipment specs, or drilling parameters.
        Input should be a standalone question. Returns a plain-text answer.
        """
        result = qa_chain.invoke({"input": query})
        answer = clean_answer(result.get("answer", ""))
        ctx    = result.get("context", [])

        seen, cites = set(), []
        for d in ctx:
            title = d.metadata.get("title", "Untitled")
            if title not in seen:
                seen.add(title)
                cites.append(f"- {title}")

        if cites:
            answer += "\n\nSources:\n" + "\n".join(cites)
        return answer or "No answer found in the reports."

    @tool("report_summariser")
    def report_summariser(request: str) -> str:
        """
        Use this tool when the user asks for a SUMMARY of a well completion
        report. Input should include the well name and focus area.
        Returns a concise plain-text paragraph.
        """
        return summarize_report(summ_chain, request, max_words=180)

    @tool("nodal_analysis_calculator")
    def nodal_analysis_calculator(well_id: str) -> str:
        """
        Use this tool when the user asks to perform nodal analysis,
        estimate production capacity, or calculate the operating point
        for a specific well. Input should be the well ID (e.g. HAG-GT-02).
        Automatically extracts parameters from reports and runs calculations.
        """
        import re
        params = extract_nodal_parameters(qa_chain, well_id)

        def parse_float(val: str, default: float) -> float:
            if not val or not isinstance(val, str):
                return default
            nums = re.findall(r"\d+\.?\d*", str(val))
            if not nums:
                return default
            try:
                return float(nums[0])
            except (ValueError, IndexError):
                return default

        result = run_nodal_analysis(
            well_name=well_id,
            reservoir_pressure=parse_float(params.get("reservoir_pressure", ""), 250.0),
            wellhead_pressure=parse_float(params.get("wellhead_pressure", ""), 10.0),
            pi=parse_float(params.get("productivity_index", ""), 5.0),
        )

        response = result["summary"] + "\n\nExtracted Parameters:\n"
        for k, v in params.items():
            response += f"  {k}: {v}\n"
        return response

    return [well_report_retriever, report_summariser, nodal_analysis_calculator]


# ============================================================
# UPGRADE 3 — UPDATED build_agent
# ============================================================

def build_agent(llm, vectorstore: Chroma, k: int = 6, verbose: bool = True):
    """Initialise the ReAct master agent with all tools."""
    tools = build_tools(llm, vectorstore, k)

    system_message = (
        "You are a geothermal well engineering assistant for SPE GeoHackathon 2025.\n\n"
        "When the input begins with [TASK: ...], treat that as a strong routing hint:\n"
        "- [TASK: Generate a report summary]    → use report_summariser\n"
        "- [TASK: Extract a specific parameter] → use well_report_retriever\n"
        "- [TASK: Run nodal analysis calculation] → use nodal_analysis_calculator\n"
        "- [TASK: Query production data]        → use well_report_retriever\n\n"
        "If a 'Current well in discussion' prefix is provided, apply it to any "
        "tool call that needs a well ID when none is explicitly mentioned.\n\n"
        "Always be concise and include source titles for RAG answers."
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_message},
    )
    print("Agent ready with 3 tools.")
    return agent
