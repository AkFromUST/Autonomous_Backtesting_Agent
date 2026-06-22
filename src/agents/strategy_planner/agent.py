from __future__ import annotations

import asyncio
import os
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from src.state import AgentState

# ---------------------------------------------------------------------------
# Prompt for the ReAct agent
# ---------------------------------------------------------------------------

_TASK_TEMPLATE = """\
You are a quant developer tasked with extracting a complete, implementation-ready trading strategy from a research paper.

Your goal: produce a DEVELOPER BRIEF with these five sections, every field populated with exact numbers and formulas from the paper:

## Asset Universe
## Signal Construction (exact formulas and parameters)
## Entry / Exit Rules
## Position Sizing
## Key Parameters

You have one tool: query_paper(question). Use it as many times as you need.

You are done when ALL of the following are true:
- Asset universe is specific (exact instruments, selection criteria)
- Signal formula is written out mathematically with exact parameter values
- Entry and exit conditions have exact numeric thresholds
- Position sizing has an explicit formula or method with numbers
- Every key parameter has its value from the paper

How to use the tool effectively:
- If a query comes back vague or says the paper does not specify something, do NOT accept that — ask a more targeted follow-up. Quote section names, equation numbers, or specific terms from the abstract to narrow the retrieval.
- For formulas, always follow up with: "Quote the exact equation verbatim as it appears in the paper, including all variable definitions."
- For thresholds, ask: "What are the exact numeric values used for [threshold name]?"
- Keep querying until you have concrete numbers. Vague answers mean you need a better question, not that the information is missing.

Use exact numbers from the paper. Mark anything genuinely not stated as "(not specified)".

Paper abstract:
{abstract}
"""


# ---------------------------------------------------------------------------
# LLM factory (preserved — tests call this)
# ---------------------------------------------------------------------------

def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-120b:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("GPT_OSS_API_KEY", ""),
    )


# ---------------------------------------------------------------------------
# paper-qa setup helpers
# ---------------------------------------------------------------------------

def _build_pqa_settings(pqa_model: str):
    """Build a paperqa Settings object with all four LLM fields and a local embedding."""
    from paperqa.settings import Settings, AgentSettings, ParsingSettings

    return Settings(
        llm=pqa_model,
        summary_llm=pqa_model,
        embedding="st-BAAI/bge-m3",
        parsing=ParsingSettings(enrichment_llm=pqa_model, multimodal=False),
        agent=AgentSettings(agent_llm=pqa_model),
        temperature=0.0,
    )


async def _build_docs(pdf_path: str, settings) -> object:
    """Create a paperqa Docs object and add the PDF to it asynchronously."""
    from paperqa import Docs

    docs = Docs()
    await docs.aadd(Path(pdf_path), settings=settings)
    return docs


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

def run(state: AgentState, llm: ChatOpenAI | None = None) -> AgentState:
    paper_id: str = state["paper_id"]
    abstract: str = state["parsed_sections"].get("abstract", "")
    pdf_path = f"/tmp/arxiv_cache/{paper_id.replace('/', '_')}.pdf"

    flags: list[str] = []
    retry_counts: dict[str, int] = dict(state.get("retry_counts", {}))
    retry_counts["strategy_planner"] = retry_counts.get("strategy_planner", 0) + 1

    # -----------------------------------------------------------------------
    # Verify the PDF exists
    # -----------------------------------------------------------------------
    if not Path(pdf_path).exists():
        flags.append(f"[strategy_planner] PDF not found at {pdf_path}")
        raise FileNotFoundError(f"PDF not found at {pdf_path} — cannot run paper-qa")

    # -----------------------------------------------------------------------
    # Build paper-qa Settings & Docs (async — run in current thread's event loop)
    # -----------------------------------------------------------------------
    pqa_model = f"openrouter/{os.getenv('GPT_OSS_MODEL', 'openai/gpt-oss-120b:free')}"

    # litellm reads OPENAI_API_KEY when the provider prefix is openrouter
    os.environ["OPENAI_API_KEY"] = os.getenv("GPT_OSS_API_KEY", "")
    os.environ["OPENROUTER_API_KEY"] = os.getenv("GPT_OSS_API_KEY", "")

    pqa_settings = _build_pqa_settings(pqa_model)

    try:
        docs = asyncio.run(_build_docs(pdf_path, pqa_settings))
    except RuntimeError:
        # Already inside a running event loop (e.g. Jupyter / some test runners)
        import nest_asyncio  # type: ignore[import-not-found]
        nest_asyncio.apply()
        docs = asyncio.get_event_loop().run_until_complete(_build_docs(pdf_path, pqa_settings))

    # -----------------------------------------------------------------------
    # Define the query_paper tool (closure over docs + pqa_settings)
    # -----------------------------------------------------------------------
    @tool
    def query_paper(question: str) -> str:
        """Query the research paper using RAG. Ask questions about the trading strategy described in the paper."""
        async def _query():
            session = await docs.aquery(question, settings=pqa_settings)
            return session.answer

        try:
            return asyncio.run(_query())
        except RuntimeError:
            import nest_asyncio  # type: ignore[import-not-found]
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(_query())

    # -----------------------------------------------------------------------
    # Build and invoke the ReAct agent
    # -----------------------------------------------------------------------
    if llm is None:
        llm = make_llm()

    agent = create_agent(llm, tools=[query_paper])
    task = _TASK_TEMPLATE.format(abstract=abstract)

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            {"recursion_limit": 50},
        )
        messages = result["messages"]
        ai_messages = [m for m in messages if hasattr(m, "type") and m.type == "ai"]
        content: str = ai_messages[-1].content.strip() if ai_messages else ""
    except Exception as exc:
        flags.append(f"[strategy_planner] ReAct agent failed: {exc}")
        content = ""

    flags.append(f"[strategy_planner] developer brief produced — {len(content)} chars")

    # -----------------------------------------------------------------------
    # Second pass: broad extraction of implementation + backtest methodology
    # This is NOT targeted Q&A — broad sweeps so the paper text decides what
    # matters, not the model.
    # -----------------------------------------------------------------------
    implementation_details_path = ""
    try:
        _Q1 = (
            "Extract all implementation details from this paper: exact formulas with variable "
            "definitions, data alignment requirements, which variables are known at time t vs t+1, "
            "forward-look bias considerations, training/test split boundaries, lookback window "
            "specifications, rebalancing frequency, and any implementation warnings or caveats "
            "mentioned by the authors."
        )
        _Q2 = (
            "Extract all backtesting methodology details from this paper: the exact test period, "
            "out-of-sample vs in-sample splits, benchmark used, transaction cost assumptions, how "
            "positions are sized, how the portfolio is rebalanced, and any specific backtest "
            "construction choices the authors made."
        )

        result1 = query_paper.invoke({"question": _Q1})
        result2 = query_paper.invoke({"question": _Q2})

        implementation_details = (
            f"=== Implementation Details ===\n{result1}\n\n"
            f"=== Backtesting Methodology ===\n{result2}"
        )

        safe_id = paper_id.replace("/", "_")
        impl_path = f"/tmp/arxiv_cache/{safe_id}_implementation_details.txt"
        Path(impl_path).parent.mkdir(parents=True, exist_ok=True)
        Path(impl_path).write_text(implementation_details, encoding="utf-8")

        implementation_details_path = impl_path
        flags.append(
            f"[strategy_planner] implementation details written — {len(implementation_details)} chars → {impl_path}"
        )
    except Exception as exc:
        flags.append(f"[strategy_planner] implementation details pass failed: {exc}")

    return {
        **state,
        "strategy_description": content,
        "implementation_details_path": implementation_details_path,
        "flags": flags,
        "retry_counts": retry_counts,
    }
