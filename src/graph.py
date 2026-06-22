from __future__ import annotations

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from src.patched_llm import PatchedChatOpenAI
from src.state import AgentState
from src.agents.ingestor.agent import run as ingestor_run
from src.agents.strategy_planner.agent import run as strategy_planner_run
from src.agents.phase1_guiding.agent import run as phase1_guiding_run
from src.agents.openhands_data.agent import run as openhands_data_run
from src.agents.normalizer.agent import run as normalizer_run
from src.agents.data_checker.agent import run as data_checker_run
from src.agents.phase2_guiding_v2.agent import run as phase2_guiding_run
from src.agents.openhands_code_v2.agent import run as openhands_code_run
from src.agents.code_checker.agent import run as code_checker_run
from src.agents.reporter.agent import run as reporter_run

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MAX_LOOP = 5

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def make_react() -> PatchedChatOpenAI:
    return PatchedChatOpenAI(
        model=os.getenv("REACT_MODEL", "openai/gpt-oss-120b:free"),
        base_url=OPENROUTER_BASE,
        api_key=os.getenv("GPT_API_KEY_PAID"),
    )


# ---------------------------------------------------------------------------
# Node wrappers
# ---------------------------------------------------------------------------

def ingestor_node(state: AgentState) -> AgentState:
    return ingestor_run(state)

def strategy_planner_node(state: AgentState) -> AgentState:
    return strategy_planner_run(state, llm=make_react())

def phase1_guiding_node(state: AgentState) -> AgentState:
    return phase1_guiding_run(state, llm=make_react())

def openhands_data_node(state: AgentState) -> AgentState:
    return openhands_data_run(state)

def normalizer_node(state: AgentState) -> AgentState:
    return normalizer_run(state)

def data_checker_node(state: AgentState) -> AgentState:
    return data_checker_run(state)

def phase2_guiding_node(state: AgentState) -> AgentState:
    return phase2_guiding_run(state, llm=make_react())

def openhands_code_node(state: AgentState) -> AgentState:
    return openhands_code_run(state)

def code_checker_node(state: AgentState) -> AgentState:
    return code_checker_run(state)

def reporter_node(state: AgentState) -> AgentState:
    return reporter_run(state)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_strategy_planner(state: AgentState) -> str:
    strategy = state.get("strategy_description", "")
    if strategy.strip().startswith("NOT_TRADEABLE:"):
        return "reporter"
    return "phase1_guiding"


def route_after_data_checker(state: AgentState) -> str:
    verdict = state.get("data_checker_verdict", "ERROR")
    if verdict == "PASS":
        return "phase2_guiding"
    retries = state.get("retry_counts", {}).get("phase1_loop", 0)
    if retries >= MAX_LOOP:
        return "escalate"
    return "phase1_guiding"


def route_after_code_checker(state: AgentState) -> str:
    verdict = state.get("code_checker_verdict", "ERROR")
    if verdict == "PASS":
        return "reporter"
    retries = state.get("retry_counts", {}).get("phase2_loop", 0)
    if retries >= MAX_LOOP:
        return "escalate"
    return "phase2_guiding"


def dispatch_node(state: AgentState) -> AgentState:
    return state

def escalate_node(state: AgentState) -> AgentState:
    return {"status": "HUMAN_REVIEW_NEEDED", "flags": []}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("dispatch", dispatch_node)
    g.add_node("ingestor", ingestor_node)
    g.add_node("strategy_planner", strategy_planner_node)
    g.add_node("phase1_guiding", phase1_guiding_node)
    g.add_node("openhands_data", openhands_data_node)
    g.add_node("normalizer", normalizer_node)
    g.add_node("data_checker", data_checker_node)
    g.add_node("phase2_guiding", phase2_guiding_node)
    g.add_node("openhands_code", openhands_code_node)
    g.add_node("code_checker", code_checker_node)
    g.add_node("reporter", reporter_node)
    g.add_node("escalate", escalate_node)

    g.set_entry_point("dispatch")
    g.add_conditional_edges(
        "dispatch",
        lambda s: "phase2_guiding" if s.get("skip_phase1") else "ingestor",
        {"ingestor": "ingestor", "phase2_guiding": "phase2_guiding"},
    )
    g.add_edge("ingestor", "strategy_planner")
    g.add_conditional_edges(
        "strategy_planner",
        route_after_strategy_planner,
        {"phase1_guiding": "phase1_guiding", "reporter": "reporter"},
    )
    g.add_edge("phase1_guiding", "openhands_data")
    g.add_edge("openhands_data", "normalizer")
    g.add_edge("normalizer", "data_checker")
    g.add_conditional_edges(
        "data_checker",
        route_after_data_checker,
        {"phase2_guiding": "phase2_guiding", "phase1_guiding": "phase1_guiding", "escalate": "escalate"},
    )
    g.add_edge("phase2_guiding", "openhands_code")
    g.add_edge("openhands_code", "code_checker")
    g.add_conditional_edges(
        "code_checker",
        route_after_code_checker,
        {"reporter": "reporter", "phase2_guiding": "phase2_guiding", "escalate": "escalate"},
    )
    g.add_edge("reporter", END)
    g.add_edge("escalate", END)

    return g.compile()


graph = build_graph()
