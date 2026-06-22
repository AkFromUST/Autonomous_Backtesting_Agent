from __future__ import annotations

import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schemas — outputs of each agent
# ---------------------------------------------------------------------------

class LintError(BaseModel):
    line: int | None
    code: str
    message: str


class LogicReport(BaseModel):
    verdict: str = Field(description="pass | fail")
    issues: list[str] = Field(description="List of specific logic mismatches found, with line references")
    notes: str = ""


# ---------------------------------------------------------------------------
# LangGraph shared state — flows through every node
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Input
    paper_url: str
    paper_id: str

    # Ingestor output
    parsed_sections: dict                    # {abstract, methodology, data, results, raw_text}

    # Strategy Planner output
    strategy_description: str               # code-ready developer brief
    implementation_details_path: str        # path to implementation details file

    # Phase1Agent output
    data_manifest: dict                # {ticker: {path, available_from, rows, columns, status}}
    data_dir: str                      # timestamped data directory path, passed to Phase2Agent
    data_coverage_note: str            # one-sentence summary of what was fetched vs. what strategy implied
    data_feasibility: str              # "full" | "partial" | "insufficient"

    # Checker agent outputs
    data_checker_verdict: str          # "PASS" | "FAIL" | "ERROR"
    data_checker_feedback: str         # full structured diagnosis from data_checker
    code_checker_verdict: str          # "PASS" | "FAIL" | "ERROR"
    code_checker_feedback: str         # full structured diagnosis from code_checker

    # Phase2Agent (backtest) output
    generated_code: str
    lint_errors: list[LintError]
    execution_result: str                    # subprocess stdout/stderr from code execution

    # Reporter output
    final_report: str

    # VerB.32: guiding-node briefs for the separate OpenHands nodes
    openhands_data_brief: str               # written by phase1_guiding, consumed by openhands_data node
    openhands_code_brief: str               # written by phase2_guiding, consumed by openhands_code node

    # Orchestration
    skip_phase1: bool                        # True → dispatch jumps directly to phase2_guiding
    retry_counts: dict[str, int]             # {agent_name: attempt_count}
    flags: Annotated[list[str], operator.add]  # accumulates warnings; add reducer safe for parallel appends
    status: str                              # RUNNING | DONE | HUMAN_REVIEW_NEEDED
    current_agent: str                       # last dispatched agent name
