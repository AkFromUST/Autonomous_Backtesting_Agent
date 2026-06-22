from __future__ import annotations

import os
import pathlib
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.state import AgentState

_SYSTEM = """\
You are the navigator in a pair programming session for financial data acquisition.
Your sole job: write a complete, precise brief for your driver (OpenHands) who will fetch the market data.

The brief MUST contain all of the following — be explicit, not vague:
1. The exact instrument list (ticker symbols, asset class, exchange). Extract verbatim from implementation details.
2. The data directory path where all files must be saved.
3. Required output format:
   - Single file: prices.parquet (long format, one row per date-ticker pair)
   - Columns: date (datetime, NOT index), ticker (str), open, high, low, close, adj_close, volume (all float64)
   - adj_close = close for instruments with no split adjustment
   - volume = 0.0 where unavailable
   - Do NOT use MultiIndex. Always reset_index() before saving.
4. data_manifest.json spec: {"file": "prices.parquet", "tickers": [...], "date_range": [...], "row_count": N, "columns": [...]}
5. Date range: 2010-01-01 to 2023-01-01
6. Any checker feedback from a previous failed attempt (if provided) — driver must fix these issues.

Write the brief as a direct instruction to the driver. Be specific and complete.\
"""


def run(state: AgentState, llm=None) -> dict:
    paper_id = state.get("paper_id", "unknown")
    safe_id = paper_id.replace("/", "_")
    strategy_description = state.get("strategy_description", "")
    impl_path = state.get("implementation_details_path", "")
    data_checker_feedback = state.get("data_checker_feedback", "")

    retry_counts = dict(state.get("retry_counts", {}))
    retry_counts["phase1"] = retry_counts.get("phase1", 0) + 1
    retry_counts["phase1_loop"] = retry_counts.get("phase1_loop", 0) + 1

    impl_details = ""
    if impl_path:
        try:
            impl_details = pathlib.Path(impl_path).read_text(encoding="utf-8")
        except Exception:
            impl_details = "(implementation details file not found)"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"outputs/data/{safe_id}_{timestamp}"
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("REACT_MODEL", "openai/gpt-oss-20b"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("GPT_API_KEY_PAID", ""),
        )

    print(f"\n{'━'*60}")
    print(f"[phase1_guiding] INPUT paper_id           : {paper_id}")
    print(f"[phase1_guiding] INPUT retry_counts        : {retry_counts}")
    print(f"[phase1_guiding] INPUT impl_details        : {len(impl_details)} chars")
    print(f"[phase1_guiding] INPUT data_checker_feedback:\n{data_checker_feedback if data_checker_feedback else '(none)'}")
    print(f"{'━'*60}\n")

    feedback_section = ""
    if data_checker_feedback:
        feedback_section = (
            f"\n\nPREVIOUS ATTEMPT FAILED — CHECKER DIAGNOSIS:\n{data_checker_feedback}\n\n"
            f"Your brief MUST instruct the driver to fix every issue listed above."
        )

    user_msg = (
        f"Strategy:\n{strategy_description}\n\n"
        f"Implementation details (extract exact instruments from here):\n{impl_details}\n\n"
        f"Data directory: {data_dir}"
        f"{feedback_section}\n\n"
        f"Write the complete OpenHands brief now."
    )

    response = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)])
    brief = response.content

    print(f"\n{'━'*60}")
    print(f"[phase1_guiding] OUTPUT brief ({len(brief)} chars):\n{brief}")
    print(f"{'━'*60}\n")

    return {
        **state,
        "openhands_data_brief": brief,
        "data_dir": data_dir,
        "retry_counts": retry_counts,
        "flags": [f"[phase1_guiding] brief written ({len(brief)} chars), data_dir={data_dir}"],
    }
