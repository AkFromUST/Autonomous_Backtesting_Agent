from __future__ import annotations

import os
import pathlib

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.state import AgentState

_SYSTEM = """\
You are the navigator in a pair programming session for backtest implementation.
Your sole job: write a complete, precise brief for your driver (OpenHands) who will implement the vectorbt backtest.

The brief MUST contain all of the following:
1. The exact strategy algorithm — signal formula, lookback windows, position sizing, rebalancing frequency.
   Extract verbatim from the implementation details. Do not paraphrase.
2. The canonical data loading pattern (always include this exactly):
   import pandas as pd
   import json
   with open("{data_dir}/data_manifest.json") as f:
       manifest = json.load(f)
   df = pd.read_parquet("{data_dir}/prices.parquet")
   # columns: date, ticker, open, high, low, close, adj_close, volume
   close = df.pivot(index="date", columns="ticker", values="adj_close")
3. The output script path where the backtest must be written.
4. The required metric output lines (driver must print these, real floats not NaN):
   Sharpe Ratio: <value>
   Total Return: <value>
   Annualized Return: <value>
5. Any checker feedback from a previous failed attempt — driver must fix these issues.
6. A ## DIAGNOSTIC CHECKPOINTS section at the end of the brief.
   Based on the algorithm in the implementation details, identify 3–5 key computation stages
   where a one-line print would confirm that stage completed correctly.
   For each checkpoint, write only the stage name and what property to verify.
   Do NOT name specific Python variables — the driver chooses those from their own implementation.
   Always include a final checkpoint for the metrics output.

   Format this section exactly as:
   ## DIAGNOSTIC CHECKPOINTS
   1. After <stage>: confirm <property>
   2. After <stage>: confirm <property>
   ...
   Final: print Sharpe Ratio, Total Return, Annualized Return as real floats.

Rules for your brief:
- No data downloads inside the script — all data is already on disk.
- Forward-look bias is a critical failure — signals must only use data up to t-1.
- Do NOT prescribe libraries beyond the data loading pattern.\
"""


def run(state: AgentState, llm=None) -> dict:
    paper_id = state.get("paper_id", "unknown")
    safe_id = paper_id.replace("/", "_")
    strategy_description = state.get("strategy_description", "")
    impl_path = state.get("implementation_details_path", "")
    data_dir = state.get("data_dir", "")
    data_manifest = state.get("data_manifest", {})
    code_checker_feedback = state.get("code_checker_feedback", "")
    execution_result = state.get("execution_result", "")

    retry_counts = dict(state.get("retry_counts", {}))
    retry_counts["phase2"] = retry_counts.get("phase2", 0) + 1
    retry_counts["phase2_loop"] = retry_counts.get("phase2_loop", 0) + 1

    script_path = f"outputs/backtests/{safe_id}.py"
    pathlib.Path(script_path).parent.mkdir(parents=True, exist_ok=True)
    if retry_counts.get("phase2_loop", 0) <= 1:
        pathlib.Path(script_path).unlink(missing_ok=True)

    impl_details = ""
    if impl_path:
        try:
            impl_details = pathlib.Path(impl_path).read_text(encoding="utf-8")
        except Exception:
            impl_details = "(implementation details file not found)"

    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("REACT_MODEL", "openai/gpt-oss-120b:free"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("GPT_API_KEY_PAID", ""),
        )

    print(f"\n{'━'*60}")
    print(f"[phase2_guiding_v2] INPUT paper_id            : {paper_id}")
    print(f"[phase2_guiding_v2] INPUT retry_counts         : {retry_counts}")
    print(f"[phase2_guiding_v2] INPUT impl_details         : {len(impl_details)} chars")
    print(f"[phase2_guiding_v2] INPUT execution_result     :\n{execution_result[-2000:] if execution_result else '(none)'}")
    print(f"[phase2_guiding_v2] INPUT code_checker_feedback:\n{code_checker_feedback if code_checker_feedback else '(none)'}")
    print(f"{'━'*60}\n")

    feedback_section = ""
    if code_checker_feedback:
        feedback_section = (
            f"\n\nPREVIOUS ATTEMPT FAILED — CHECKER DIAGNOSIS:\n{code_checker_feedback}\n\n"
            f"Your brief MUST instruct the driver to fix every issue listed above."
        )

    tickers = data_manifest.get("tickers", [])
    date_range = data_manifest.get("date_range", [])

    user_msg = (
        f"Strategy:\n{strategy_description}\n\n"
        f"Implementation details (exact formulas and parameters):\n{impl_details}\n\n"
        f"Data directory: {data_dir}\n"
        f"Available tickers: {tickers}\n"
        f"Date range: {date_range}\n"
        f"Output script path: {script_path}"
        f"{feedback_section}\n\n"
        f"Write the complete OpenHands brief now."
    )

    response = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)])
    brief = response.content

    print(f"\n{'━'*60}")
    print(f"[phase2_guiding_v2] OUTPUT brief ({len(brief)} chars):\n{brief}")
    print(f"{'━'*60}\n")

    return {
        **state,
        "openhands_code_brief": brief,
        "retry_counts": retry_counts,
        "flags": [f"[phase2_guiding_v2] brief written ({len(brief)} chars), script={script_path}"],
    }
