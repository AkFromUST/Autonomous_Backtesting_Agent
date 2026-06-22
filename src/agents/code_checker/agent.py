from __future__ import annotations

import json
import os
import pathlib
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.patched_llm import PatchedChatOpenAI
from src.state import AgentState

PROJECT_ROOT = pathlib.Path(".").resolve()

_SYSTEM = """\
You are a quantitative finance code auditor for an autonomous backtesting pipeline.
You receive a backtest script, its execution output, and the paper's implementation details.
Your job: decide whether the backtest correctly implements the paper's strategy.
Respond with a single JSON object — no markdown fences, no commentary outside the JSON.\
"""

_PROMPT = """\
Audit this backtest for paper: {paper_id}

IMPLEMENTATION DETAILS (your ground truth — exact formulas from the paper):
{impl_details}

BACKTEST SCRIPT ({script_path}):
{script_text}

LAST EXECUTION OUTPUT:
{execution_result}

OUTPUT VALIDITY PRE-CHECK (computed by the pipeline before this audit):
{output_validity_summary}

CHECKS:
1. OUTPUT VALIDITY — are Sharpe Ratio, Total Return, and Annualized Return present as real non-NaN floats?
   Read the pre-check line above. If it says CONFIRMED, Check 1 PASSES — do NOT override this.
   Negative Sharpe or negative returns are valid strategy results, not errors. Only NaN or missing lines FAIL.
   IMPORTANT: stderr RuntimeWarnings (e.g. divide by zero, invalid value) do NOT affect Check 1.
   Only judge by what is printed in the metric lines, not by warnings.
2. ALGORITHM FAITHFULNESS — does the script match the paper's signal formula, lookback windows,
   entry/exit thresholds, and position sizing?
3. FORWARD-LOOK BIAS — are all signals computed using only data available at time t (no t+1 leakage)?
4. SIGNAL CHAIN — does the chain from raw prices → signal → position → returns → metrics all connect?
   A signal that is computed but never used to open positions is a FAIL.

Respond with ONLY this JSON (no markdown fences):
{{
  "verdict": "PASS" or "FAIL",
  "what_is_wrong": "Specific issues with exact line references. Empty string if PASS.",
  "how_to_fix": "Exact code-level fix with line references and corrected snippets. Empty string if PASS.",
  "why": "Why each issue makes the backtest incorrect. Empty string if PASS."
}}

Write PASS only if ALL checks pass and metrics are real non-NaN values.
Write FAIL if any check fails or any metric is NaN.\
"""


def run(state: AgentState) -> dict:
    paper_id = state.get("paper_id", "unknown")
    safe_id = paper_id.replace("/", "_")
    execution_result = state.get("execution_result", "(no execution output)")
    impl_path = state.get("implementation_details_path", "")

    impl_details = ""
    if impl_path:
        try:
            impl_details = pathlib.Path(impl_path).read_text(encoding="utf-8")[:3000]
        except Exception:
            impl_details = "(implementation details not available)"

    script_path = f"outputs/backtests/{safe_id}.py"
    script_text = "(script not found)"
    try:
        script_text = pathlib.Path(script_path).read_text(encoding="utf-8")
    except Exception:
        pass

    print(f"\n{'━'*60}")
    print(f"[code_checker] INPUT paper_id        : {paper_id}")
    print(f"[code_checker] INPUT script_path     : {script_path} ({len(script_text.splitlines())} lines)")
    print(f"[code_checker] INPUT execution_result:\n{execution_result[-1000:]}")
    print(f"{'━'*60}\n")

    # Pre-extract metric values from execution_result (Python-level, no LLM)
    _sharpe_m = re.search(r'Sharpe Ratio:\s*(-?[\d.]+(?:e[+-]?\d+)?)', execution_result)
    _total_m  = re.search(r'Total Return:\s*(-?[\d.]+(?:e[+-]?\d+)?)', execution_result)
    _ann_m    = re.search(r'Annualized Return:\s*(-?[\d.]+(?:e[+-]?\d+)?)', execution_result)
    if _sharpe_m and _total_m and _ann_m:
        output_validity_summary = (
            f"CONFIRMED — Sharpe={_sharpe_m.group(1)}, "
            f"TotalRet={_total_m.group(1)}, AnnRet={_ann_m.group(1)} "
            f"(all real floats — Check 1 PASSES regardless of sign or magnitude)"
        )
    else:
        _missing = [k for k, m in [('Sharpe Ratio', _sharpe_m), ('Total Return', _total_m), ('Annualized Return', _ann_m)] if not m]
        output_validity_summary = f"MISSING — {_missing} not found in execution output — Check 1 FAILS."

    print(f"[code_checker] output_validity_summary: {output_validity_summary}")

    # Fast-path FAIL if no script
    if script_text == "(script not found)":
        verdict = "FAIL"
        feedback = f"VERDICT: FAIL\n\nWHAT IS WRONG:\nNo backtest script found at {script_path}.\n\nHOW TO FIX:\nRe-run openhands_code to generate the script."
        print(f"[code_checker] fast-path FAIL (no script)")
        return {
            **state,
            "code_checker_verdict": verdict,
            "code_checker_feedback": feedback,
            "flags": [f"[code_checker] verdict=FAIL (fast-path: no script) for {paper_id}"],
        }

    llm = PatchedChatOpenAI(
        model=os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-120b:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("GPT_OSS_API_KEY", ""),
    )

    prompt = _PROMPT.format(
        paper_id=paper_id,
        impl_details=impl_details,
        script_path=script_path,
        script_text=script_text[:8000],
        execution_result=execution_result[-1500:],
        output_validity_summary=output_validity_summary,
    )

    response = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    raw = response.content
    print(f"[code_checker] raw LLM response:\n{raw}")

    verdict = "ERROR"
    feedback = "code_checker LLM returned unparseable response"

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            verdict = str(parsed.get("verdict", "ERROR")).upper()
            if verdict not in ("PASS", "FAIL"):
                verdict = "ERROR"
            feedback = (
                f"VERDICT: {verdict}\n\n"
                f"WHAT IS WRONG:\n{parsed.get('what_is_wrong', '')}\n\n"
                f"HOW TO FIX:\n{parsed.get('how_to_fix', '')}\n\n"
                f"WHY:\n{parsed.get('why', '')}"
            )
        except Exception as exc:
            verdict = "ERROR"
            feedback = f"code_checker JSON parse error: {exc}\nRaw: {raw[:500]}"
    else:
        feedback = f"code_checker: no JSON found in response\nRaw: {raw[:500]}"

    print(f"\n{'━'*60}")
    print(f"[code_checker] verdict  : {verdict}")
    print(f"[code_checker] feedback :\n{feedback}")
    print(f"{'━'*60}\n")

    return {
        **state,
        "code_checker_verdict": verdict,
        "code_checker_feedback": feedback,
        "flags": [f"[code_checker] verdict={verdict} for {paper_id}"],
    }
