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
You are a data quality auditor for a quantitative finance backtesting pipeline.
You receive a manifest of fetched market data and the paper's implementation details.
Your job: decide whether the data is sufficient and correct for the strategy to be backtested.
Respond with a single JSON object — no markdown fences, no commentary outside the JSON.\
"""

_PROMPT = """\
Audit the fetched data for this strategy.

PAPER: {paper_id}

STRATEGY:
{strategy_description}

IMPLEMENTATION DETAILS (exact instruments and data requirements):
{impl_details}

FETCHED DATA MANIFEST:
  Tickers : {ticker_count} instruments
  Row count: {row_count}
  Date range: {date_range}
  Columns  : {columns}

CHECKS TO PERFORM:
1. row_count > 0  (fast-fail if 0 — no data fetched)
2. Ticker type matches strategy universe (individual stocks vs ETFs vs futures)
3. Date range is sufficient for the strategy (covers what the implementation details or brief specify)
4. All required columns present: date, ticker, open, high, low, close, adj_close, volume
5. Ticker count is plausible for the described universe

Respond with ONLY this JSON (no markdown fences):
{{
  "verdict": "PASS" or "FAIL",
  "what_is_wrong": "Specific description referencing actual observed values. Empty string if PASS.",
  "how_to_fix": "Concrete fix instructions for the fetcher. Empty string if PASS.",
  "why": "Why this matters for the strategy. Empty string if PASS."
}}

Write PASS only if ALL checks pass. Write FAIL if ANY check fails.\
"""


def run(state: AgentState) -> dict:
    paper_id = state.get("paper_id", "unknown")
    data_manifest = state.get("data_manifest", {})
    strategy_description = state.get("strategy_description", "(none)")
    impl_path = state.get("implementation_details_path", "")

    impl_details = ""
    if impl_path:
        try:
            impl_details = pathlib.Path(impl_path).read_text(encoding="utf-8")[:3000]
        except Exception:
            impl_details = "(implementation details not available)"

    row_count = data_manifest.get("row_count", 0)
    tickers = data_manifest.get("tickers", [])
    date_range = data_manifest.get("date_range", [])
    columns = data_manifest.get("columns", [])

    print(f"\n{'━'*60}")
    print(f"[data_checker] INPUT paper_id        : {paper_id}")
    print(f"[data_checker] INPUT ticker_count    : {len(tickers)}")
    print(f"[data_checker] INPUT row_count       : {row_count}")
    print(f"[data_checker] INPUT date_range      : {date_range}")
    print(f"[data_checker] INPUT columns         : {columns}")
    print(f"{'━'*60}\n")

    # Fast-path FAIL if manifest is missing or empty
    if not data_manifest or row_count == 0:
        verdict = "FAIL"
        feedback = "VERDICT: FAIL\n\nWHAT IS WRONG:\nManifest is empty or row_count=0 — no data was fetched.\n\nHOW TO FIX:\nRe-run the data fetcher and verify prices.parquet exists with row_count > 0."
        print(f"[data_checker] fast-path FAIL (no data)")
        return {
            **state,
            "data_checker_verdict": verdict,
            "data_checker_feedback": feedback,
            "flags": [f"[data_checker] verdict=FAIL (fast-path: no data) for {paper_id}"],
        }

    llm = PatchedChatOpenAI(
        model=os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-120b:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("GPT_OSS_API_KEY", ""),
    )

    prompt = _PROMPT.format(
        paper_id=paper_id,
        strategy_description=strategy_description[:1000],
        impl_details=impl_details,
        ticker_count=len(tickers),
        row_count=row_count,
        date_range=f"{date_range[0]} to {date_range[1]}" if len(date_range) == 2 else str(date_range),
        columns=columns,
    )

    response = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    raw = response.content
    print(f"[data_checker] raw LLM response:\n{raw}")

    verdict = "ERROR"
    feedback = "data_checker LLM returned unparseable response"

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
            feedback = f"data_checker JSON parse error: {exc}\nRaw: {raw[:500]}"
    else:
        feedback = f"data_checker: no JSON found in response\nRaw: {raw[:500]}"

    print(f"\n{'━'*60}")
    print(f"[data_checker] verdict  : {verdict}")
    print(f"[data_checker] feedback :\n{feedback}")
    print(f"{'━'*60}\n")

    return {
        **state,
        "data_checker_verdict": verdict,
        "data_checker_feedback": feedback,
        "flags": [f"[data_checker] verdict={verdict} for {paper_id}"],
    }
