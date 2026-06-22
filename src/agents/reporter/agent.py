from __future__ import annotations

import json
import os
import pathlib
import re
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.state import AgentState

_OUTPUTS_DIR = pathlib.Path(__file__).parents[3] / "outputs"
_REPORTS_DIR = _OUTPUTS_DIR / "reports"
_SUMMARY_PATH = _OUTPUTS_DIR / "run_summary.json"

_SYSTEM = """\
You are a quantitative research analyst writing a concise post-backtest report for a portfolio manager.
The audience is a systematic trading professional — be precise, direct, and avoid filler language.
Lead with numbers. Use markdown tables for metrics. Every claim must reference the data or the paper.\
"""

_PROMPT = """\
Write a professional backtest report in markdown for the paper below.

REPORT STRUCTURE — follow this order exactly:
1. A one-line headline at the top: strategy name, backtest period, and the key result number.
2. ## Performance Results — a markdown table with Sharpe Ratio, Total Return (%), and Annualized Return (%).
   Then 2-3 sentences interpreting what the numbers mean (good/bad/expected for this strategy type).
3. ## Strategy Logic — explain the trading logic the paper proposes. Include the exact signal rule,
   lookback windows, entry/exit thresholds, and rebalancing frequency. Be specific.
4. ## Backtest Implementation — describe how the strategy was implemented in code: pair selection method,
   OU parameter estimation, signal construction, position sizing, and the test period used.
5. ## Data — one short paragraph: universe, source, date range, number of instruments, data format.
6. ## Caveats & Limitations — bullet list. Include: no transaction costs, any forward-look risk noted,
   limitations of the paper, and any implementation caveats from below.
7. ## Verdict — 1-2 sentences. Is this result economically meaningful? Should it be investigated further?

---

PAPER INFORMATION
Paper ID: {paper_id}
Title: {title}
Strategy description (from paper-qa RAG):
{strategy_description}

Implementation details (exact formulas from paper):
{impl_details}

---

BACKTEST RESULTS
{metrics_block}

Raw execution output:
{execution_result}

---

DATA COVERAGE
Tickers: {ticker_count} instruments
Date range: {date_range}
Data manifest: {manifest_summary}

---

CODE CHECKER VERDICT: {checker_verdict}
Checker feedback: {checker_feedback}

Pipeline flags:
{flags}
"""


def _extract_metrics(execution_result: str) -> dict:
    """Pull Sharpe, Total Return, Annualized Return out of execution_result stdout."""
    patterns = {
        "sharpe": r"Sharpe\s*Ratio\s*[:\s]+(-?\d+\.?\d*(?:e[+-]?\d+)?)",
        "total_return": r"Total\s*Return\s*[:\s]+(-?\d+\.?\d*(?:e[+-]?\d+)?)",
        "annual_return": r"Annualized\s*Return\s*[:\s]+(-?\d+\.?\d*(?:e[+-]?\d+)?)",
    }
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, execution_result, re.IGNORECASE)
        out[key] = float(m.group(1)) if m else None
    return out


def _format_metrics_block(metrics: dict) -> str:
    def fmt(val, pct=False):
        if val is None:
            return "N/A"
        if pct:
            return f"{val * 100:.2f}%"
        return f"{val:.4f}"

    return (
        f"| Metric             | Value          |\n"
        f"|-------------------|----------------|\n"
        f"| Sharpe Ratio      | {fmt(metrics.get('sharpe'))}     |\n"
        f"| Total Return      | {fmt(metrics.get('total_return'), pct=True)} |\n"
        f"| Annualized Return | {fmt(metrics.get('annual_return'), pct=True)} |"
    )


def run(state: AgentState) -> dict:
    paper_id = state.get("paper_id", "unknown")
    safe_id = paper_id.replace("/", "_")
    sections = state.get("parsed_sections", {})
    title = sections.get("title") or sections.get("Title") or paper_id
    strategy_description = state.get("strategy_description", "") or "Not extracted."
    execution_result = state.get("execution_result", "") or "Not executed."
    data_manifest = state.get("data_manifest", {})
    checker_verdict = state.get("code_checker_verdict", "N/A")
    checker_feedback = state.get("code_checker_feedback", "") or "None."
    flags = state.get("flags", [])
    status = state.get("status", "RUNNING")

    # Read implementation details from disk
    impl_details = ""
    impl_path = state.get("implementation_details_path", "")
    if impl_path:
        try:
            impl_details = pathlib.Path(impl_path).read_text(encoding="utf-8")[:3000]
        except Exception:
            impl_details = "(implementation details not available)"

    # Extract metrics
    metrics = _extract_metrics(execution_result)
    metrics_block = _format_metrics_block(metrics)

    # Data coverage
    tickers = data_manifest.get("tickers", [])
    ticker_count = len(tickers)
    date_range = data_manifest.get("date_range", ["unknown", "unknown"])
    manifest_summary = (
        f"{ticker_count} tickers, {data_manifest.get('row_count', '?')} rows, "
        f"columns: {data_manifest.get('columns', [])}"
        if data_manifest else "No manifest available."
    )

    llm = ChatOpenAI(
        model=os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-120b:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("GPT_OSS_API_KEY", ""),
    )

    prompt = _PROMPT.format(
        paper_id=paper_id,
        title=title,
        strategy_description=strategy_description[:2000],
        impl_details=impl_details,
        metrics_block=metrics_block,
        execution_result=execution_result[-2000:],
        ticker_count=ticker_count,
        date_range=f"{date_range[0]} to {date_range[1]}" if len(date_range) == 2 else str(date_range),
        manifest_summary=manifest_summary,
        checker_verdict=checker_verdict,
        checker_feedback=checker_feedback[:500],
        flags="\n".join(f"- {f}" for f in flags[-20:]) if flags else "None",
    )

    print(f"[reporter] generating report for {paper_id} (status={status})")
    report_md = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]).content

    # Prepend a generated-by footer
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"> *Report generated autonomously by NineMasts Quant Pipeline · {timestamp}*\n\n"
    report_md = header + report_md

    # Write per-paper report
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORTS_DIR / f"{safe_id}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"[reporter] report written to {report_path}")

    # Append to run_summary.json
    summary_entry = {
        "paper_id": paper_id,
        "title": title,
        "status": "DONE",
        "checker_verdict": checker_verdict,
        "metrics": metrics,
        "report_path": str(report_path),
        "timestamp": timestamp,
    }
    summary: list = []
    if _SUMMARY_PATH.exists():
        try:
            raw = _SUMMARY_PATH.read_text().strip()
            if raw:
                loaded = json.loads(raw)
                summary = loaded if isinstance(loaded, list) else []
        except Exception:
            pass
    summary.append(summary_entry)
    _SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {**state, "final_report": report_md, "status": "DONE"}
