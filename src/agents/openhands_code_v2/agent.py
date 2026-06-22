from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from datetime import datetime

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.tools import FileEditorTool, TerminalTool, register_default_tools
from pydantic import SecretStr

from src.state import AgentState, LintError

PROJECT_ROOT = pathlib.Path(".").resolve()
CONV_LOG_DIR = PROJECT_ROOT / "outputs" / "openhands_logs"
SCRIPT_TIMEOUT = 3600

register_default_tools(enable_browser=False)

_PHASE2_SUFFIX = f"""\
You are the driver in a pair programming setup. Your navigator reads everything you print and will \
correct you if your algorithm interpretation is wrong. Use that channel.

COMMUNICATION PROTOCOL — mandatory, before writing any code:
  ALGORITHM INTERPRETATION: [your understanding of the exact signal formula, lookback windows, \
position sizing, and rebalancing frequency from the brief]
  ASSUMPTION: [any parameter you are choosing that was not explicitly stated in the brief]

If a formula detail is unclear, print:
  UNCERTAIN: [parameter or formula element] — proceeding with [value/interpretation] because [reason].

Your navigator reads your ALGORITHM INTERPRETATION before you write a single line of code. \
If it is wrong they will correct it immediately — be explicit now rather than wrong later.

You are part of an autonomous pipeline that independently replicates quantitative finance research papers. \
Your job is to implement a working backtest that faithfully reproduces the paper's strategy.

WORKSPACE — mandatory, read before writing any file:
- Your working directory is {PROJECT_ROOT}
- Write the backtest script to the path given in your brief (under {PROJECT_ROOT}/outputs/backtests/).
- Do NOT create subdirectories that are not in your brief. Do NOT write to scripts/, tmp/, or any \
new directory — they do not exist and file_editor will fail with ENOENT.

Data is already normalized and ready to use. Load it with the pattern given in your brief.
The data path is specified exactly in your brief — use that absolute path verbatim.
Do NOT list or explore outputs/data/ — it contains data from other papers and will mislead you.

DIAGNOSTIC CHECKPOINTS — your brief contains a ## DIAGNOSTIC CHECKPOINTS section:
After implementing each listed stage, add ONE print() statement confirming that stage completed
and its key property (e.g., count, shape, or non-NaN status). Choose variable names from your
own implementation. These diagnostic prints must appear in the script BEFORE the final metrics.

PERFORMANCE METRICS — mandatory constraints, no exceptions:
- All metrics must be computed from percentage (dimensionless) daily returns, never raw dollar PnL.
- For any long/short position: daily contribution = position_weight × pct_change_of_held_asset.
- If the paper does not specify capital allocation, assume equal dollar weighting (each leg receives
  the same notional, so its contribution is its percentage return, not its price change).
- Equity curve: starts at 1.0, updated each day as equity *= (1 + daily_return).
- Total Return: equity[-1] / equity[0] - 1  → a number like 0.15 means 15%, not $15.
- Sharpe Ratio: annualised from daily percentage returns, sqrt(252) scaling, rfr=0.
- A Total Return above 100x or a negative equity curve signals returns are in dollars not percentages — fix it.

SPREAD / PAIR POSITION RETURNS — if the strategy trades a spread between two assets:
- NEVER compute returns as (spread_t - spread_{{t-1}}) / spread_{{t-1}}.
  The spread can be negative (e.g. P_i < P_j), which flips the sign and makes profitable
  mean reversion look like a loss. This is a silent bug that produces large negative equity.
- ALWAYS compute returns from individual leg percentage changes:
    ret_i = (price_i_today - price_i_prev) / price_i_prev
    ret_j = (price_j_today - price_j_prev) / price_j_prev
    long_spread_return  = ret_i - ret_j   # long i, short j
    short_spread_return = ret_j - ret_i   # short i, long j
- This is correct regardless of whether the spread is positive, negative, or near zero.
- After implementing, add a diagnostic: print the mean daily return for the first 5 active pairs
  to sanity-check sign before computing portfolio metrics.

MEAN-REVERSION SPEED — if the strategy estimates a mean-reversion parameter (lambda, kappa,
half-life, or similar speed-of-adjustment coefficient):
- After estimation, print the distribution: count, mean, min, and max of half-lives.
  Half-life = ln(2) / lambda  (in the same time unit as your data, e.g. trading days).
- Pairs/assets with half-life > 2× your test horizon are effectively random walks over that
  horizon — their signals will fire but the spread will not revert. Filter them out or flag them.
- If the paper does not specify a minimum speed threshold, use half-life < test_horizon_days / 2
  as a reasonable default and log how many pairs pass.
- A min_lambda near zero (e.g. < 1e-4 per day) is a warning sign: add a print so it is visible.

Constraints:
- No data downloads inside the script — all data is already on disk.
- Do NOT re-discover the data format. The schema is fixed (from your brief).
- The script must output performance metrics (Sharpe ratio, total return, annualized return).
- Only signal done when the script exits with code 0 and terminal shows real (non-NaN) metrics.
"""


def run(state: AgentState) -> dict:
    brief = state.get("openhands_code_brief", "")
    paper_id = state.get("paper_id", "unknown")
    safe_id = paper_id.replace("/", "_")
    script_path = f"outputs/backtests/{safe_id}.py"

    oh_llm = LLM(
        model="openrouter/" + os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-120b:free"),
        api_key=SecretStr(os.getenv("GPT_OSS_API_KEY", "")),
        reasoning_effort="none",
        usage_id="phase2_v2",
    )

    oh_agent = Agent(
        llm=oh_llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ],
        agent_context=AgentContext(system_message_suffix=_PHASE2_SUFFIX),
        max_iterations=100,
        condenser=LLMSummarizingCondenser(llm=oh_llm, max_size=240, keep_first=10),
    )

    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    persistence_dir = str(CONV_LOG_DIR / safe_id / f"phase2_oh_v2_{_ts}")
    conversation = Conversation(
        agent=oh_agent,
        workspace=str(PROJECT_ROOT),
        persistence_dir=persistence_dir,
    )

    flags: list[str] = []
    attempt_history: list[str] = []
    MAX_DIALOGUE = 3

    conversation.send_message(brief)
    for turn in range(MAX_DIALOGUE):
        print(f"[openhands_code_v2] dialogue turn {turn + 1}/{MAX_DIALOGUE}")
        try:
            conversation.run()
            flags.append(f"[openhands_code_v2] OpenHands run complete (turn {turn + 1})")
        except Exception as exc:
            flags.append(f"[openhands_code_v2] OpenHands error (turn {turn + 1}): {exc}")
            break

        script_file_check = pathlib.Path(script_path)
        if not script_file_check.exists():
            print(f"[openhands_code_v2] turn {turn + 1}: script not found at {script_path}")
            attempt_history.append(f"Turn {turn + 1}: script was not written to {script_path}")
            if turn < MAX_DIALOGUE - 1:
                conversation.send_message(
                    f"Script was not written. Use file_editor to create it at {script_path}, "
                    f"then run it with the terminal tool and confirm real metrics."
                )
            continue

        try:
            check_result = subprocess.run(
                [sys.executable, str(script_file_check)],
                capture_output=True, text=True, timeout=SCRIPT_TIMEOUT,
                cwd=str(PROJECT_ROOT),
            )
            check_rc = check_result.returncode
            check_out = (check_result.stdout + check_result.stderr).strip()
        except subprocess.TimeoutExpired:
            check_rc = -1
            check_out = f"TimeoutExpired: exceeded {SCRIPT_TIMEOUT}s"
        except Exception as exc:
            check_rc = -1
            check_out = str(exc)

        print(f"[openhands_code_v2] turn {turn + 1} rc={check_rc} check output:\n{check_out[-800:]}")

        # Only check NaN in the three required metric lines, not in diagnostic prints
        _metric_lines = [l for l in check_out.split('\n')
                         if any(k in l for k in ('Sharpe Ratio:', 'Total Return:', 'Annualized Return:'))]
        _metrics_present = len(_metric_lines) == 3
        _metrics_nan = any('nan' in l.lower() for l in _metric_lines)

        script_ok = (
            check_rc == 0
            and _metrics_present
            and not _metrics_nan
            and "Traceback" not in check_out
        )

        if script_ok:
            flags.append(f"[openhands_code_v2] script ran cleanly with output on turn {turn + 1}")
            break

        if turn < MAX_DIALOGUE - 1:
            # Determine issue label for history
            if check_rc != 0:
                issue_label = f"rc={check_rc}, error: {check_out[:150]}"
            elif _metrics_nan:
                issue_label = f"rc=0 but metric lines contain NaN: {_metric_lines}"
            elif not _metrics_present:
                issue_label = f"rc=0 but only {len(_metric_lines)}/3 metric lines printed"
            elif "Traceback" in check_out:
                issue_label = f"exception: {check_out[:150]}"
            else:
                issue_label = f"rc=0 unexpected: {check_out[:150]}"
            attempt_history.append(f"Turn {turn + 1}: {issue_label}")

            # Read current script for self-contained correction
            script_text = script_file_check.read_text(encoding="utf-8") if script_file_check.exists() else "(not found)"
            script_lines = len(script_text.splitlines()) if script_text != "(not found)" else 0

            history_block = (
                "\n".join(attempt_history[:-1]) if len(attempt_history) > 1
                else "(this is the first failure)"
            )

            if check_rc != 0:
                instruction = "Fix the error. Do not rewrite from scratch unless the logic is fundamentally broken."
            elif _metrics_nan:
                instruction = (
                    "The METRIC LINES (Sharpe Ratio / Total Return / Annualized Return) show NaN. "
                    "Trace the signal → position → return chain: "
                    "check that signals are non-zero, positions are opened, and returns are non-zero. "
                    "Use your diagnostic checkpoint prints to narrow which stage produces no output. "
                    "Diagnostic prints showing np.float64(nan) on early rows are expected — only fix the final metric lines."
                )
            else:
                instruction = (
                    "The script ran but did not print all three required metric lines. "
                    "Add: print(f'Sharpe Ratio: {sharpe}'), print(f'Total Return: {total}'), "
                    "print(f'Annualized Return: {ann}') at the end of the script."
                )

            correction = (
                f"ATTEMPT HISTORY (what has been tried):\n{history_block}\n\n"
                f"CURRENT SCRIPT ({script_path}, {script_lines} lines):\n"
                f"{script_text}\n\n"
                f"CURRENT ISSUE (rc={check_rc}):\n{check_out[-600:]}\n\n"
                f"{instruction}"
            )
            print("[openhands_code_v2] sending self-contained correction")
            conversation.send_message(correction)

    # Final execution to capture execution_result for state
    script_file = pathlib.Path(script_path)
    generated_code = ""
    execution_result = ""
    lint_errors: list[LintError] = []

    if script_file.exists():
        generated_code = script_file.read_text(encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(script_file)],
                capture_output=True,
                text=True,
                timeout=SCRIPT_TIMEOUT,
                cwd=str(PROJECT_ROOT),
            )
            execution_result = (
                f"returncode: {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
            if result.returncode == 0:
                flags.append("[openhands_code_v2] script executed successfully")
            else:
                flags.append(f"[openhands_code_v2] script exited with returncode {result.returncode}")
                lint_errors = [LintError(line=None, code="ExecutionError", message=execution_result)]
        except subprocess.TimeoutExpired:
            execution_result = f"TimeoutExpired: exceeded {SCRIPT_TIMEOUT}s"
            lint_errors = [LintError(line=None, code="ExecutionError", message=execution_result)]
            flags.append("[openhands_code_v2] script timed out")
        except Exception as exc:
            execution_result = f"SubprocessError: {exc}"
            lint_errors = [LintError(line=None, code="ExecutionError", message=execution_result)]
            flags.append(f"[openhands_code_v2] execution error: {exc}")
    else:
        execution_result = "Agent did not write a backtest file"
        lint_errors = [LintError(line=None, code="ExecutionError", message=execution_result)]
        flags.append("[openhands_code_v2] no backtest script produced")

    return {
        **state,
        "generated_code": generated_code,
        "execution_result": execution_result,
        "lint_errors": lint_errors,
        "flags": flags,
    }
