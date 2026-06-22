from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.tools import FileEditorTool, TerminalTool, register_default_tools
from pydantic import SecretStr

from src.state import AgentState

PROJECT_ROOT = pathlib.Path(".").resolve()
CONV_LOG_DIR = PROJECT_ROOT / "outputs" / "openhands_logs"

register_default_tools(enable_browser=False)

_PHASE1_SUFFIX = f"""\
You are the driver in a pair programming setup. Your navigator reads everything you print and \
will correct you if your assumptions are wrong. Use that channel.

COMMUNICATION PROTOCOL — mandatory, do this before every fetch:
  PLAN: I will fetch [exact instrument list] using [provider] because [reason from brief].
  ASSUMPTION: [anything you are assuming that was not explicit in the brief]

If uncertain about which exact instruments to use, print:
  UNCERTAIN: I am not sure whether [option A] or [option B]. Proceeding with [choice] because [reason].

After saving, print:
  SUMMARY: Saved [N] tickers to prices.parquet. Row count: [N]. Date range: [min] to [max].

WORKSPACE — mandatory:
- Your working directory is {PROJECT_ROOT}
- Write all Python fetch scripts to {PROJECT_ROOT} (e.g. {PROJECT_ROOT}/fetch_data.py)
- Do NOT create subdirectories (scripts/, tmp/, src/). They do not exist.
- The data directory path is specified exactly in your brief — use that absolute path verbatim.
- Do NOT list or explore outputs/data/ — it contains data from other papers and will mislead you.

WEB SEARCH — use this to resolve any API uncertainty before writing code:
  python -c "from ddgs import DDGS; [print(r['title'], r['href'], r['body'][:300]) for r in DDGS().text('YOUR QUERY HERE', max_results=3)]"
  Examples:
    - 'yfinance download OHLCV adjusted close 2024 pandas dataframe'
    - 'openbb platform equity historical data python example'
    - 'S&P 500 constituents list Wikipedia python requests'
  Run a search BEFORE writing any data-fetching code if you are unsure of the current API.

Output contract:
- Use the exact date range specified in your brief — do not assume or hardcode any dates.
- Write Python scripts and run them — do not use inline python -c with embedded quotes

DATA FORMAT REQUIREMENT — mandatory, no exceptions:
Save all price data as a SINGLE parquet file at the exact path given in your brief.
Required schema (long format — one row per date-ticker pair):
  - date: datetime column (NOT the DataFrame index — call reset_index() before saving)
  - ticker: string column
  - open, high, low, close: float columns
  - adj_close: float column (use close if split adjustment unavailable)
  - volume: float column (use 0.0 if unavailable)

Do NOT save one file per ticker.
Do NOT use MultiIndex columns.
Do NOT leave date as the DataFrame index — always call reset_index() before df.to_parquet().

After writing prices.parquet, write data_manifest.json to the same directory:
  {{"file": "prices.parquet", "tickers": [...], "date_range": ["YYYY-MM-DD", "YYYY-MM-DD"], "row_count": N, "columns": [...]}}

Signal done ONLY when:
  1. prices.parquet exists at the exact path from your brief with row_count > 0
  2. data_manifest.json exists in the same directory
"""


def _find_fetch_script(project_root: pathlib.Path) -> tuple[str, str] | tuple[None, None]:
    """Return (name, content) of the most recently modified fetch script at project root."""
    skip = {"test_phase1_enhanced.py", "test_phase1.py", "test_phase2.py", "test_phase2_enhanced.py"}
    candidates = [
        f for f in project_root.glob("*.py")
        if f.name not in skip and ("fetch" in f.name.lower() or f.name.startswith("fetch"))
    ]
    if not candidates:
        all_py = [
            f for f in project_root.glob("*.py")
            if f.name not in skip and not f.name.startswith("test_")
        ]
        candidates = all_py
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda f: f.stat().st_mtime)
    try:
        return latest.name, latest.read_text(encoding="utf-8")
    except Exception:
        return latest.name, None


def run(state: AgentState) -> dict:
    brief = state.get("openhands_data_brief", "")
    paper_id = state.get("paper_id", "unknown")
    safe_id = paper_id.replace("/", "_")
    data_dir = state.get("data_dir", "")

    oh_llm = LLM(
        model="openrouter/" + os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-120b:free"),
        api_key=SecretStr(os.getenv("GPT_OSS_API_KEY", "")),
        reasoning_effort="none",
        usage_id="phase1",
    )

    oh_agent = Agent(
        llm=oh_llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ],
        agent_context=AgentContext(system_message_suffix=_PHASE1_SUFFIX),
        max_iterations=50,
        condenser=LLMSummarizingCondenser(llm=oh_llm, max_size=240, keep_first=10),
    )

    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    persistence_dir = str(CONV_LOG_DIR / safe_id / f"phase1_oh_{_ts}")
    conversation = Conversation(
        agent=oh_agent,
        workspace=str(PROJECT_ROOT),
        persistence_dir=persistence_dir,
    )

    parquet_path = pathlib.Path(data_dir) / "prices.parquet"
    manifest_path = pathlib.Path(data_dir) / "data_manifest.json"

    flags: list[str] = []
    attempt_history: list[str] = []
    MAX_DIALOGUE = 3

    conversation.send_message(brief)
    for turn in range(MAX_DIALOGUE):
        print(f"[openhands_data] dialogue turn {turn + 1}/{MAX_DIALOGUE}")
        try:
            conversation.run()
            flags.append(f"[openhands_data] OpenHands run complete (turn {turn + 1})")
        except Exception as exc:
            flags.append(f"[openhands_data] OpenHands error (turn {turn + 1}): {exc}")
            break

        parquet_ok = parquet_path.exists() and parquet_path.stat().st_size > 0
        row_count = 0
        if manifest_path.exists():
            try:
                row_count = json.loads(manifest_path.read_text()).get("row_count", 0)
            except Exception:
                pass

        print(
            f"[openhands_data] turn {turn + 1}: "
            f"parquet_ok={parquet_ok}, row_count={row_count}"
        )

        if parquet_ok and row_count > 0:
            flags.append(f"[openhands_data] data verified on turn {turn + 1}")
            break

        if turn < MAX_DIALOGUE - 1:
            if not parquet_path.exists():
                issue = f"prices.parquet NOT found at {parquet_path}"
            elif not parquet_ok:
                issue = f"prices.parquet exists but is empty (0 bytes)"
            else:
                issue = f"prices.parquet exists but row_count={row_count} (manifest reports 0 rows)"

            attempt_history.append(f"Turn {turn + 1}: {issue}")
            history_block = (
                "\n".join(attempt_history[:-1]) if len(attempt_history) > 1
                else "(this is the first failure)"
            )

            script_name, script_content = _find_fetch_script(PROJECT_ROOT)
            script_block = ""
            if script_name and script_content:
                lines = script_content.splitlines()
                script_block = (
                    f"\nCURRENT FETCH SCRIPT ({script_name}, {len(lines)} lines):\n"
                    f"{script_content}\n"
                )

            correction = (
                f"ATTEMPT HISTORY:\n{history_block}\n"
                f"{script_block}\n"
                f"CURRENT ISSUE: {issue}\n\n"
                f"Fix the fetch script, run it, and verify {parquet_path} exists with row_count > 0. "
                f"Then write data_manifest.json to {manifest_path}. "
                f"If the API call is failing, use the WEB SEARCH command in your instructions to look up the correct current API."
            )
            print(f"[openhands_data] sending self-contained correction")
            conversation.send_message(correction)

    data_manifest: dict = {}
    if manifest_path.exists():
        try:
            data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            flags.append(f"[openhands_data] fetched {len(data_manifest.get('tickers', []))} tickers")
        except Exception:
            flags.append("[openhands_data] data_manifest.json malformed")
    else:
        flags.append("[openhands_data] no data_manifest.json — data fetch failed")

    return {**state, "data_manifest": data_manifest, "flags": flags}
