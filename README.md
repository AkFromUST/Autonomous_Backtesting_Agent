# Quant Paper Autonomous Backtesting Agent — Prototype 1

Fully autonomous pipeline: feed in an arXiv paper URL, get a backtest report out. No human steps in the loop.

Built as a proof-of-concept for autonomous quantitative research replication. The agent reads a strategy paper, extracts the algorithm, fetches the required market data, writes and self-corrects a vectorbt backtest, validates the output, and produces a markdown report — end to end. 

As a prototype, the objective was to engineer the foundational layer where we leverage harness, reinforcement_loop and create the 2 phases, data_fetching and backtesting. The current prototype can handle easy-medium papers but don't rely on its output 100% as it is still in developement.

---

## Setup

### 1. Prerequisites

- Python 3.11 or higher
- A free [OpenRouter](https://openrouter.ai) account — all LLM calls route through OpenRouter

### 2. Clone and install

```bash
git clone <repo-url>
cd prototype_1

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The install pulls in LangGraph, OpenHands SDK, paper-qa, vectorbt, openbb, and sentence-transformers (for local BAAI/bge-m3 embeddings). First install takes several minutes. The bge-m3 model (~2GB) is downloaded on first run.

### 3. Configure API keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then fill it in:

```env
# OpenRouter API keys — both can be the same key, from openrouter.ai/keys
GPT_OSS_API_KEY=sk-or-v1-...
GPT_API_KEY_PAID=sk-or-v1-...

# OpenRouter model IDs — these work with a free account
GPT_OSS_MODEL=openai/gpt-oss-120b:free
REACT_MODEL=openai/gpt-oss-120b:free

# Groq — optional, not used in core pipeline
GROQ_API_KEY=
GROQ_MODEL=
```

**Getting an OpenRouter key:**
1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Go to **Keys** → **Create Key**
3. Copy the `sk-or-v1-...` key into both `GPT_OSS_API_KEY` and `GPT_API_KEY_PAID`

Both keys can be the same OpenRouter key. They are split in config to allow routing different agents to different rate-limit tiers if needed.

**Key roles:**

| Variable | Used by | Notes |
|---|---|---|
| `GPT_OSS_API_KEY` | openhands_data, openhands_code, data_checker, code_checker, reporter | High-throughput agents |
| `GPT_API_KEY_PAID` | phase1_guiding, phase2_guiding | Strategy/brief writing — can use a higher-tier model here |
| `GPT_OSS_MODEL` | All `GPT_OSS_API_KEY` agents | Any OpenRouter model ID |
| `REACT_MODEL` | All `GPT_API_KEY_PAID` agents | Any OpenRouter model ID |

### 4. Verify setup

```bash
source .venv/bin/activate
python -c "import langgraph, openhands, paperqa, vectorbt; print('OK')"
```

If that prints `OK`, you're ready to run.

---

## Running

### Full pipeline — paper URL in, report out

```bash
./run.sh
```

Reads `papers.txt` (one arXiv URL per line) and runs the complete pipeline for each paper. Outputs report to `outputs/reports/{paper_id}.md`.

To run a single paper directly:
```bash
./run.sh https://arxiv.org/pdf/2412.12458v1
```

### Phase 2 shortcut — skip data fetching, use pre-loaded data

```bash
./run.sh --p2
# or explicitly:
./run.sh --p2 paper1
```

Bypasses ingestor, strategy_planner, and the entire data-fetch phase. Loads pre-verified data from `data/paper1/` and runs directly from phase2_guiding → backtest → report. Use this when data is already on disk and you want to go straight to backtest generation.

Currently pre-loaded papers:

| Key | Paper | Universe |
|---|---|---|
| `paper1` | OU Pairs Trading (arXiv 2412.12458v1) | S&P 500, 496 tickers, 2010–2022, 1.5M rows |

---

## Architecture

```
papers.txt
    │
    ▼
[ dispatch ] ──────────────────────────────────────────────┐
    │ skip_phase1=False                    skip_phase1=True │
    ▼                                                       │
[ ingestor ]                                               │
    │                                                       │
    ▼                                                       │
[ strategy_planner ]                                        │
    │ NOT_TRADEABLE? ──► [ reporter ]                       │
    ▼                                                       │
[ phase1_guiding ] ◄── retry ──┐                           │
    │                          │                           │
    ▼                          │                           │
[ openhands_data ]             │                           │
    │                          │                           │
    ▼                          │                           │
[ normalizer ]                 │                           │
    │                          │                           │
    ▼                          │                           │
[ data_checker ] ──FAIL────────┘                          │
    │ PASS                                                  │
    └───────────────────────────────────────────────────────┘
    │
    ▼
[ phase2_guiding ] ◄── retry ──┐
    │                          │
    ▼                          │
[ openhands_code ]             │
    │                          │
    ▼                          │
[ code_checker ] ──FAIL────────┘
    │ PASS
    ▼
[ reporter ]
    │
    ▼
outputs/reports/{paper_id}.md
```

Orchestrated as a **LangGraph `StateGraph`**. State flows through all nodes as a typed dict (`AgentState`). Each retry loop has a max of 5 outer cycles before escalating to `HUMAN_REVIEW_NEEDED`.

---

## Phases

### Dispatch
Single routing node. Reads `skip_phase1: bool` from state. If True, jumps directly to `phase2_guiding`, bypassing all of phase 1. Otherwise falls through to `ingestor`. No computation — pure routing.

### Ingestor
Fetches the paper PDF. Supports arXiv URLs/IDs, ScienceDirect DOIs (via Unpaywall), and local PDF paths. Uses `arxiv.Client` + `pdfplumber`. Caches to `/tmp/arxiv_cache/`. Sets `paper_id` and `parsed_sections` in state.

### Strategy Planner
Runs a **paper-qa RAG loop** over the PDF to extract a structured strategy brief and detailed implementation spec. Uses `BAAI/bge-m3` local embeddings (sentence-transformers) for document chunking and retrieval. The agent issues multiple `query_paper(question)` tool calls — one per strategy dimension (universe, signal formula, parameters, execution rules). Outputs:
- `strategy_description` — 5-section developer brief
- `implementation_details_path` — path to a detailed formula file in `/tmp/arxiv_cache/`

`use_doc_details=False` is enforced in both top-level `Settings` and `ParsingSettings` to prevent Semantic Scholar API calls during document ingestion.

Prefixes `NOT_TRADEABLE:` on `strategy_description` if the paper's instrument universe is incompatible (futures, alternative data) — this routes directly to `reporter`, skipping all backtesting.

### phase1_guiding
Direct LLM call (`REACT_MODEL` via `GPT_API_KEY_PAID`). Reads `strategy_description` and `implementation_details_path`, writes a precise data-fetch brief (`openhands_data_brief`) to state. Specifies: exact ticker list, asset class, date range (2010-01-01 to 2023-01-01), output path, and required parquet schema. Sets a timestamped `data_dir` in `outputs/data/{paper_id}_{timestamp}/`.

### openhands_data
**OpenHands SDK agent** (`GPT_OSS_MODEL`, `max_iterations=30`). Receives the brief from phase1_guiding and executes inside the project workspace with `TerminalTool` + `FileEditorTool`. Fetches OHLCV data via `openbb`/`yfinance`. Required outputs:
- `prices.parquet` — long format, columns: `date, ticker, open, high, low, close, adj_close, volume`. `date` must be a column (not index — always `reset_index()` before saving).
- `data_manifest.json` — `{file, tickers, date_range, row_count, columns}`

Uses `LLMSummarizingCondenser(max_size=240, keep_first=10)` to prevent within-cycle context bloat. Timestamped `persistence_dir` per outer cycle prevents cross-cycle context flooding. Self-contained correction messages re-embed the current fetch script and full attempt history on each retry turn. Agent is only considered done when terminal confirms `row_count > 0`.

### Normalizer
Lightweight stateless node. Reads `data_manifest.json` from `data_dir` into state. Populates `data_manifest`, `data_coverage_note`.

### data_checker
**Direct LLM call** (`GPT_OSS_MODEL` via `GPT_OSS_API_KEY`). Not an OpenHands agent. Reads `data_manifest` from state and `implementation_details` from disk. Checks: row_count > 0, ticker type matches paper universe, date range covers the strategy window, required columns present, ticker count plausible. Returns `data_checker_verdict` (`PASS`/`FAIL`) and `data_checker_feedback` directly into state via JSON regex parse — no file writing. Fast-paths FAIL if manifest is empty or row_count=0.

### phase2_guiding
Direct LLM call (`REACT_MODEL` via `GPT_API_KEY_PAID`). Reads `strategy_description`, `implementation_details`, `data_dir`, `data_manifest`, and any `code_checker_feedback` from a previous failed cycle. Writes `openhands_code_brief` to state — a complete coding brief containing:
- Verbatim signal formula, lookback windows, entry/exit thresholds, position sizing
- Canonical data loading pattern (hardcoded `data_dir` path, pivot to wide-format close matrix)
- Output script path: `outputs/backtests/{paper_id}.py`
- Required metric output lines (`Sharpe Ratio: <float>`, `Total Return: <float>`, `Annualized Return: <float>`)
- Previous checker feedback with fix instructions (on retry cycles)
- `## DIAGNOSTIC CHECKPOINTS` section — 3–5 abstract stage checkpoints derived from the algorithm (stage names only, no variable names — the driver implements the actual print statements)

On the first attempt (`phase2_loop <= 1`) deletes any stale script at the output path. On retries, preserves the existing script for targeted fixes.

### openhands_code
**OpenHands SDK agent** (`GPT_OSS_MODEL`, `max_iterations=100`). The core backtest writer. Receives the brief and iterates: write script → run script → read terminal output → fix errors → run again, until the terminal shows non-NaN Sharpe Ratio, Total Return, and Annualized Return. The agent is mandated to signal done **only when the terminal confirms real float metrics** — not when the script is merely written.

Key runtime constraints enforced via system suffix:
- **SPREAD/PAIR POSITION RETURNS** — never compute returns as `(spread_t - spread_{t-1}) / spread_{t-1}` (inverts when spread < 0). Always use leg pct returns: `ret_long - ret_short`.
- **MEAN-REVERSION SPEED** — print half-life distribution after OU estimation; filter pairs where `half_life > test_horizon_days / 2`; flag if `lambda < 1e-4/day`.
- **DIAGNOSTIC CHECKPOINTS** — implement all checkpoints from the brief as `print()` statements at the corresponding algorithm stages.
- **PERFORMANCE METRICS** — output as `% returns` (multiply by 100), never raw fractions.

Uses `LLMSummarizingCondenser(max_size=240, keep_first=10)` and a new timestamped `persistence_dir` per outer cycle. Self-contained correction messages re-embed the full current script text and attempt history.

After the agent exits, the node runs the script in a subprocess with a 3600s timeout and captures stdout+stderr as `execution_result` in state.

### code_checker
**Direct LLM call** (`GPT_OSS_MODEL` via `GPT_OSS_API_KEY`). Not an OpenHands agent. Reads `execution_result` and the script from `outputs/backtests/{paper_id}.py` directly. Checks:
1. **Output validity** — Sharpe Ratio, Total Return, Annualized Return present as real non-NaN floats. Auto-FAIL if any metric is NaN.
2. **Algorithm faithfulness** — signal formula, lookback windows, entry/exit thresholds match `implementation_details`.
3. **Forward-look bias** — all signals computed using only data available at time t.
4. **Signal chain completeness** — raw prices → signal → position → returns → metrics all connected. A computed-but-unused signal is a FAIL.

Returns `code_checker_verdict` and `code_checker_feedback` into state via JSON regex parse.

### Reporter
Direct LLM call (`GPT_OSS_MODEL`). Writes a professional markdown report to `outputs/reports/{paper_id}.md` structured as:
1. Headline — strategy name, backtest period, key result
2. Performance Results — metrics table (Sharpe, Total Return %, Annualized Return %)
3. Strategy Logic — exact signal rule, lookback windows, thresholds, rebalancing
4. Backtest Implementation — pair selection, parameter estimation, signal construction, test period
5. Data — universe, source, date range, instrument count
6. Caveats & Limitations — no transaction costs, forward-look risk, implementation gaps
7. Verdict — economic significance, whether to investigate further

Also appends a structured entry to `outputs/run_summary.json` with extracted metrics.

---

## State Schema

All agents communicate exclusively through `AgentState` (LangGraph `TypedDict`). No agent writes to shared files for inter-agent communication — only the final backtest script and report are written to disk.

```python
class AgentState(TypedDict):
    paper_url: str
    paper_id: str
    parsed_sections: dict                 # ingestor output
    strategy_description: str            # strategy_planner output
    implementation_details_path: str     # path to formula file in /tmp/arxiv_cache/
    skip_phase1: bool                    # dispatch routing flag
    openhands_data_brief: str            # phase1_guiding → openhands_data
    openhands_code_brief: str            # phase2_guiding → openhands_code
    data_manifest: dict                  # {tickers, date_range, row_count, columns}
    data_dir: str                        # timestamped data directory
    data_checker_verdict: str            # "PASS" | "FAIL" | "ERROR"
    data_checker_feedback: str
    code_checker_verdict: str            # "PASS" | "FAIL" | "ERROR"
    code_checker_feedback: str
    generated_code: str
    execution_result: str                # subprocess stdout+stderr
    final_report: str
    retry_counts: dict[str, int]         # {"phase1_loop": N, "phase2_loop": N}
    flags: Annotated[list[str], operator.add]
    status: str                          # "RUNNING" | "DONE" | "HUMAN_REVIEW_NEEDED"
```

---

## Tech Stack

| Component | Library |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| LLM calls | `PatchedChatOpenAI` → OpenRouter |
| Paper fetching | `arxiv.Client` + `pdfplumber` |
| Paper RAG | `paper-qa[local]` + `BAAI/bge-m3` (sentence-transformers) |
| Agentic execution | OpenHands SDK (`LLM`, `Agent`, `Conversation`, `LLMSummarizingCondenser`) |
| Market data | `openbb` + `openbb-yfinance` |
| Backtesting | `vectorbt` |
| Context management | `LLMSummarizingCondenser(max_size=240, keep_first=10)` |

---

## Outputs

```
outputs/
├── reports/          ← {paper_id}.md — full markdown report per paper
├── backtests/        ← {paper_id}.py — generated backtest script
├── data/             ← {paper_id}_{timestamp}/ — fetched parquet + manifest (full pipeline)
├── openhands_logs/   ← conversation persistence per agent cycle
└── run_summary.json  ← structured JSON log of all runs with extracted metrics
```

---

## Pre-loaded Data (shortcut mode)

`data/paper1/` contains pre-fetched, pre-verified data for paper 1 (OU Pairs Trading):

| File | Contents |
|---|---|
| `prices.parquet` | 496 S&P 500 constituents, daily OHLCV, 2010-01-04 to 2022-12-30, 1,524,003 rows |
| `data_manifest.json` | Manifest with ticker list, date range, row count, columns |
| `strategy_description.txt` | Pre-extracted strategy brief (loaded into state in `--p2` mode) |
| `implementation_details.txt` | Exact formulas and parameters extracted from the paper |
