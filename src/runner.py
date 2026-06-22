from __future__ import annotations

import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.graph import graph
from src.state import AgentState

_DEFAULT_PAPERS_FILE = Path(__file__).parent.parent / "papers.txt"
_SHARED_DATA = Path(__file__).parent.parent / "data"

_P2_SHORTCUTS = {
    "paper1": {
        "paper_url": "https://arxiv.org/pdf/2412.12458v1",
        "paper_id": "2412.12458v1",
        "data_dir": str(_SHARED_DATA / "paper1"),
    },
    "paper8": {
        "paper_url": "local://paper8.pdf",
        "paper_id": "paper8_BAB",
        "data_dir": str(_SHARED_DATA / "paper8"),
    },
}


def _load_papers(source: str | None) -> list[str]:
    if source and source.startswith("http"):
        return [source]
    path = Path(source) if source else _DEFAULT_PAPERS_FILE
    if not path.exists():
        print(f"No papers file found at {path}. Pass a URL or create papers.txt")
        sys.exit(1)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _build_p2_state(shortcut_key: str) -> AgentState:
    cfg = _P2_SHORTCUTS[shortcut_key]
    data_dir = Path(cfg["data_dir"])

    manifest = json.loads((data_dir / "data_manifest.json").read_text())
    strategy_description = (data_dir / "strategy_description.txt").read_text()
    impl_details_path = str(data_dir / "implementation_details.txt")

    print(f"\n{'='*60}")
    print(f"  Quant Paper Agent — {cfg['paper_id']}")
    print(f"{'='*60}")
    print(f"  Data obtained from a previous run.")
    print(f"  Tickers   : {len(manifest.get('tickers', []))} instruments")
    print(f"  Date range: {manifest.get('date_range', ['?', '?'])[0]} to {manifest.get('date_range', ['?', '?'])[1]}")
    print(f"  Rows      : {manifest.get('row_count'):,}")
    print(f"\n  Starting Phase 2 now...")
    print(f"{'='*60}\n")

    return {
        "paper_url": cfg["paper_url"],
        "paper_id": cfg["paper_id"],
        "parsed_sections": {},
        "strategy_description": strategy_description,
        "implementation_details_path": impl_details_path,
        "skip_phase1": True,
        "openhands_data_brief": "",
        "openhands_code_brief": "",
        "data_manifest": manifest,
        "data_dir": cfg["data_dir"],
        "data_coverage_note": f"Pre-fetched: {len(manifest.get('tickers', []))} tickers.",
        "data_feasibility": "full",
        "data_checker_verdict": "PASS",
        "data_checker_feedback": "Data verified from previous run.",
        "code_checker_verdict": "",
        "code_checker_feedback": "",
        "generated_code": "",
        "lint_errors": [],
        "execution_result": "",
        "final_report": "",
        "retry_counts": {"phase1_loop": 1, "phase2_loop": 0},
        "flags": [f"[runner] data loaded from previous run: {cfg['data_dir']}"],
        "status": "RUNNING",
        "current_agent": "phase2_guiding",
    }


def run_pipeline(papers: list[str]) -> None:
    total = len(papers)
    print(f"\n{'='*60}")
    print(f"Quant Paper Agent — {total} paper(s) to process")
    print(f"{'='*60}\n")

    for i, paper_url in enumerate(papers, 1):
        print(f"[{i}/{total}] Processing: {paper_url}")

        initial_state: AgentState = {
            "paper_url": paper_url,
            "paper_id": "",
            "parsed_sections": {},
            "strategy_description": "",
            "implementation_details_path": "",
            "skip_phase1": False,
            "openhands_data_brief": "",
            "openhands_code_brief": "",
            "data_manifest": {},
            "data_dir": "",
            "data_coverage_note": "",
            "data_feasibility": "full",
            "data_checker_verdict": "",
            "data_checker_feedback": "",
            "code_checker_verdict": "",
            "code_checker_feedback": "",
            "generated_code": "",
            "lint_errors": [],
            "execution_result": "",
            "final_report": "",
            "retry_counts": {"phase1_loop": 0, "phase2_loop": 0},
            "flags": [],
            "status": "RUNNING",
            "current_agent": "",
        }

        try:
            final_state = graph.invoke(initial_state)
            _print_result(final_state, paper_url)
        except Exception as e:
            print(f"  ✗ Pipeline error: {e}")

        print()

    print(f"{'='*60}")
    print(f"Run complete. Summary: outputs/run_summary.json")
    print(f"{'='*60}\n")


def run_p2_shortcut(shortcut_key: str) -> None:
    if shortcut_key not in _P2_SHORTCUTS:
        print(f"Unknown shortcut '{shortcut_key}'. Available: {list(_P2_SHORTCUTS.keys())}")
        sys.exit(1)

    initial_state = _build_p2_state(shortcut_key)
    try:
        final_state = graph.invoke(initial_state)
        _print_result(final_state, _P2_SHORTCUTS[shortcut_key]["paper_url"])
    except Exception as e:
        print(f"  ✗ Pipeline error: {e}")

    print(f"\n{'='*60}")
    print(f"Run complete. Summary: outputs/run_summary.json")
    print(f"{'='*60}\n")


def _print_result(final_state: dict, paper_url: str) -> None:
    status = final_state.get("status", "UNKNOWN")
    paper_id = final_state.get("paper_id", paper_url)
    safe_id = paper_id.replace("/", "_")

    if status == "DONE":
        print(f"  ✓ Done — report: outputs/reports/{safe_id}.md")
    elif status == "HUMAN_REVIEW_NEEDED":
        print(f"  ⚠ Escalated — needs human review. Flags:")
        for flag in final_state.get("flags", []):
            print(f"    • {flag}")
    else:
        print(f"  ? Unexpected status: {status}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] == "--p2":
        shortcut_key = args[1] if len(args) > 1 else "paper1"
        run_p2_shortcut(shortcut_key)
    else:
        source = args[0] if args else None
        papers = _load_papers(source)
        run_pipeline(papers)
