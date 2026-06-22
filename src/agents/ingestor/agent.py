from __future__ import annotations

from src.state import AgentState
from src.tools.arxiv_fetch import fetch_paper
from src.tools.pdf_parse import parse_pdf


def run(state: AgentState) -> AgentState:
    paper_url = state["paper_url"]

    paper_id, pdf_path, title, summary = fetch_paper(paper_url)
    sections = parse_pdf(pdf_path)

    parsed_sections = {
        "title": title,
        "abstract": summary,
        "raw_text": sections.get("raw_text", ""),
    }

    return {
        **state,
        "paper_id": paper_id,
        "parsed_sections": parsed_sections,
        "flags": [],
        "retry_counts": state.get("retry_counts", {}),
        "status": "RUNNING",
    }
