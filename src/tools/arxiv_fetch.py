from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import arxiv


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

def _extract_arxiv_id(url_or_id: str) -> str:
    match = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", url_or_id)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract arXiv ID from: {url_or_id}")


def _fetch_arxiv(url_or_id: str, cache_dir: str) -> tuple[str, str, str, str]:
    paper_id = _extract_arxiv_id(url_or_id)
    cache_path = Path(cache_dir) / f"{paper_id.replace('/', '_')}.pdf"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_id])
    results = list(client.results(search))
    if not results:
        raise ValueError(f"No arXiv paper found for ID: {paper_id}")

    paper = results[0]
    if not cache_path.exists():
        urllib.request.urlretrieve(paper.pdf_url, str(cache_path))

    return paper_id, str(cache_path), paper.title, paper.summary


# ---------------------------------------------------------------------------
# ScienceDirect
# ---------------------------------------------------------------------------

def _unpaywall_pdf_url(doi: str) -> str | None:
    """Query Unpaywall API for an open-access PDF URL for a given DOI."""
    try:
        encoded = urllib.parse.quote(doi, safe="")
        api_url = f"https://api.unpaywall.org/v2/{encoded}?email=ai.models.personaluse@gmail.com"
        req = urllib.request.Request(api_url, headers={"User-Agent": "NineMastsBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("is_oa") and data.get("best_oa_location"):
            return data["best_oa_location"].get("url_for_pdf")
    except Exception:
        pass
    return None


def _fetch_sciencedirect(url: str, cache_dir: str) -> tuple[str, str, str, str]:
    pii_match = re.search(r"/pii/([A-Za-z0-9]+)", url)
    if not pii_match:
        raise ValueError(f"Cannot extract PII from ScienceDirect URL: {url}")
    pii = pii_match.group(1).upper()
    paper_id = f"sd_{pii}"

    cache_path = Path(cache_dir) / f"{paper_id}.pdf"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Scrape abstract page for DOI + metadata (public even for paywalled papers)
    title = pii
    summary = f"ScienceDirect paper: {pii}"
    doi: str | None = None

    try:
        abstract_url = f"https://www.sciencedirect.com/science/article/abs/pii/{pii}"
        req = urllib.request.Request(
            abstract_url,
            headers={"User-Agent": "Mozilla/5.0 (research bot; ai.models.personaluse@gmail.com)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        def _meta(name: str, text: str) -> str | None:
            # Handles both attribute orderings
            for pat in [
                rf'<meta[^>]+name="{name}"[^>]+content="([^"]+)"',
                rf'<meta[^>]+content="([^"]+)"[^>]+name="{name}"',
            ]:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            return None

        doi = _meta("citation_doi", html)
        t = _meta("citation_title", html)
        if t:
            title = t
        a = _meta("citation_abstract", html)
        if a:
            summary = a
    except Exception:
        pass  # metadata scraping is best-effort; proceed with what we have

    # Fast path: user placed PDF in cache manually
    if cache_path.exists():
        return paper_id, str(cache_path), title, summary

    # Try Unpaywall for open-access PDF
    if doi:
        pdf_url = _unpaywall_pdf_url(doi)
        if pdf_url:
            urllib.request.urlretrieve(pdf_url, str(cache_path))
            return paper_id, str(cache_path), title, summary

    raise ValueError(
        f"PDF not available for ScienceDirect paper {pii}.\n"
        f"The paper may be paywalled or Unpaywall has no open-access copy.\n"
        f"To run the pipeline manually, place the PDF at:\n  {cache_path}"
    )


# ---------------------------------------------------------------------------
# Local PDF
# ---------------------------------------------------------------------------

def _fetch_local_pdf(file_path: str, cache_dir: str = "/tmp/arxiv_cache") -> tuple[str, str, str, str]:
    import shutil

    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Local PDF not found: {file_path}")

    paper_id = re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem)[:60]

    # Copy into cache dir so strategy_planner can find it at /tmp/arxiv_cache/{paper_id}.pdf
    cache_path = Path(cache_dir) / f"{paper_id}.pdf"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        shutil.copy2(str(path), str(cache_path))

    title = paper_id
    summary = f"Local PDF: {path.name}"

    try:
        import pdfplumber
        with pdfplumber.open(str(cache_path)) as pdf:
            first_text = pdf.pages[0].extract_text() or ""
            lines = [l.strip() for l in first_text.splitlines() if l.strip()]
            if lines:
                title = lines[0][:200]
            combined = " ".join((pdf.pages[i].extract_text() or "") for i in range(min(3, len(pdf.pages))))
            abs_m = re.search(
                r"(?i)\babstract\b[:\s]+(.{100,1500}?)(?=\n\s*\n|\Z|\bintroduction\b|\bkeywords\b)",
                combined, re.DOTALL,
            )
            if abs_m:
                summary = abs_m.group(1).strip()
            else:
                summary = first_text[:600]
    except Exception:
        pass

    return paper_id, str(cache_path), title, summary


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_paper(url_or_id: str, cache_dir: str = "/tmp/arxiv_cache") -> tuple[str, str, str, str]:
    """
    Returns (paper_id, pdf_path, title, summary).

    Accepts:
    - arXiv URLs / IDs  (e.g. "2412.12458v1", "https://arxiv.org/pdf/2412.12458v1")
    - ScienceDirect URLs  (e.g. "https://www.sciencedirect.com/science/article/abs/pii/S1544612325000200")
    - Local PDF paths  (e.g. "/Users/aarav/Desktop/paper.pdf" or "file:///Users/...")
    """
    s = url_or_id.strip()

    if s.startswith("file://"):
        return _fetch_local_pdf(s[7:])

    if s.startswith("/") and s.lower().endswith(".pdf"):
        return _fetch_local_pdf(s)

    if "sciencedirect.com" in s:
        return _fetch_sciencedirect(s, cache_dir)

    return _fetch_arxiv(s, cache_dir)
