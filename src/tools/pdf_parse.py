from __future__ import annotations

import re
import pdfplumber


# Section headers we look for (case-insensitive)
_SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "data": r"\bdata\b|\bdataset\b|\bdata description\b",
    "methodology": r"\bmethodology\b|\bmethod\b|\bapproach\b|\bmodel\b|\bstrategy\b",
    "results": r"\bresults?\b|\bempirical results?\b|\bfindings\b|\bperformance\b",
}


def parse_pdf(pdf_path: str) -> dict[str, str]:
    """
    Returns dict with keys: abstract, data, methodology, results, raw_text.
    Sections are best-effort — raw_text always contains the full document.
    """
    pages_text: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                continue
            midpoint = page.width / 2
            left_words = [w for w in words if w["x0"] < midpoint]
            right_words = [w for w in words if w["x0"] >= midpoint]

            def _words_to_text(word_list: list[dict]) -> str:
                rows: dict[int, list[dict]] = {}
                for w in word_list:
                    bucket = round(w["top"] / 4) * 4
                    rows.setdefault(bucket, []).append(w)
                lines = [
                    " ".join(w["text"] for w in sorted(row_words, key=lambda w: w["x0"]))
                    for _, row_words in sorted(rows.items())
                ]
                return "\n".join(lines)

            left_text = _words_to_text(left_words)
            right_text = _words_to_text(right_words)
            pages_text.append(left_text + "\n" + right_text)

    raw_text = "\n\n".join(pages_text)
    sections = _split_sections(raw_text)
    sections["raw_text"] = raw_text
    return sections


def _split_sections(text: str) -> dict[str, str]:
    lines = text.split("\n")
    section_hits: dict[str, int] = {}  # section_name → line index of header

    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if len(stripped) > 60:
            continue  # headers are short
        for name, pattern in _SECTION_PATTERNS.items():
            if re.search(pattern, stripped) and name not in section_hits:
                section_hits[name] = i

    # Sort by line index so we can slice between headers
    ordered = sorted(section_hits.items(), key=lambda x: x[1])
    sections: dict[str, str] = {}

    for idx, (name, start_line) in enumerate(ordered):
        end_line = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(lines)
        sections[name] = "\n".join(lines[start_line:end_line]).strip()

    # Fill any missing sections with empty string
    for name in _SECTION_PATTERNS:
        sections.setdefault(name, "")

    return sections
