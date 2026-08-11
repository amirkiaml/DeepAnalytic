"""
section_parser.py

Splits each SEP article into (section_title, section_text) pairs using
the article's own Table of Contents as ground truth for which sections
exist, then locates where each heading actually starts in the body text.

Why this matters for retrieval quality: chunking within section
boundaries means a chunk never straddles two unrelated topics, and
attaching the section title to chunk metadata gives the reranker/LLM
useful context (e.g. "Nicomachean Ethics — Section 3: The Doctrine of
the Mean") instead of an anonymous blob of text. This is the mechanism
behind the "wrong-sense rate" idea from the portfolio plan — sections
give you a cheap, real signal for disambiguating polysemous terms.

Fallback chain, in order:
  1. toc_match     — every heading in the TOC was located in the body text
  2. toc_partial    — some (not all) TOC headings were located
  3. regex_fallback — no usable TOC; scan the body text directly for
                       numbered-heading-shaped lines
  4. failed         — nothing locatable; whole article becomes one section

parsed_how is recorded per-article so you can audit parsing quality
across the corpus after a run (see the summary ingest.py prints).
"""

import re


def parse_toc(toc_text: str) -> dict:
    """
    Parses a TOC string into {"1": {"title": ..., "subsections": {...}}}.

    (Bug fix from the original notebook code: it reassigned the `toc`
    parameter to {} before reading it, so every call crashed with
    AttributeError: 'dict' object has no attribute 'splitlines'.)
    """
    sections = {}
    current = None

    for line in toc_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if m := re.match(r'^(\d+)\.\s+(.+)$', line):
            current = m.group(1)
            # "section_title" here means the heading text as written in the
            # TOC (e.g. "Historical Outline") — NOT the article's own Title
            # column. Named explicitly to avoid confusion with row["Title"]
            # in ingest.py, which is the actual article title.
            sections[current] = {"section_title": m.group(2).strip(), "subsections": {}}

        elif m := re.match(r'^(\d+\.\d+)\s+(.+)$', line):
            if current is not None:
                title = m.group(2).strip().replace('\u201c', '').replace('\u201d', '')
                sections[current]["subsections"][m.group(1)] = title

    return sections


def locate_headings(text: str, expected_numbers: list) -> dict:
    """
    Finds the character offset where each numbered heading actually
    starts in the article body.

    Matches on the HEADING NUMBER only (e.g. a line starting "3." or
    "3.1"), not the title text — SEP body headings don't always match
    TOC title wording verbatim (capitalization, italics markup,
    trailing punctuation differences), so matching on the number is
    far more robust than trying to match title strings.

    Returns {heading_number: start_char_index} — numbers that weren't
    found are simply absent, so len(result) vs len(expected_numbers)
    tells the caller toc_match vs toc_partial.

    NOTE: this regex is a reasonable starting heuristic for SEP's usual
    formatting, but SEP articles span decades of authors and some
    variation is likely. After your first real run, spot-check a
    handful of articles with check_chunks.py and tighten this pattern
    if you see systematic misses.
    """
    found = {}
    pattern = re.compile(r'^(\d+(?:\.\d+)?)\.?\s+\S', re.MULTILINE)

    for match in pattern.finditer(text):
        number = match.group(1)
        if number in expected_numbers and number not in found:
            found[number] = match.start()

    return found


def _regex_fallback(body_text: str):
    """No usable TOC — scan the body directly for heading-shaped lines."""
    pattern = re.compile(r'^(\d+(?:\.\d+)?)\.?\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(body_text))

    if not matches:
        return [("Full Text", body_text)], "failed"

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body_text)
        sections.append((m.group(2).strip(), body_text[start:end]))

    return sections, "regex_fallback"


def split_into_sections(toc_text: str, body_text: str):
    """
    Main entry point.

    Returns:
        sections: list of (section_title, section_text) tuples, in
                   document order, covering the whole article
        parsed_how: 'toc_match' | 'toc_partial' | 'regex_fallback' | 'failed'
    """
    toc = parse_toc(toc_text) if toc_text and toc_text.strip() else {}

    if not toc:
        return _regex_fallback(body_text)

    expected_numbers = list(toc.keys())
    positions = locate_headings(body_text, expected_numbers)

    if not positions:
        return _regex_fallback(body_text)

    parsed_how = "toc_match" if len(positions) == len(expected_numbers) else "toc_partial"

    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    sections = []
    for i, (number, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(body_text)
        section_title = toc.get(number, {}).get("section_title", f"Section {number}")
        sections.append((section_title, body_text[start:end]))

    return sections, parsed_how
