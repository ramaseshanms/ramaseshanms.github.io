"""
Generate _articles/*.md from writing/*.md for Jekyll static HTML generation.

Each _articles/ file gets YAML front matter (title, author, date) extracted from
the source. The H1 title line, italic author/date line, and first horizontal rule
are stripped from the body — the layout template renders those from front matter.

The original writing/*.md files are left untouched (raw markdown, used by page.html
and accessible directly at /writing/name.md for AI agent / llms.txt references).
"""

import os
import re
from pathlib import Path

PORTFOLIO = Path(__file__).parent
WRITING   = PORTFOLIO / "writing"
ARTICLES  = PORTFOLIO / "_articles"

ARTICLES.mkdir(exist_ok=True)

# Known metadata for posts where the file lacks an explicit author/date line
METADATA = {
    "FMA-Net++":                        {"title": "FMA-Net++: Video Super-Resolution and Deblurring",      "date": "2025-12-16"},
    "enterprise-rag":                   {"title": "Building an Enterprise RAG System That Actually Enforces Access Control", "date": "2026-06-01"},
    "autonomous-ecu-diagnostic-agent":  {"date": "2026-06-06"},
    "hqq-turbo-weight-quantization":    {"date": "2026-03-27"},
    "kv-cache-quantization-android":    {"date": "2026-03-27"},
    "tq3-kcache-bug-hunt":              {"date": "2026-03-27"},
    "veda-arm64-porting-experiment":    {"date": "2026-04-10"},
    "vulkan-llm-runtime":               {"date": "2026-03-25"},
}

MONTH_MAP = {
    "january": "01", "february": "02", "march": "03",    "april": "04",
    "may": "05",     "june": "06",     "july": "07",     "august": "08",
    "september":"09","october": "10",  "november": "11", "december": "12",
}

def parse_author_date(line: str):
    """Extract author and date from '*Author | Month YYYY*' italic line."""
    inner = line.strip().strip("*").strip()
    if "|" in inner:
        author_part, date_part = [p.strip() for p in inner.split("|", 1)]
    else:
        return inner, None
    # Parse "Month YYYY" -> "YYYY-MM-DD"
    parts = date_part.lower().split()
    if len(parts) == 2 and parts[0] in MONTH_MAP:
        date_iso = f"{parts[1]}-{MONTH_MAP[parts[0]]}-01"
    else:
        date_iso = None
    return author_part, date_iso


def strip_preamble(lines: list[str], slug: str):
    """
    Strip H1 title, optional blank lines, optional italic author/date line,
    optional blank lines, and optional first '---' horizontal rule from content.
    Returns (title, author, body_lines).
    """
    i = 0
    title = METADATA.get(slug, {}).get("title")
    author = "M S Ramaseshan"
    date_from_line = None

    # Strip H1
    if i < len(lines) and lines[i].startswith("# "):
        if title is None:
            title = lines[i][2:].strip()
        i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1

    # Strip italic author/date line
    if i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
            a, d = parse_author_date(stripped)
            if a:
                author = a
            if d:
                date_from_line = d
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1

    # Strip first horizontal rule (---) that follows the preamble
    if i < len(lines) and lines[i].strip() == "---":
        i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1

    return title, author, date_from_line, lines[i:]


def make_front_matter(slug, title, author, date_override):
    # METADATA has the exact date; prefer it over the month-only date parsed from the author line
    date = METADATA.get(slug, {}).get("date") or date_override or "2026-01-01"
    safe_title = title.replace('"', '\\"') if title else slug
    return f'---\ntitle: "{safe_title}"\nauthor: "{author}"\ndate: {date}\n---\n\n'


for md_file in sorted(WRITING.glob("*.md")):
    slug = md_file.stem
    lines = md_file.read_text(encoding="utf-8").splitlines(keepends=True)

    title, author, date_from_line, body_lines = strip_preamble(lines, slug)

    if title is None:
        title = slug.replace("-", " ").title()

    front = make_front_matter(slug, title, author, date_from_line)
    body  = "".join(body_lines)

    out = ARTICLES / md_file.name
    out.write_text(front + body, encoding="utf-8")
    print(f"  {md_file.name}  ->  _articles/{md_file.name}  [{title[:55]}]")

print(f"\nDone — {len(list(ARTICLES.glob('*.md')))} files in _articles/")
