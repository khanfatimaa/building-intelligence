"""Turn the supplied BNS and IPC PDFs into section-level JSONL records.

Run with: uv run python -m building_with_rag.ingest

Needs no database, network, credentials, or .env. Text is kept as it appears in the PDF; only
repeated headers, footers, page numbers, and spacing are normalised. Anything uncertain is kept
and flagged with ``needs_review`` instead of being repaired silently.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from importlib import metadata
from itertools import pairwise
from pathlib import Path

PARSER = "pypdf"
ROOT = Path(__file__).resolve().parents[2]

ACTS = {
    "bns": {
        "act": "BNS_2023",
        "act_label": "Bharatiya Nyaya Sanhita, 2023",
        "status": "in_force",
        "pdf": "BNS_2023_bare_act.pdf",
        "out": "bns_sections.jsonl",
        "expected_max": 358,  # sanity check for the report only; never written to a record
        "heading_index": True,  # headings are printed above the number; the index lists them
        "restarts": True,  # the PDF holds an index, a table and the act: numbering restarts
    },
    "ipc": {
        "act": "IPC_1860",
        "act_label": "Indian Penal Code, 1860",
        "status": "repealed",
        "pdf": "IPC_1860_bare_act.pdf",
        "out": "ipc_sections.jsonl",
        "expected_max": 511,
        "heading_index": False,
        "restarts": False,
    },
}

FIELDS = [
    "section_id",
    "act",
    "act_label",
    "status",
    "chapter",
    "chapter_title",
    "section_number",
    "heading",
    "text",
    "source_pdf",
    "source_sha256",
    "parser",
    "parser_version",
    "source_status_version",
    "needs_review",
]

CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLC]+[A-Z]?)\b[\s.:—–-]*(.*)$")
# A section starts "115.(1) text", "3. Heading.--text" or "7*[5. Heading.--text" (the "7*["
# is an amendment marker). The text after the number must not start with a digit ("3.5").
START_RE = re.compile(r"^(?:\d+\*+\[|\[)?(\d+)([A-Z]{0,2})\.\s*(?!\d)(\S.*)$")
# The IPC PDF prints "4." alone, then the heading, then the full "4. Heading.--text" line.
LONE_RE = re.compile(r"^(?:\d+\*+\[|\[)?(\d+)([A-Z]{0,2})\.$")
# Amendment footnotes also look like "3. Subs. by Act ..."; they are not sections.
FOOTNOTE_RE = re.compile(
    r"^(\d+[A-Z]{0,2}\.\s+(Subs|Ins|Omitted|Rep|Added|Renumbered|Re-numbered|Substituted|"
    r"Inserted|Repealed|Vide|Ibid|The words|The figures|The Act|Clause|Clauses|Sub-section)\b"
    r"|Ss?\. \d+[A-Z]? (was|were) )"
)
END_RE = re.compile(
    r"^(INDEX|CORRESPONDING SECTION|CORRESPONDENCE|COMPARATIVE|COMPARISON|CONCORDANCE|TABLE|APPENDIX|"
    r"THE SCHEDULE|STATEMENT OF OBJECTS)\b"
)
# Web-page chrome and rule lines in the supplied PDFs, not law text.
JUNK_RE = re.compile(
    r"^(Note: For Reference only\.?|HomePage|Chapters and Sections HomePage|[—–_-]{3,})$"
)
GROUP_TITLE_RE = re.compile(r"^Of [^.]{3,80}$")
HEADING_SPLIT_RE = re.compile(r"^(?P<heading>[^(\s].{1,200}?)\s*\.\s*(?:--+|[—–-])\s*(?P<rest>.*)$")
PAGE_NUMBER_RE = re.compile(r"^[\s—–-]*(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?[\s—–-]*$", re.IGNORECASE)
RANGE_RE = re.compile(r"^(\d+)[A-Z]{0,2} to (\d+)([A-Z]{0,2})\.")
NEW_RUN_MIN_LAST = 50  # a "1." only restarts the numbering after a long run (index -> act)
EDGE_LINES = 3
MIN_PAGE_CHARS = 20


@dataclass
class Line:
    page: int
    text: str


@dataclass
class Draft:
    number: int
    chapter: str | None
    chapter_title: str | None
    heading: str | None
    lines: list[Line]
    lone: bool = False  # started by a lone "N." line (IPC layout)
    heading_in_text: bool = False  # heading lines come before a full line repeating them
    review: set[str] = field(default_factory=set)


# --- provenance and hashing -------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_provenance(path: Path) -> dict[str, dict[str, str]]:
    """Read the file table from PROVENANCE.md: file name -> sha256, retrieved, url."""
    rows: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or ".pdf`" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 5:
            rows[cells[0].strip("`")] = {
                "url": cells[2],
                "retrieved": cells[3],
                "sha256": cells[4].strip("`").lower(),
            }
    return rows


def status_version(entry: dict[str, str] | None) -> str | None:
    if not entry or not entry.get("retrieved") or not entry.get("url"):
        return None
    return f"Retrieved {entry['retrieved']} (UTC) from {entry['url']}"


# --- text extraction and cleaning -------------------------------------------------------


def extract_pages(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def _shape(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip().lower()


def clean_pages(pages: list[str]) -> tuple[list[Line], list[int]]:
    """Drop repeated headers, footers, and page numbers; return lines and text-less pages."""
    page_lines = [[ln.strip() for ln in text.splitlines() if ln.strip()] for text in pages]
    edge_counts: Counter[str] = Counter()
    for lines in page_lines:
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:] if len(lines) > EDGE_LINES else lines
        edge_counts.update({_shape(ln) for ln in edges})
    threshold = max(3, int(len(pages) * 0.3))

    kept: list[Line] = []
    empty_pages: list[int] = []
    for number, lines in enumerate(page_lines, start=1):
        total = len(lines)
        body: list[str] = []
        for index, ln in enumerate(lines):
            at_edge = index < EDGE_LINES or index >= total - EDGE_LINES
            if JUNK_RE.match(ln):
                continue
            if at_edge and (PAGE_NUMBER_RE.match(ln) or edge_counts[_shape(ln)] >= threshold):
                continue
            body.append(re.sub(r"\s+", " ", ln))
        if sum(len(ln) for ln in body) < MIN_PAGE_CHARS:
            empty_pages.append(number)
        kept.extend(Line(number, ln) for ln in body)
    return kept, empty_pages


# --- section scanning -------------------------------------------------------------------


def split_heading(text: str) -> tuple[str | None, str]:
    """Split 'Heading.--text' into (heading, text); (None, text) when there is no heading."""
    match = HEADING_SPLIT_RE.match(text)
    if match:
        return match.group("heading").strip(), match.group("rest").strip()
    return None, text


def _read_chapter_title(lines: list[Line], index: int, inline: str) -> tuple[str | None, int]:
    """Chapter title from the chapter line itself or the lines after it; (title, lines used)."""
    inline = inline.strip(" .:—–-")
    if inline:
        return inline, 0
    parts: list[str] = []
    used = 0
    for ln in lines[index + 1 : index + 4]:
        text = ln.text
        if START_RE.match(text) or CHAPTER_RE.match(text) or len(text) > 120:
            break
        if parts and not text.isupper():
            break
        parts.append(text)
        used += 1
    return (" ".join(parts) or None), used


def scan_runs(lines: list[Line], restarts: bool = True) -> tuple[list[list[Draft]], list[str]]:
    """Walk the lines and collect runs of consecutive sections (1, 2, 3, ...).

    A PDF can hold several runs, such as an index followed by the act itself; the caller keeps
    the one with the most text. With ``restarts`` off, only one run is ever started (a later
    "1." is a footnote, not a new act). Lettered sections (e.g. 153A) and repealed ranges (e.g.
    "161 to 165A") are not collected: they end the preceding section and are listed as omitted.
    """
    runs: list[list[Draft]] = []
    run: list[Draft] | None = None
    current: Draft | None = None
    last = 0
    chapter: str | None = None
    chapter_title: str | None = None
    chapter_seen = False
    omitted: list[str] = []

    def omit(label: str) -> None:
        if label not in omitted:
            omitted.append(label)

    index = 0
    while index < len(lines):
        line = lines[index]
        text = line.text
        chapter_match = CHAPTER_RE.match(text)
        if chapter_match:
            current = None
            chapter = chapter_match.group(1)
            chapter_title, used = _read_chapter_title(lines, index, chapter_match.group(2))
            chapter_seen = True
            index += 1 + used
            continue
        if END_RE.match(text):
            current = None
            index += 1
            continue
        range_match = RANGE_RE.match(text)
        if range_match and run is not None:
            omit(f"{range_match.group(1)} to {range_match.group(2)}{range_match.group(3)}")
            last = max(last, int(range_match.group(2)))
            current = None
            index += 1
            continue

        lone = LONE_RE.match(text)
        section = lone or (None if FOOTNOTE_RE.match(text) else START_RE.match(text))
        if section:
            number, suffix = int(section.group(1)), section.group(2)
            rest = None if lone else section.group(3)
            if suffix:
                if run is not None and number == last:
                    omit(f"{number}{suffix}")
                    current = None
                    index += 1
                    continue
            else:
                gap = False
                if run is not None and number == last + 1:
                    pass
                elif (
                    number == 1
                    and (chapter_seen or run is None)
                    and (run is None or (restarts and last >= NEW_RUN_MIN_LAST))
                ):
                    run = []
                    runs.append(run)
                    last = 0
                elif (
                    run is not None
                    and last + 2 <= number <= last + 5
                    and (lone or split_heading(rest or "")[0])
                ):
                    gap = True
                else:
                    number = -1
                if number > 0:
                    draft = Draft(
                        number,
                        chapter,
                        chapter_title,
                        None,
                        [] if rest is None else [Line(line.page, rest)],
                        lone=bool(lone),
                    )
                    if gap:
                        draft.review.add(f"gap_before_{number}")
                    if chapter is None:
                        draft.review.add("no_chapter")
                    elif chapter_title is None:
                        draft.review.add("no_chapter_title")
                    assert run is not None
                    run.append(draft)
                    current = draft
                    last = number
                    chapter_seen = False
                    index += 1
                    continue

        if current is not None:
            current.lines.append(line)
        index += 1
    return runs, omitted


def split_lone_start(draft: Draft) -> None:
    """Sort out a section that began with a lone "N." line (the IPC layout).

    The lines after it are the heading, then the full "N. Heading.--text" line. Repealed
    sections have only a bracketed title and no full line; those keep all their lines as text.
    """
    if not draft.lone:
        return
    full = re.compile(rf"^(?:\d+\*+\[|\[)?{draft.number}(?:\.\s*|\s+)(?!\d)(\S.*)$")
    for j, ln in enumerate(draft.lines[:6]):
        match = full.match(ln.text)
        if match:
            heading = re.sub(r"\s+", " ", " ".join(x.text for x in draft.lines[:j])).strip()
            draft.heading = heading or None
            draft.heading_in_text = bool(heading)
            draft.lines = [Line(ln.page, match.group(1))] + draft.lines[j + 1 :]
            return
    draft.review.add("no_full_section_line")


def _key(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()


def _heading_tail_size(lines: list[Line], heading: str) -> int:
    """Number of trailing lines that spell the heading (allowing small wording differences)."""
    want = _key(heading)
    best, best_ratio = 0, 0.0
    for size in range(1, 7):
        ratio = SequenceMatcher(None, _key(" ".join(ln.text for ln in lines[-size:])), want).ratio()
        if ratio > best_ratio:
            best, best_ratio = size, ratio
    return best if best_ratio >= 0.85 else 0


def _drop_group_titles(lines: list[Line]) -> None:
    """Drop trailing group titles such as "Of hurt"; they head a group of sections."""
    while lines and GROUP_TITLE_RE.match(lines[-1].text):
        lines.pop()


def apply_index_headings(body: list[Draft], index_run: list[Draft]) -> None:
    """Take BNS headings from the index; remove them from the end of the previous section.

    The BNS PDF prints a heading above its section number, so it lands at the end of the
    previous section's lines.
    """
    headings = {}
    for d in index_run:
        entry = list(d.lines)
        _drop_group_titles(entry)
        headings[d.number] = _normalise(entry)
    previous: Draft | None = None
    for draft in body:
        heading = headings.get(draft.number)
        if heading and draft.heading is None:
            draft.heading = heading
            if previous is not None and previous.chapter == draft.chapter:
                size = _heading_tail_size(previous.lines, heading)
                if size:
                    del previous.lines[-size:]
                    _drop_group_titles(previous.lines)
                else:
                    draft.review.add("heading_not_found_before_section")
        previous = draft


# --- records ----------------------------------------------------------------------------


def _normalise(lines: list[Line]) -> str:
    return re.sub(r"\s+", " ", " ".join(ln.text for ln in lines)).strip()


def _remove_prefix(text: str, heading: str) -> str | None:
    """Remove a leading heading (ignoring spacing) and the dash after it; None if absent."""
    i = 0
    for ch in heading:
        if ch.isspace():
            continue
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != ch:
            return None
        i += 1
    return re.sub(r"^\s*(?:--+|[—–-]+)?\s*", "", text[i:])


def _dedupe(text: str) -> str:
    """Remove a clear exact repeat: the whole text printed twice, joined by one space."""
    half = len(text) // 2
    if (
        len(text) > 40
        and len(text) % 2 == 1
        and text[half] == " "
        and text[:half] == text[half + 1 :]
    ):
        return text[:half]
    return text


def _has_repeated_line(lines: list[Line]) -> bool:
    return any(a.text == b.text and len(a.text) >= 40 for a, b in pairwise(lines))


def build_records(
    key: str,
    runs: list[list[Draft]],
    empty_pages: list[int],
    sha: str,
    parser_version: str,
    provenance_status: str | None,
    hash_matches: bool,
) -> list[dict]:
    meta = ACTS[key]
    if not runs:
        return []
    body = max(runs, key=lambda r: sum(len(_normalise(d.lines)) for d in r))
    if meta["heading_index"] and runs[0] is not body and len(runs[0]) >= NEW_RUN_MIN_LAST:
        apply_index_headings(body, runs[0])
    records = []
    for draft in body:
        split_lone_start(draft)
        _drop_group_titles(draft.lines)
        text = _normalise(draft.lines)
        if draft.heading_in_text and draft.heading:
            stripped = _remove_prefix(text, draft.heading)
            if stripped is not None:
                text = stripped
            else:
                # The heading above differs in wording from the full line; trust the full line.
                heading, rest = split_heading(text)
                if heading:
                    draft.heading, text = heading, rest
                else:
                    draft.review.add("heading_not_found_in_text")
        elif draft.heading is None:
            draft.heading, text = split_heading(text)
        text = _dedupe(text)
        review = set(draft.review)
        if draft.heading is None:
            review.add("no_heading")
        if any(FOOTNOTE_RE.match(ln.text) for ln in draft.lines):
            review.add("contains_footnote_text")
        if _has_repeated_line(draft.lines):
            review.add("possible_duplicate_text")
        pages = [ln.page for ln in draft.lines]
        if pages and any(p in empty_pages for p in range(min(pages), max(pages) + 1)):
            review.add("spans_page_without_text")
        if not text:
            continue  # text must never be empty; the number shows up as missing in the report
        if not hash_matches:
            review.add("pdf_hash_differs_from_provenance")
        record = {
            "section_id": f"{key}:{draft.number}",
            "act": meta["act"],
            "act_label": meta["act_label"],
            "status": meta["status"],
            "chapter": draft.chapter,
            "chapter_title": draft.chapter_title,
            "section_number": draft.number,
            "heading": draft.heading.rstrip(".").strip() if draft.heading else None,
            "text": text,
            "source_pdf": meta["pdf"],
            "source_sha256": sha,
            "parser": PARSER,
            "parser_version": parser_version,
            "source_status_version": provenance_status,
            "needs_review": bool(review) or provenance_status is None,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
        }
        record["_review_reasons"] = sorted(review)
        records.append(record)
    return records


# --- safe writes ------------------------------------------------------------------------


def is_output_current(path: Path, sha: str) -> bool:
    """True when the file exists, is valid JSONL, and every record came from this PDF hash."""
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return False
        ids: set[str] = set()
        for line in lines:
            record = json.loads(line)
            if any(name not in record for name in FIELDS):
                return False
            if record["source_sha256"] != sha or not record["text"]:
                return False
            ids.add(record["section_id"])
        return len(ids) == len(lines)
    except (OSError, ValueError, TypeError):
        return False


def write_atomic(path: Path, records: list[dict]) -> None:
    """Write all records to a temp file in the same folder, then replace the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                clean = {k: v for k, v in record.items() if not k.startswith("_")}
                handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# --- command ----------------------------------------------------------------------------


def ingest_act(key: str, raw_dir: Path, out_dir: Path, provenance: dict) -> int:
    meta = ACTS[key]
    pdf = raw_dir / meta["pdf"]
    out = out_dir / meta["out"]
    if not pdf.is_file():
        print(f"[{key}] missing input: {pdf}", file=sys.stderr)
        return 2

    sha = sha256_file(pdf)
    entry = provenance.get(meta["pdf"])
    expected = entry["sha256"] if entry else None
    hash_matches = expected == sha
    if not hash_matches:
        print(
            f"[{key}] WARNING: SHA-256 of {pdf.name} is {sha}, PROVENANCE.md has {expected}. "
            "All records will be flagged needs_review.",
            file=sys.stderr,
        )

    if is_output_current(out, sha):
        print(f"[{key}] {out.name} is valid and matches the PDF hash; left untouched.")
        return 0

    try:
        pages = extract_pages(pdf)
    except Exception as error:  # noqa: BLE001 - any parser failure means an unreadable PDF
        print(f"[{key}] cannot read {pdf.name}: {error}", file=sys.stderr)
        return 2
    lines, empty_pages = clean_pages(pages)
    runs, omitted = scan_runs(lines, restarts=meta["restarts"])
    records = build_records(
        key,
        runs,
        empty_pages,
        sha,
        metadata.version(PARSER),
        status_version(entry),
        hash_matches,
    )
    if not records:
        print(f"[{key}] no sections found in {pdf.name}; nothing written.", file=sys.stderr)
        return 2

    write_atomic(out, records)

    flagged = [r for r in records if r["needs_review"]]
    reasons = Counter(reason for r in records for reason in r["_review_reasons"])
    numbers = [r["section_number"] for r in records]
    missing = sorted(set(range(1, meta["expected_max"] + 1)) - set(numbers))
    print(f"[{key}] wrote {len(records)} records to {out} ({len(flagged)} need review).")
    print(f"[{key}] pages: {len(pages)}; pages without usable text: {empty_pages or 'none'}")
    print(f"[{key}] section runs found: {len(runs)} (largest kept)")
    print(f"[{key}] omitted lettered sections: {', '.join(omitted) or 'none'}")
    print(f"[{key}] missing section numbers (1..{meta['expected_max']}): {missing or 'none'}")
    if numbers[-1] > meta["expected_max"]:
        print(f"[{key}] WARNING: last section number {numbers[-1]} is above {meta['expected_max']}")
    if reasons:
        print(f"[{key}] review reasons: {dict(reasons)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m building_with_rag.ingest",
        description="Extract section records from the BNS and IPC PDFs into data/processed/.",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args(argv)

    prov_path = args.raw_dir / "PROVENANCE.md"
    if not prov_path.is_file():
        print(f"missing input: {prov_path}", file=sys.stderr)
        return 2
    provenance = read_provenance(prov_path)
    codes = [ingest_act(key, args.raw_dir, args.out_dir, provenance) for key in ACTS]
    return max(codes)


if __name__ == "__main__":
    sys.exit(main())
