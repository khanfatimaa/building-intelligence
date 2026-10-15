"""Structured exact retrieval: one read-only lookup of a BNS or IPC section by its identifiers.

The rule-based classifier (`signals.py`) is the only reader of the question. Only its validated
act and section number (plus an optional validated chapter, status filter and the public access
level) build the MongoDB predicate; question text never reaches the database. This module makes
no write call and no arbitrary query.

Diagnostic command (from the repository root, with an untracked .env):
    uv run python -m building_with_rag.retrieval.structured "What does BNS section 103 say?"

Output is compact: statuses, ids, reasons and lengths. It never prints passage text, keys or URIs.
A found section is source text with its stored status; it is not a claim of current applicability.
"""

import argparse
import sys
from typing import Any

from pydantic import ValidationError
from pymongo.database import Database
from pymongo.errors import PyMongoError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk, SemanticFilters
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.signals import CLASSIFIER_NAME, classify, validate_chapter
from building_with_rag.settings import Settings, get_settings

PATTERN = "structured"
ACT_BY_CODE = {"BNS": "BNS_2023", "IPC": "IPC_1860"}
EXAMPLE = "Try: 'BNS section 103' or 'IPC section 302'."

CLARIFICATIONS = {
    "missing_act": "A section number was found but no act. Say BNS or IPC.",
    "missing_section": "An act was found but no section number.",
    "ambiguous_act": "Both BNS and IPC were named. Ask about one act at a time.",
    "multiple_sections": "More than one section or a range was named. Ask for one section.",
    "lettered_section_unsupported": "Lettered sections (for example 498A) are not supported.",
    "section_out_of_range": "The section number must be between 1 and 999.",
    "invalid_chapter": "The chapter must be a Roman numeral such as VI.",
}


def _result(status: str, message: str, trace: dict[str, Any], results=None) -> QueryResult:
    return QueryResult(
        pattern=PATTERN, status=status, message=message, trace=trace, results=results or []
    )


def _predicate(act: str, number: int, chapter: str | None, filters: SemanticFilters | None):
    """The only query ever sent: validated values in a fixed shape."""
    predicate: dict[str, Any] = {
        "act": act,
        "section_number": number,
        "access_level": schema.ACCESS_LEVEL_PUBLIC,
    }
    if filters and filters.status:
        predicate["status"] = {"$in": list(filters.status)}
    if chapter:
        predicate["chapter"] = chapter
    return predicate


def _scope_excludes(act: str, filters: SemanticFilters | None) -> bool:
    if semantic.effective_scope(filters) is None:
        return True
    return bool(filters and filters.act and act not in filters.act)


def _lookup(db: Database, predicate: dict[str, Any]):
    """Read-only: one find on sections, then chunks and the source by id."""
    try:
        sections = list(db[schema.SECTIONS].find(predicate))
        if not sections:
            return None, [], None
        section = sections[0]
        chunks = sorted(
            db[schema.CHUNKS].find({"section_id": section["section_id"]}),
            key=lambda c: c.get("chunk_index", 0),
        )
        sources = list(db[schema.SOURCES].find({"_id": section.get("source_id")}))
    except PyMongoError:
        raise semantic.Unavailable("database_error", "MongoDB could not be queried") from None
    return section, chunks, (sources[0] if sources else None)


def retrieve(
    request: QueryRequest, *, settings: Settings | None = None, db: Database | None = None
) -> QueryResult:
    """Statuses: ok, not_found, clarification_needed, recommendation, unavailable."""
    settings = settings or get_settings()
    outcome = classify(request.question, request.chapter)
    signals = outcome.signals
    trace: dict[str, Any] = {
        "caller_id": settings.webui_demo_caller_id,
        "ignored_fields": [n for n in ("required_acts",) if getattr(request, n)],
        "classifier": {
            "name": CLASSIFIER_NAME,
            "signals": signals.model_dump() if signals else None,
            "reason": outcome.reason,
            "rule": outcome.rule,
        },
        "predicate": None,
        "database_called": False,
    }
    if outcome.reason:
        trace["reason"] = outcome.reason

    if outcome.kind == "clarification":
        return _result(
            "clarification_needed", f"{CLARIFICATIONS[outcome.reason]} {EXAMPLE}", trace
        )
    if outcome.kind == "recommendation":
        what = (
            f"This looks like a '{signals.intent}' request, which structured mode does not run."
            if signals
            else "This is not a request for one named section."
        )
        return _result(
            "recommendation",
            f"{what} For a meaning-based question choose the semantic or hybrid mode; "
            f"{EXAMPLE} No search was run.",
            trace,
        )

    assert signals is not None and signals.act and signals.section_number
    act = ACT_BY_CODE[signals.act]
    number = int(signals.section_number)
    label = f"{signals.act} section {number}"
    not_found = (
        f"{label} is not in the loaded corpus. Some sections (for example lettered ones) are not "
        "loaded; this does not say the section does not exist in law."
    )
    if _scope_excludes(act, request.filters):
        trace["reason"] = "scope_excludes_request"
        return _result("not_found", not_found, trace)

    predicate = _predicate(act, number, validate_chapter(request.chapter), request.filters)
    trace["predicate"] = predicate
    try:
        if db is None:
            if not settings.mongodb_uri:
                raise semantic.Unavailable("missing_setting", "MONGODB_URI is not set")
            db = semantic._database(settings)
        trace["database_called"] = True
        section, chunks, source = _lookup(db, predicate)
    except semantic.Unavailable as exc:
        trace["reason"] = exc.reason
        if exc.reason == "missing_setting":
            trace["database_called"] = False
        return _result(
            "unavailable", f"Structured retrieval is unavailable: {exc.detail}.", trace
        )

    if section is None:
        trace["reason"] = "no_matching_section"
        return _result("not_found", not_found, trace)
    trace["section_id"] = section["section_id"]
    trace["chunks_total"] = len(chunks)
    if not chunks:
        trace["reason"] = "section_without_chunks"
        trace["chunks_returned"] = 0
        return _result(
            "unavailable",
            "Structured retrieval is unavailable: the section has no chunk records.",
            trace,
        )

    results = []
    for chunk in chunks[: request.limit]:
        source_meta = semantic._source(chunk, section, source)
        source_meta["source_status_version"] = section.get("source_status_version") or (
            source or {}
        ).get("source_status_version")
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                section_id=section["section_id"],
                act=section["act"],
                text=chunk["text"],
                heading=section.get("heading"),
                source=source_meta,
            )
        )
    trace["chunks_returned"] = len(results)
    return _result(
        "ok",
        f"Found {label}: returned {len(results)} of {len(chunks)} stored passage(s) by exact "
        f"lookup. Status is shown as stored ({section.get('status')}); this is source text, not a "
        "statement of current legal applicability.",
        trace,
        results,
    )


# --- diagnostic command -----------------------------------------------------------------------


def _print_result(result: QueryResult) -> None:
    trace = result.trace
    print(f"status:  {result.status}")
    print(f"message: {result.message}")
    print(f"mode:    {result.pattern}")
    print(f"classifier: {trace['classifier']}")
    print(f"predicate: {trace['predicate']}")
    print(f"database_called: {trace['database_called']}")
    if "reason" in trace:
        print(f"reason: {trace['reason']}")
    for item in result.results:
        pages = (item.source or {}).get("page_start"), (item.source or {}).get("page_end")
        print(f"- chunk_id={item.chunk_id} section_id={item.section_id} act={item.act}")
        print(
            f"  status={(item.source or {}).get('status')} heading={item.heading!r} "
            f"pages={pages} text_chars={len(item.text)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Structured exact lookup diagnostic.")
    parser.add_argument("question")
    parser.add_argument("--chapter")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        request = QueryRequest(
            question=args.question,
            pattern=PATTERN,
            caller_id=settings.webui_demo_caller_id,
            limit=args.limit,
            chapter=args.chapter,
        )
    except ValidationError as exc:
        print(f"Invalid input: {[e['loc'] for e in exc.errors()]}")
        return 2
    result = retrieve(request, settings=settings)
    _print_result(result)
    return 1 if result.status == "unavailable" else 0


if __name__ == "__main__":
    sys.exit(main())
