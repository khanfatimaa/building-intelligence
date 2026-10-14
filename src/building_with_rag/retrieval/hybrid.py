"""Hybrid retrieval: a semantic route and a keyword route fused with Reciprocal Rank Fusion.

Diagnostic command (from the repository root, with an untracked .env):
    uv run python -m building_with_rag.retrieval.hybrid "What is culpable homicide?"
    uv run python -m building_with_rag.retrieval.hybrid --check

Output is compact: statuses, counts, ids, ranks, scores and lengths. It never prints vectors, keys
or passage text. A fused score is a ranking, not proof that a passage answers the question.
"""

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError
from pymongo.database import Database
from pymongo.errors import PyMongoError

from building_with_rag.contracts import (
    OmittedCandidate,
    QueryRequest,
    QueryResult,
    RetrievedChunk,
    SemanticFilters,
)
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval.semantic import (
    SCORE_CAVEAT,
    Unavailable,
    _database,
    _ensure_ready,
    _mongo_failure,
    _require,
    check_readiness,
    effective_scope,
    embed_query,
    resolve_hits,
    vector_hits,
)
from building_with_rag.settings import Settings, get_settings

PATTERN = "hybrid"
ROUTE_CANDIDATES_FLOOR = 20
ROUTE_CANDIDATES_MULTIPLIER = 4
ROUTE_CANDIDATES_CAP = 50
FUSION_NAME = "reciprocal_rank_fusion"
RRF_K = 60
ROUTE_WEIGHTS = {"semantic": 1.0, "keyword": 1.0}
READINESS_CACHE_SECONDS = 60
KEYWORD_REASONS = {
    "keyword_index_missing": f"keyword index {schema.KEYWORD_INDEX_NAME} does not exist; run "
    "`uv run python -m building_with_rag.ingestion.keyword_index`",
    "keyword_index_incompatible": f"keyword index {schema.KEYWORD_INDEX_NAME} has an unexpected "
    "definition",
    "keyword_index_not_ready": f"keyword index {schema.KEYWORD_INDEX_NAME} is not queryable yet",
    "keyword_search_failed": "the keyword search could not be run",
    "unresolved_hits": "search hits did not resolve to chunks and sections",
}
HYBRID_CAVEAT = (
    "Results fuse a semantic ranking and a keyword ranking (reciprocal rank fusion). "
    + SCORE_CAVEAT
)

_keyword_ready_until: dict[tuple[str, str], float] = {}


def _keyword_unavailable(reason: str) -> Unavailable:
    return Unavailable(reason, KEYWORD_REASONS[reason])


def route_candidates(limit: int) -> int:
    wanted = min(max(limit * ROUTE_CANDIDATES_MULTIPLIER, ROUTE_CANDIDATES_FLOOR), ROUTE_CANDIDATES_CAP)
    return max(wanted, limit)


# --- keyword route ----------------------------------------------------------------------------


def keyword_stage(question: str, scope: dict[str, list[str]], count: int) -> list[dict[str, Any]]:
    """The $search pipeline. Scope clauses come only from validated lists, inside the stage."""
    return [
        {
            "$search": {
                "index": schema.KEYWORD_INDEX_NAME,
                "compound": {
                    "must": [{"text": {"query": question, "path": schema.KEYWORD_FIELD}}],
                    "filter": [
                        {"in": {"path": field, "value": values}} for field, values in scope.items()
                    ],
                },
            }
        },
        {"$limit": count},
        {"$project": {"chunk_id": 1, "score": {"$meta": "searchScore"}}},
    ]


def keyword_hits(
    db: Database, question: str, scope: dict[str, list[str]], count: int
) -> list[dict[str, Any]]:
    try:
        return list(db[schema.CHUNKS].aggregate(keyword_stage(question, scope, count)))
    except PyMongoError as exc:
        raise Unavailable(
            "keyword_search_failed", f"the keyword search failed ({type(exc).__name__})"
        ) from None


def check_keyword_readiness(db: Database) -> dict[str, Any]:
    """Raises Unavailable if the keyword index is missing, incompatible or not queryable."""
    try:
        found = list(db[schema.CHUNKS].list_search_indexes(schema.KEYWORD_INDEX_NAME))
    except PyMongoError as exc:
        raise _mongo_failure(exc) from None
    if not found:
        raise _keyword_unavailable("keyword_index_missing")
    if not schema.keyword_index_matches(found[0]):
        raise _keyword_unavailable("keyword_index_incompatible")
    if not found[0].get("queryable"):
        raise _keyword_unavailable("keyword_index_not_ready")
    return {
        "keyword_index": schema.KEYWORD_INDEX_NAME,
        "keyword_field": schema.KEYWORD_FIELD,
        "keyword_analyzer": schema.KEYWORD_ANALYZER,
        "keyword_index_queryable": True,
    }


def _ensure_keyword_ready(db: Database, settings: Settings) -> None:
    key = (settings.mongodb_uri, settings.mongodb_db_name)
    if _keyword_ready_until.get(key, 0) > time.monotonic():
        return
    check_keyword_readiness(db)
    _keyword_ready_until[key] = time.monotonic() + READINESS_CACHE_SECONDS


# --- fusion -----------------------------------------------------------------------------------


def fuse(
    semantic: list[RetrievedChunk], keyword: list[RetrievedChunk], limit: int
) -> tuple[list[RetrievedChunk], list[OmittedCandidate]]:
    """Reciprocal Rank Fusion over two route lists, each already in its own route order.

    fused_score = sum over routes containing the chunk of weight / (RRF_K + rank_in_route), with
    1-based ranks. Raw route scores are kept for inspection and never enter the fusion.
    """
    merged: dict[str, dict[str, Any]] = {}
    for route, passages in (("semantic", semantic), ("keyword", keyword)):
        for rank, passage in enumerate(passages, 1):
            entry = merged.setdefault(passage.chunk_id, {"chunk": passage, "fused": 0.0})
            entry["fused"] += ROUTE_WEIGHTS[route] / (RRF_K + rank)
            entry[f"{route}_rank"] = rank
            entry[f"{route}_score"] = passage.score
    ordered = sorted(merged.values(), key=lambda e: (-e["fused"], e["chunk"].chunk_id))

    fused: list[RetrievedChunk] = []
    omitted: list[OmittedCandidate] = []
    for position, entry in enumerate(ordered, 1):
        if position > limit:
            omitted.append(
                OmittedCandidate(chunk_id=entry["chunk"].chunk_id, omitted_reason="below_limit")
            )
            continue
        fused.append(
            entry["chunk"].model_copy(
                update={
                    "score": entry["fused"],
                    "fused_score": entry["fused"],
                    "fused_rank": position,
                    "semantic_score": entry.get("semantic_score"),
                    "semantic_rank": entry.get("semantic_rank"),
                    "keyword_score": entry.get("keyword_score"),
                    "keyword_rank": entry.get("keyword_rank"),
                }
            )
        )
    return fused, omitted


def route_label(item: RetrievedChunk) -> str:
    if item.semantic_rank is not None and item.keyword_rank is not None:
        return "both"
    return "semantic" if item.semantic_rank is not None else "keyword"


def _route_results(
    db: Database, hits: list[dict[str, Any]]
) -> tuple[list[RetrievedChunk], int]:
    ordered = sorted(hits, key=lambda h: (-h["score"], h["chunk_id"]))
    results = [chunk for _, chunk in resolve_hits(db, ordered) if chunk is not None]
    return results, len(hits) - len(results)


# --- entry point ------------------------------------------------------------------------------


def retrieve(
    request: QueryRequest,
    *,
    settings: Settings | None = None,
    db: Database | None = None,
    embed: Callable[[str, Settings], list[float]] | None = None,
) -> QueryResult:
    """Hybrid retrieval. Statuses: ok, no_results, unavailable."""
    settings = settings or get_settings()
    question = request.question.strip()
    scope = effective_scope(request.filters)
    count = route_candidates(request.limit)
    trace: dict[str, Any] = {
        "query": question,
        "limit": request.limit,
        "effective_filters": scope,
        "embedding_model": schema.EMBEDDING_MODEL,
        "embedding_version": schema.EMBEDDING_VERSION,
        "index": schema.VECTOR_INDEX_NAME,
        "keyword_index": schema.KEYWORD_INDEX_NAME,
        "keyword_field": schema.KEYWORD_FIELD,
        "keyword_analyzer": schema.KEYWORD_ANALYZER,
        "candidates_per_route": count,
        "fusion": {"name": FUSION_NAME, "k": RRF_K, "weights": ROUTE_WEIGHTS},
        "caller_id": settings.webui_demo_caller_id,
        "ignored_fields": [n for n in ("chapter", "required_acts") if getattr(request, n)],
    }

    if scope is None:
        trace.update(
            semantic_candidates_returned=0,
            keyword_candidates_returned=0,
            route_overlap=0,
            semantic_only=0,
            keyword_only=0,
            unresolved_hits=0,
        )
        return QueryResult(
            pattern=PATTERN,
            status="no_results",
            message="No passages can match: the requested filters leave an empty scope. "
            "This is not a service error.",
            trace=trace,
        )
    try:
        _require(settings, need_voyage=embed is None)
        db = db if db is not None else _database(settings)
        _ensure_ready(db, settings)
        _ensure_keyword_ready(db, settings)
        vector = (embed or embed_query)(question, settings)
        semantic_hits = vector_hits(db, vector, scope, count)
        keyword_found = keyword_hits(db, question, scope, count)
        semantic, semantic_unresolved = _route_results(db, semantic_hits)
        keyword, keyword_unresolved = _route_results(db, keyword_found)
        if (semantic_hits and not semantic) or (keyword_found and not keyword):
            raise Unavailable("unresolved_hits")
    except Unavailable as exc:
        trace["reason"] = exc.reason
        return QueryResult(
            pattern=PATTERN,
            status="unavailable",
            message=f"Hybrid retrieval is unavailable: {exc.detail}.",
            trace=trace,
        )

    results, omitted = fuse(semantic, keyword, request.limit)
    both = {p.chunk_id for p in semantic} & {p.chunk_id for p in keyword}
    trace.update(
        semantic_candidates_returned=len(semantic_hits),
        keyword_candidates_returned=len(keyword_found),
        route_overlap=len(both),
        semantic_only=len(semantic) - len(both),
        keyword_only=len(keyword) - len(both),
        unresolved_hits=semantic_unresolved + keyword_unresolved,
        semantic_unresolved=semantic_unresolved,
        keyword_unresolved=keyword_unresolved,
    )
    if not results:
        return QueryResult(
            pattern=PATTERN,
            status="no_results",
            message="No passages matched this question within the requested scope, by either "
            "route. This is not a service error.",
            trace=trace,
        )
    return QueryResult(
        pattern=PATTERN,
        status="ok",
        message=f"Returned {len(results)} source passage(s) ranked by fused score. {HYBRID_CAVEAT}",
        trace=trace,
        results=results,
        omitted_candidates=omitted,
    )


# --- diagnostic command -----------------------------------------------------------------------


def _fmt(value: float | None, digits: int | None = None) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if digits is not None else str(value)


def _print_result(result: QueryResult) -> None:
    print(f"status:  {result.status}")
    print(f"message: {result.message}")
    print(f"mode:    {result.pattern}")
    for key in (
        "query",
        "limit",
        "effective_filters",
        "candidates_per_route",
        "semantic_candidates_returned",
        "keyword_candidates_returned",
        "route_overlap",
        "semantic_only",
        "keyword_only",
        "unresolved_hits",
        "fusion",
    ):
        if key in result.trace:
            print(f"{key}: {result.trace[key]}")
    if "reason" in result.trace:
        print(f"reason: {result.trace['reason']}")
    for item in result.results:
        print(
            f"{item.fused_rank}. fused={_fmt(item.fused_score, 4)} route={route_label(item)} "
            f"semantic(rank={_fmt(item.semantic_rank)}, score={_fmt(item.semantic_score, 4)}) "
            f"keyword(rank={_fmt(item.keyword_rank)}, score={_fmt(item.keyword_score, 4)})"
        )
        print(f"   chunk_id={item.chunk_id} section_id={item.section_id} act={item.act}")
        source = item.source or {}
        print(
            f"   heading={item.heading!r} pages={source.get('page_start')}-"
            f"{source.get('page_end')} text_chars={len(item.text)}"
        )
    if result.omitted_candidates:
        print(f"omitted_below_limit: {[c.chunk_id for c in result.omitted_candidates]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid retrieval diagnostic (compact output).")
    parser.add_argument("question", nargs="?", help="the question to retrieve passages for")
    parser.add_argument("--check", action="store_true", help="readiness only")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--act", action="append")
    parser.add_argument("--status", action="append")
    parser.add_argument("--access-level", action="append", dest="access_level")
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.check:
        try:
            _require(settings, need_voyage=True)
            db = _database(settings)
            report = {**check_readiness(db), **check_keyword_readiness(db)}
        except Unavailable as exc:
            print(f"status: unavailable\nreason: {exc.reason}\nmessage: {exc.detail}")
            return 1
        print("status: ready")
        for key, value in report.items():
            print(f"{key}: {value}")
        return 0
    if not args.question:
        parser.error("a question is required unless --check is given")
    try:
        filters = SemanticFilters(act=args.act, status=args.status, access_level=args.access_level)
        request = QueryRequest(
            question=args.question,
            pattern=PATTERN,
            caller_id=settings.webui_demo_caller_id,
            filters=filters,
            limit=args.limit,
        )
    except ValidationError as exc:
        print(f"Invalid input: {[e['loc'] for e in exc.errors()]}")
        return 2
    result = retrieve(request, settings=settings)
    _print_result(result)
    return 1 if result.status == "unavailable" else 0


if __name__ == "__main__":
    sys.exit(main())
