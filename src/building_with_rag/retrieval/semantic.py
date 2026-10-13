"""Semantic retrieval: embed the question, run Atlas $vectorSearch, return source passages.

Diagnostic command (from the repository root, with an untracked .env):
    uv run python -m building_with_rag.retrieval.semantic "What is the punishment for murder?"
    uv run python -m building_with_rag.retrieval.semantic --check

Output is compact: statuses, counts, ids, scores and lengths. It never prints vectors, keys or
passage text. A score is a similarity ranking, not proof that a passage answers the question.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk, SemanticFilters
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import Settings, get_settings

PATTERN = "semantic"
QUERY_TIMEOUT_SECONDS = 20
NUM_CANDIDATES_MULTIPLIER = 10
NUM_CANDIDATES_FLOOR = 50
NUM_CANDIDATES_CAP = 200
READINESS_CACHE_SECONDS = 60
ACTS = ("BNS_2023", "IPC_1860")
SCORE_CAVEAT = "Scores are similarity rankings, not proof that a passage answers the question."

REASONS = {
    "no_sections": "the sections collection is missing or lacks one of the acts",
    "no_chunks": "the chunks collection is empty",
    "no_embeddings": "the embeddings collection is empty",
    "embeddings_incomplete": "chunks and embeddings counts differ",
    "embedding_settings_mismatch": "stored embeddings use a different model, version or size",
    "index_missing": f"index {schema.VECTOR_INDEX_NAME} does not exist",
    "index_incompatible": f"index {schema.VECTOR_INDEX_NAME} has an unexpected definition",
    "index_not_ready": f"index {schema.VECTOR_INDEX_NAME} is not queryable yet",
    "mongodb_error": "MongoDB could not be queried",
    "embedding_failed": "the Voyage embedding request failed",
    "embedding_rate_limited": "the Voyage rate limit was hit; wait and try again",
    "unresolved_hits": "search hits did not resolve to chunks and sections",
}

_databases: dict[tuple[str, str], Database] = {}
_ready_until: dict[tuple[str, str], float] = {}


class Unavailable(Exception):
    def __init__(self, reason: str, detail: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or REASONS.get(reason, reason)


# --- scope ------------------------------------------------------------------------------------


def effective_scope(filters: SemanticFilters | None) -> dict[str, list[str]] | None:
    """The public base scope narrowed by the caller. None means nothing can match."""
    levels = [schema.ACCESS_LEVEL_PUBLIC]
    if filters and filters.access_level is not None:
        levels = [v for v in levels if v in filters.access_level]
    if not levels:
        return None
    scope: dict[str, list[str]] = {schema.FILTER_ACCESS_LEVEL: levels}
    if filters and filters.act:
        scope[schema.FILTER_ACT] = list(filters.act)
    if filters and filters.status:
        scope[schema.FILTER_STATUS] = list(filters.status)
    return scope


def build_filter(scope: dict[str, list[str]]) -> dict[str, Any]:
    clauses = [{field: {"$in": values}} for field, values in scope.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def num_candidates(limit: int) -> int:
    wanted = min(max(limit * NUM_CANDIDATES_MULTIPLIER, NUM_CANDIDATES_FLOOR), NUM_CANDIDATES_CAP)
    return max(wanted, limit)


# --- readiness --------------------------------------------------------------------------------


def _fields(definition: dict[str, Any]) -> str:
    return json.dumps(definition.get("fields"), sort_keys=True)


def _mongo_failure(exc: PyMongoError) -> Unavailable:
    return Unavailable("mongodb_error", f"MongoDB could not be queried ({type(exc).__name__})")


def check_readiness(db: Database) -> dict[str, Any]:
    """Counts and metadata only. Raises Unavailable with a short reason if not searchable."""
    try:
        by_act = {
            row["_id"]: row["n"]
            for row in db[schema.SECTIONS].aggregate(
                [{"$group": {"_id": "$act", "n": {"$sum": 1}}}]
            )
        }
        if any(by_act.get(act, 0) == 0 for act in ACTS):
            raise Unavailable("no_sections")
        chunks = db[schema.CHUNKS].count_documents({})
        embeddings = db[schema.EMBEDDINGS].count_documents({})
        if chunks == 0:
            raise Unavailable("no_chunks")
        if embeddings == 0:
            raise Unavailable("no_embeddings")
        if embeddings != chunks:
            raise Unavailable("embeddings_incomplete")
        mismatched = db[schema.EMBEDDINGS].count_documents(
            {
                "$or": [
                    {"embedding_model": {"$ne": schema.EMBEDDING_MODEL}},
                    {"embedding_version": {"$ne": schema.EMBEDDING_VERSION}},
                    {"embedding_dimensions": {"$ne": schema.EMBEDDING_DIMENSIONS}},
                ]
            }
        )
        sample = db[schema.EMBEDDINGS].find_one({}, {schema.VECTOR_FIELD: 1}) or {}
        if mismatched or len(sample.get(schema.VECTOR_FIELD, [])) != schema.EMBEDDING_DIMENSIONS:
            raise Unavailable("embedding_settings_mismatch")
        found = list(db[schema.EMBEDDINGS].list_search_indexes(schema.VECTOR_INDEX_NAME))
    except PyMongoError as exc:
        raise _mongo_failure(exc) from None
    if not found:
        raise Unavailable("index_missing")
    index = found[0]
    have = index.get("latestDefinition") or index.get("definition") or {}
    wanted = schema.vector_index_definition()
    if index.get("type") != "vectorSearch" or _fields(have) != _fields(wanted):
        raise Unavailable("index_incompatible")
    if not index.get("queryable"):
        raise Unavailable("index_not_ready")
    return {
        "sections_by_act": by_act,
        "chunks": chunks,
        "embeddings": embeddings,
        "embedding_dimensions": schema.EMBEDDING_DIMENSIONS,
        "index": schema.VECTOR_INDEX_NAME,
        "index_queryable": True,
    }


def _database(settings: Settings) -> Database:
    key = (settings.mongodb_uri, settings.mongodb_db_name)
    if key not in _databases:
        client: MongoClient = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)
        _databases[key] = client[settings.mongodb_db_name]
    return _databases[key]


def _require(settings: Settings, *, need_voyage: bool) -> None:
    if not settings.mongodb_uri:
        raise Unavailable("missing_setting", "MONGODB_URI is not set")
    if need_voyage and not settings.voyage_api_key:
        raise Unavailable("missing_setting", "VOYAGE_API_KEY is not set")


def _ensure_ready(db: Database, settings: Settings) -> None:
    key = (settings.mongodb_uri, settings.mongodb_db_name)
    if _ready_until.get(key, 0) > time.monotonic():
        return
    check_readiness(db)
    _ready_until[key] = time.monotonic() + READINESS_CACHE_SECONDS


# --- embedding --------------------------------------------------------------------------------


def embed_query(question: str, settings: Settings) -> list[float]:
    """One bounded Voyage request with the same model settings as the document embeddings."""
    import voyageai
    from voyageai.error import RateLimitError

    client = voyageai.Client(
        api_key=settings.voyage_api_key, max_retries=0, timeout=QUERY_TIMEOUT_SECONDS
    )
    try:
        result = client.embed(
            [question],
            model=schema.EMBEDDING_MODEL,
            input_type="query",
            output_dimension=schema.EMBEDDING_DIMENSIONS,
        )
    except RateLimitError:
        raise Unavailable("embedding_rate_limited") from None
    except Exception as exc:  # noqa: BLE001 - any provider failure is "unavailable"
        raise Unavailable(
            "embedding_failed", f"the Voyage embedding request failed ({type(exc).__name__})"
        ) from None
    vectors = result.embeddings
    if len(vectors) != 1 or len(vectors[0]) != schema.EMBEDDING_DIMENSIONS:
        raise Unavailable("embedding_failed", "Voyage returned an unexpected vector size")
    return vectors[0]


# --- search -----------------------------------------------------------------------------------


def _source(chunk: dict, section: dict, source: dict | None) -> dict[str, Any]:
    def pick(key: str) -> Any:
        value = section.get(key)
        return value if value is not None else (source or {}).get(key)

    number = section.get("section_number")
    code = str(section.get("act", "")).split("_")[0]
    return {
        "section_label": f"{code} section {number}" if code and number is not None else None,
        "act_label": pick("act_label"),
        "section_number": number,
        "chapter": section.get("chapter"),
        "chapter_title": section.get("chapter_title"),
        "status": section.get("status"),
        "source_id": section.get("source_id"),
        "source_pdf": pick("source_pdf"),
        "source_sha256": pick("source_sha256"),
        "page_start": section.get("page_start"),
        "page_end": section.get("page_end"),
        "needs_review": section.get("needs_review"),
        "chunk_index": chunk.get("chunk_index"),
    }


def search(
    db: Database, vector: list[float], scope: dict[str, list[str]], limit: int
) -> tuple[list[RetrievedChunk], int, int]:
    """Returns (results, hits returned by the database, unresolved hits)."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": schema.VECTOR_INDEX_NAME,
                "path": schema.VECTOR_FIELD,
                "queryVector": vector,
                "numCandidates": num_candidates(limit),
                "limit": limit,
                "filter": build_filter(scope),
            }
        },
        {"$project": {"chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    try:
        hits = list(db[schema.EMBEDDINGS].aggregate(pipeline))
        chunk_ids = [h["chunk_id"] for h in hits]
        chunks = {c["_id"]: c for c in db[schema.CHUNKS].find({"_id": {"$in": chunk_ids}})}
        section_ids = list({c["section_id"] for c in chunks.values()})
        sections = {s["_id"]: s for s in db[schema.SECTIONS].find({"_id": {"$in": section_ids}})}
        source_ids = list({s["source_id"] for s in sections.values() if s.get("source_id")})
        sources = {s["_id"]: s for s in db[schema.SOURCES].find({"_id": {"$in": source_ids}})}
    except PyMongoError as exc:
        raise _mongo_failure(exc) from None

    results: list[RetrievedChunk] = []
    unresolved = 0
    for hit in sorted(hits, key=lambda h: (-h["score"], h["chunk_id"])):
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if not chunk or not section:
            unresolved += 1
            continue
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                section_id=section["section_id"],
                act=section["act"],
                text=chunk["text"],
                heading=section.get("heading"),
                score=float(hit["score"]),
                source=_source(chunk, section, sources.get(section.get("source_id"))),
            )
        )
    return results[:limit], len(hits), unresolved


# --- entry point ------------------------------------------------------------------------------


def retrieve(
    request: QueryRequest,
    *,
    settings: Settings | None = None,
    db: Database | None = None,
    embed: Callable[[str, Settings], list[float]] | None = None,
) -> QueryResult:
    """Semantic retrieval. Statuses: ok, no_results, unavailable."""
    settings = settings or get_settings()
    question = request.question.strip()
    scope = effective_scope(request.filters)
    trace: dict[str, Any] = {
        "query": question,
        "limit": request.limit,
        "num_candidates": num_candidates(request.limit),
        "effective_filters": scope,
        "embedding_model": schema.EMBEDDING_MODEL,
        "embedding_version": schema.EMBEDDING_VERSION,
        "index": schema.VECTOR_INDEX_NAME,
        "caller_id": settings.webui_demo_caller_id,
        "ignored_fields": [n for n in ("chapter", "required_acts") if getattr(request, n)],
    }

    if scope is None:
        trace.update(candidates_returned=0, unresolved_hits=0)
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
        vector = (embed or embed_query)(question, settings)
        results, returned, unresolved = search(db, vector, scope, request.limit)
        if returned and not results:
            raise Unavailable("unresolved_hits")
    except Unavailable as exc:
        trace["reason"] = exc.reason
        return QueryResult(
            pattern=PATTERN,
            status="unavailable",
            message=f"Semantic retrieval is unavailable: {exc.detail}.",
            trace=trace,
        )

    trace.update(candidates_returned=returned, unresolved_hits=unresolved)
    if not results:
        return QueryResult(
            pattern=PATTERN,
            status="no_results",
            message="No passages matched this question within the requested scope. "
            "This is not a service error.",
            trace=trace,
        )
    return QueryResult(
        pattern=PATTERN,
        status="ok",
        message=f"Returned {len(results)} source passage(s) ranked by vector similarity. "
        f"{SCORE_CAVEAT}",
        trace=trace,
        results=results,
    )


# --- diagnostic command -----------------------------------------------------------------------


def _print_result(result: QueryResult) -> None:
    print(f"status:  {result.status}")
    print(f"message: {result.message}")
    print(f"mode:    {result.pattern}")
    for key in ("query", "limit", "effective_filters", "candidates_returned", "unresolved_hits"):
        if key in result.trace:
            print(f"{key}: {result.trace[key]}")
    if "reason" in result.trace:
        print(f"reason: {result.trace['reason']}")
    for rank, item in enumerate(result.results, 1):
        print(
            f"{rank}. score={item.score:.4f} chunk_id={item.chunk_id} section_id={item.section_id}"
        )
        print(f"   heading={item.heading!r} act={item.act} text_chars={len(item.text)}")
        print(f"   source={json.dumps(item.source, sort_keys=True)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic retrieval diagnostic (compact output).")
    parser.add_argument("question", nargs="?", help="the question to retrieve passages for")
    parser.add_argument("--check", action="store_true", help="readiness counts only")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--act", action="append")
    parser.add_argument("--status", action="append")
    parser.add_argument("--access-level", action="append", dest="access_level")
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.check:
        try:
            _require(settings, need_voyage=True)
            report = check_readiness(_database(settings))
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
