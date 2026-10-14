"""Hybrid re-ranking: a bounded set of hybrid candidates re-ordered by one re-ranking model.

Diagnostic command (from the repository root, with an untracked .env):
    uv run python -m building_with_rag.retrieval.rerank "What is culpable homicide?"
    uv run python -m building_with_rag.retrieval.rerank --check

Output is compact: statuses, counts, ids, ranks, scores and lengths. It never prints keys, vectors
or passage text. A re-rank score is one model's relevance ranking, not proof that a passage
answers the question.
"""

import argparse
import sys
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError
from pymongo.database import Database

from building_with_rag.contracts import (
    OmittedCandidate,
    QueryRequest,
    QueryResult,
    RetrievedChunk,
    SemanticFilters,
)
from building_with_rag.retrieval import hybrid
from building_with_rag.retrieval.semantic import (
    Unavailable,
    _database,
    _require,
    check_readiness,
    effective_scope,
)
from building_with_rag.settings import Settings, get_settings

PATTERN = "hybrid-reranked"
MAX_HYBRID_LIMIT = 20  # QueryRequest.limit maximum, which bounds one hybrid retrieval
RERANK_CAVEAT = (
    "Results are hybrid candidates re-ordered by one re-ranking model. Scores are relevance "
    "rankings, not proof that a passage answers the question."
)
SETTING_NAMES = {
    "rerank_api_base_url": "RERANK_API_BASE_URL",
    "rerank_api_key": "RERANK_API_KEY",
    "rerank_model_name": "RERANK_MODEL_NAME",
}

# Scores one document list against the question; returns one score per document, in input order.
RerankCall = Callable[[str, list[str], Settings], list[float]]


class RerankFailure(Exception):
    def __init__(self, cause: str):
        super().__init__(cause)
        self.cause = cause


# --- settings ---------------------------------------------------------------------------------


def missing_settings(settings: Settings) -> list[str]:
    return [
        name for attr, name in SETTING_NAMES.items() if not str(getattr(settings, attr)).strip()
    ]


def invalid_settings(settings: Settings) -> list[str]:
    """Names of limit settings that break 1 <= RETURN <= SEND <= CANDIDATE <= hybrid maximum."""
    candidate = settings.rerank_candidate_limit
    send = settings.rerank_send_limit
    ret = settings.rerank_return_limit
    bad: set[str] = set()
    if not 1 <= candidate <= MAX_HYBRID_LIMIT:
        bad.add("RERANK_CANDIDATE_LIMIT")
    if send < 1 or send > candidate:
        bad.add("RERANK_SEND_LIMIT")
    if ret < 1 or ret > send:
        bad.add("RERANK_RETURN_LIMIT")
    if settings.rerank_request_timeout_seconds < 1:
        bad.add("RERANK_REQUEST_TIMEOUT_SECONDS")
    return sorted(bad)


# --- provider call ----------------------------------------------------------------------------


def call_rerank(question: str, documents: list[str], settings: Settings) -> list[float]:
    """One Voyage-style rerank request. Returns one score per document, in input order.

    Raises RerankFailure with a coarse cause only: no key, URL, body, response or exception text.
    """
    url = f"{settings.rerank_api_base_url.strip().rstrip('/')}/rerank"
    body = {
        "model": settings.rerank_model_name,
        "query": question,
        "documents": documents,
        "top_k": len(documents),
    }
    try:
        response = httpx.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            timeout=settings.rerank_request_timeout_seconds,
        )
    except httpx.TimeoutException:
        raise RerankFailure("timeout") from None
    except httpx.HTTPError:
        raise RerankFailure("connection_error") from None
    if not 200 <= response.status_code < 300:
        raise RerankFailure(f"http_{response.status_code}")
    try:
        data = response.json()["data"]
    except (ValueError, KeyError, TypeError):
        raise RerankFailure("malformed_response") from None
    return _scores_by_index(data, len(documents))


def _scores_by_index(data: Any, count: int) -> list[float]:
    """Every sent document must have exactly one numeric score; anything else is a failure."""
    if not isinstance(data, list):
        raise RerankFailure("malformed_response")
    scores: dict[int, float] = {}
    for entry in data:
        index = entry.get("index") if isinstance(entry, dict) else None
        score = entry.get("relevance_score") if isinstance(entry, dict) else None
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < count
            or index in scores
            or not isinstance(score, int | float)
            or isinstance(score, bool)
        ):
            raise RerankFailure("malformed_response")
        scores[index] = float(score)
    if len(scores) != count:
        raise RerankFailure("malformed_response")
    return [scores[i] for i in range(count)]


# --- entry point ------------------------------------------------------------------------------


def _unavailable(trace: dict[str, Any], reason: str, detail: str) -> QueryResult:
    trace["reason"] = reason
    return QueryResult(
        pattern=PATTERN,
        status="unavailable",
        message=f"Hybrid re-ranking is unavailable: {detail}. No re-ranking was performed and "
        "no passages are returned.",
        trace=trace,
    )


def retrieve(
    request: QueryRequest,
    *,
    settings: Settings | None = None,
    db: Database | None = None,
    embed: Callable[[str, Settings], list[float]] | None = None,
    rerank: RerankCall | None = None,
) -> QueryResult:
    """Hybrid candidates, one re-rank, then the kept passages. Statuses: ok, no_results, unavailable.

    Never returns the un-re-ranked hybrid list: if re-ranking cannot run, the result is unavailable.
    """
    settings = settings or get_settings()
    question = request.question.strip()

    # An empty scope matches nothing; hybrid answers it without any outside call.
    if effective_scope(request.filters) is None:
        result = hybrid.retrieve(request, settings=settings, db=db, embed=embed)
        return result.model_copy(update={"pattern": PATTERN})

    trace: dict[str, Any] = {
        "query": question,
        "limit": request.limit,
        "caller_id": settings.webui_demo_caller_id,
    }
    missing = missing_settings(settings)
    if missing:
        return _unavailable(trace, "missing_setting", f"{', '.join(missing)} is not set")
    invalid = invalid_settings(settings)
    if invalid:
        return _unavailable(
            trace,
            "invalid_rerank_settings",
            f"invalid re-rank limit setting(s): {', '.join(invalid)}",
        )

    candidate_limit = settings.rerank_candidate_limit
    send_limit = settings.rerank_send_limit
    return_limit = settings.rerank_return_limit
    final_count = min(request.limit, return_limit)

    found = hybrid.retrieve(
        request.model_copy(update={"limit": candidate_limit}),
        settings=settings,
        db=db,
        embed=embed,
    )
    if found.status != "ok":
        return found.model_copy(update={"pattern": PATTERN})

    taken = found.results
    sent = taken[:send_limit]
    trace = {**found.trace, "limit": request.limit}
    trace["rerank"] = {
        "model": settings.rerank_model_name,
        "candidates_taken": len(taken),
        "sent": len(sent),
        "candidate_limit": candidate_limit,
        "send_limit": send_limit,
        "return_limit": return_limit,
        "final_count": final_count,
        "timeout_seconds": settings.rerank_request_timeout_seconds,
        "before": [
            {"chunk_id": c.chunk_id, "fused_rank": c.fused_rank, "fused_score": c.fused_score}
            for c in sent
        ],
    }
    try:
        scores = (rerank or call_rerank)(question, [c.text for c in sent], settings)
        if len(scores) != len(sent):
            raise RerankFailure("malformed_response")
    except RerankFailure as exc:
        trace["rerank_cause"] = exc.cause
        return _unavailable(trace, "rerank_failed", f"the re-ranking request failed ({exc.cause})")

    order = sorted(range(len(sent)), key=lambda i: (-scores[i], sent[i].fused_rank or i))
    trace["rerank"]["after"] = [
        {"chunk_id": sent[i].chunk_id, "rerank_rank": rank, "rerank_score": scores[i]}
        for rank, i in enumerate(order, 1)
    ]
    results: list[RetrievedChunk] = [
        sent[i].model_copy(
            update={"score": scores[i], "rerank_score": scores[i], "rerank_rank": rank}
        )
        for rank, i in enumerate(order[:final_count], 1)
    ]
    omitted: list[OmittedCandidate] = list(found.omitted_candidates)  # below_limit, from hybrid
    omitted += [
        OmittedCandidate(chunk_id=c.chunk_id, omitted_reason="below_send_limit")
        for c in taken[send_limit:]
    ]
    omitted += [
        OmittedCandidate(chunk_id=sent[i].chunk_id, omitted_reason="below_return_limit")
        for i in order[final_count:]
    ]
    trace["rerank"].update(
        returned=len(results),
        omitted=len(taken) - len(results),  # cut after hybrid took its candidates
        below_limit=len(found.omitted_candidates),  # cut by hybrid's candidate limit, not taken
        selected=[c.chunk_id for c in results],
    )
    return QueryResult(
        pattern=PATTERN,
        status="ok",
        message=f"Returned {len(results)} source passage(s) ranked by re-rank score. "
        f"{RERANK_CAVEAT}",
        trace=trace,
        results=results,
        omitted_candidates=omitted,
    )


# --- diagnostic command -----------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _print_result(result: QueryResult) -> None:
    print(f"status:  {result.status}")
    print(f"message: {result.message}")
    print(f"mode:    {result.pattern}")
    for key in ("query", "limit", "effective_filters"):
        if key in result.trace:
            print(f"{key}: {result.trace[key]}")
    if "reason" in result.trace:
        print(f"reason: {result.trace['reason']}")
    info = result.trace.get("rerank")
    if info:
        counts = {k: v for k, v in info.items() if k not in ("before", "after", "selected")}
        print(f"rerank: {counts}")
        print("before (fused order):")
        for row in info["before"]:
            print(f"  {row['fused_rank']}. fused={_fmt(row['fused_score'])} {row['chunk_id']}")
        if "after" in info:
            fused_rank = {r["chunk_id"]: r["fused_rank"] for r in info["before"]}
            print("after (re-rank order):")
            for row in info["after"]:
                print(
                    f"  {row['rerank_rank']}. rerank={_fmt(row['rerank_score'])} "
                    f"{row['chunk_id']} (fused rank {fused_rank[row['chunk_id']]})"
                )
    if result.results:
        print("selected evidence:")
    for item in result.results:
        source = item.source or {}
        print(f"  {item.rerank_rank}. {item.chunk_id} section_id={item.section_id} act={item.act}")
        print(
            f"     heading={item.heading!r} pages={source.get('page_start')}-"
            f"{source.get('page_end')} text_chars={len(item.text)}"
        )
    for cut in result.omitted_candidates:
        print(f"omitted: {cut.chunk_id} ({cut.omitted_reason})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid re-ranking diagnostic (compact output).")
    parser.add_argument("question", nargs="?", help="the question to retrieve passages for")
    parser.add_argument("--check", action="store_true", help="settings and readiness only")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--act", action="append")
    parser.add_argument("--status", action="append")
    parser.add_argument("--access-level", action="append", dest="access_level")
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.check:
        missing, invalid = missing_settings(settings), invalid_settings(settings)
        if missing or invalid:
            print("status: unavailable")
            if missing:
                print(f"missing: {', '.join(missing)}")
            if invalid:
                print(f"invalid: {', '.join(invalid)}")
            return 1
        try:
            _require(settings, need_voyage=True)
            db = _database(settings)
            report = {**check_readiness(db), **hybrid.check_keyword_readiness(db)}
        except Unavailable as exc:
            print(f"status: unavailable\nreason: {exc.reason}\nmessage: {exc.detail}")
            return 1
        print("status: ready")
        print(f"rerank_model: {settings.rerank_model_name}")
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
