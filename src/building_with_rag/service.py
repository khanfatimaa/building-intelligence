from collections.abc import Iterator

from .contracts import QueryRequest, QueryResult
from .registry import PATTERNS

# Retrieval modes that still hand an `ok` or `no_results` result to generation.
OPEN_PATTERNS = ("semantic", "hybrid", "hybrid-reranked")


def run_pattern_events(
    request: QueryRequest, *, answer_generation: bool = False, stream: bool = False
) -> Iterator[tuple[str, object]]:
    """The one operation behind /v1/query and /v1/chat/completions.

    Yields ("text", str) display events and ends with ("final", QueryResult). One retrieval mode
    (semantic, hybrid, hybrid-reranked, structured, decomposition or hyde) runs once; for
    requests that ask for a generated answer (`answer_generation` on) that same result goes to
    one generation/validation operation, whose text events and final result describe the same run. `stream` only chooses the provider's streaming call for answer
    writing.
    """
    if request.pattern in PATTERNS and (not request.generate_answer or answer_generation):
        if request.pattern == "hybrid-reranked":
            from .retrieval.rerank import retrieve
        elif request.pattern == "hybrid":
            from .retrieval.hybrid import retrieve
        elif request.pattern == "structured":
            from .retrieval.structured import retrieve
        elif request.pattern == "decomposition":
            from .retrieval.decomposition import retrieve
        elif request.pattern == "hyde":
            from .retrieval.hyde import retrieve
        else:
            from .retrieval.semantic import retrieve

        result = retrieve(request)
        # Structured, decomposition and hyde generate only for `ok`; their other statuses are
        # shown as the result message (no complete comparison without evidence for every step).
        generating = ("ok", "no_results") if request.pattern in OPEN_PATTERNS else ("ok",)
        if request.generate_answer and result.status in generating:
            from .generation.answer import generate_events

            if request.pattern == "decomposition":
                from .retrieval.decomposition import step_summary

                yield ("text", f"Subquestions and their evidence:\n{step_summary(result)}\n\n")
            yield from generate_events(result, request.question, stream=stream)
            return
    else:
        result = QueryResult(
            pattern=request.pattern,
            status="not_implemented",
            message=f"The '{request.pattern}' mode is not implemented yet. No retrieval was run.",
            trace={"caller_id": request.caller_id},
        )
    yield ("text", result.message)
    yield ("final", result)


def run_pattern(request: QueryRequest, *, answer_generation: bool = False) -> QueryResult:
    """Single dispatch path for /v1/query and /v1/chat/completions (final result only)."""
    final = None
    for kind, value in run_pattern_events(request, answer_generation=answer_generation):
        if kind == "final":
            final = value
    assert isinstance(final, QueryResult)
    return final
