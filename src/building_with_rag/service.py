from collections.abc import Iterator

from .contracts import QueryRequest, QueryResult


def run_pattern_events(
    request: QueryRequest, *, answer_generation: bool = False, stream: bool = False
) -> Iterator[tuple[str, object]]:
    """The one operation behind /v1/query and /v1/chat/completions.

    Yields ("text", str) display events and ends with ("final", QueryResult). Semantic retrieval
    runs once; for requests that ask for a generated answer (`answer_generation` on) that same
    result goes to one generation/validation operation, whose text events and final result
    describe the same run. `stream` only chooses the provider's streaming call for answer writing.
    """
    if request.pattern == "semantic" and (not request.generate_answer or answer_generation):
        from .retrieval.semantic import retrieve

        result = retrieve(request)
        if request.generate_answer and result.status in ("ok", "no_results"):
            from .generation.answer import generate_events

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
