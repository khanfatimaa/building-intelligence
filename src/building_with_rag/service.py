from .contracts import QueryRequest, QueryResult


def run_pattern(request: QueryRequest) -> QueryResult:
    """Single dispatch path for /v1/query and /v1/chat/completions.

    Semantic retrieval runs for retrieval-only requests. Requests that ask for a generated answer
    (chat) keep the placeholder until answer generation exists.
    """
    if request.pattern == "semantic" and not request.generate_answer:
        from .retrieval.semantic import retrieve

        return retrieve(request)
    return QueryResult(
        pattern=request.pattern,
        status="not_implemented",
        message=f"The '{request.pattern}' mode is not implemented yet. No retrieval was run.",
        trace={"caller_id": request.caller_id},
    )
