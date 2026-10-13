from .contracts import QueryRequest, QueryResult


def run_pattern(request: QueryRequest) -> QueryResult:
    """Single dispatch path for /v1/query and /v1/chat/completions."""
    return QueryResult(
        pattern=request.pattern,
        status="not_implemented",
        message=f"The '{request.pattern}' mode is not implemented yet. No retrieval was run.",
        trace={"caller_id": request.caller_id},
    )
