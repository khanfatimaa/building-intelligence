"""Story 4.2: hybrid re-ranking bounds, order, failure outcomes and shared dispatch. Offline."""

import httpx
import pytest

from building_with_rag import service
from building_with_rag.contracts import (
    OmittedCandidate,
    QueryRequest,
    QueryResult,
    RetrievedChunk,
    SemanticFilters,
)
from building_with_rag.generation import answer
from building_with_rag.retrieval import hybrid, rerank, semantic
from building_with_rag.settings import Settings


def settings(**over) -> Settings:
    base = {
        "mongodb_uri": "mongodb://x",
        "voyage_api_key": "k",
        "rerank_api_key": "rk",
        "rerank_candidate_limit": 6,
        "rerank_send_limit": 4,
        "rerank_return_limit": 3,
    }
    return Settings(_env_file=None, **{**base, **over})


def chunk(n: int) -> RetrievedChunk:
    score = 1 / (60 + n)
    return RetrievedChunk(
        chunk_id=f"bns:{n}#0",
        section_id=f"bns:{n}",
        act="BNS_2023",
        text=f"Passage {n}.",
        score=score,
        fused_score=score,
        fused_rank=n,
        semantic_rank=n,
        semantic_score=0.5,
    )


def fake_hybrid(monkeypatch, count=7, cut=1):
    """Hybrid returns `count` candidates plus `cut` ids it already cut with below_limit."""
    calls = []

    def retrieve(request, **kwargs):
        calls.append(request.limit)
        return QueryResult(
            pattern="hybrid",
            status="ok",
            message="m",
            trace={"route_overlap": 0},
            results=[chunk(n) for n in range(1, count + 1)],
            omitted_candidates=[
                OmittedCandidate(chunk_id=f"bns:{100 + i}#0", omitted_reason="below_limit")
                for i in range(cut)
            ],
        )

    monkeypatch.setattr(hybrid, "retrieve", retrieve)
    return calls


def reverse_scores(question, documents, s):
    return [float(i) for i in range(len(documents))]  # last sent document scores highest


def no_call(*args, **kwargs):
    pytest.fail("must not be called")


def request(**over) -> QueryRequest:
    return QueryRequest(question=" murder ", pattern="hybrid-reranked", **over)


def test_bounds_order_and_omitted_reasons(monkeypatch):
    calls = fake_hybrid(monkeypatch)
    result = rerank.retrieve(request(limit=5), settings=settings(), rerank=reverse_scores)

    assert calls == [6]  # hybrid asked for RERANK_CANDIDATE_LIMIT, not the request limit
    assert result.pattern == "hybrid-reranked" and result.status == "ok"
    # sent 1..4 (5, 6, 7 are beyond the send limit), reversed to 4,3,2,1, final = min(5, 3)
    assert [c.chunk_id for c in result.results] == ["bns:4#0", "bns:3#0", "bns:2#0"]
    assert [c.rerank_rank for c in result.results] == [1, 2, 3]
    assert all(c.score == c.rerank_score for c in result.results)
    assert result.results[0].fused_rank == 4 and result.results[0].semantic_rank == 4
    reasons = {c.chunk_id: c.omitted_reason for c in result.omitted_candidates}
    assert len(reasons) == len(result.omitted_candidates)  # each candidate once
    assert reasons == {
        "bns:100#0": "below_limit",
        "bns:5#0": "below_send_limit",
        "bns:6#0": "below_send_limit",
        "bns:7#0": "below_send_limit",
        "bns:1#0": "below_return_limit",
    }
    info = result.trace["rerank"]
    assert [r["chunk_id"] for r in info["before"]] == [f"bns:{n}#0" for n in (1, 2, 3, 4)]
    assert [r["chunk_id"] for r in info["after"]] == [f"bns:{n}#0" for n in (4, 3, 2, 1)]
    assert [r["rerank_score"] for r in info["after"]] == [3.0, 2.0, 1.0, 0.0]  # incl. cut bns:1
    assert info["selected"] == [c.chunk_id for c in result.results]
    assert info["candidates_taken"] == info["returned"] + info["omitted"]
    assert (info["sent"], info["final_count"], info["below_limit"]) == (4, 3, 1)


def test_request_limit_can_only_lower_the_final_count(monkeypatch):
    fake_hybrid(monkeypatch)
    result = rerank.retrieve(request(limit=2), settings=settings(), rerank=reverse_scores)
    assert len(result.results) == 2 and result.trace["rerank"]["final_count"] == 2


def test_missing_key_is_explicit_and_calls_nothing(monkeypatch):
    monkeypatch.setattr(hybrid, "retrieve", no_call)
    result = rerank.retrieve(request(), settings=settings(rerank_api_key=""), rerank=no_call)
    assert result.status == "unavailable" and result.results == []
    assert result.trace["reason"] == "missing_setting"
    assert "RERANK_API_KEY is not set" in result.message


def test_invalid_limits_are_unavailable(monkeypatch):
    monkeypatch.setattr(hybrid, "retrieve", no_call)
    result = rerank.retrieve(request(), settings=settings(rerank_return_limit=9))
    assert result.status == "unavailable" and result.trace["reason"] == "invalid_rerank_settings"
    assert "RERANK_RETURN_LIMIT" in result.message


def test_empty_scope_is_no_results_without_a_rerank_call():
    req = request(filters=SemanticFilters(access_level=["restricted"]))
    result = rerank.retrieve(req, settings=settings(), rerank=no_call)
    assert result.status == "no_results" and result.pattern == "hybrid-reranked"


def test_hybrid_no_results_makes_no_rerank_call(monkeypatch):
    empty = QueryResult(pattern="hybrid", status="no_results", message="m")
    monkeypatch.setattr(hybrid, "retrieve", lambda *a, **k: empty)
    result = rerank.retrieve(request(), settings=settings(), rerank=no_call)
    assert result.status == "no_results" and result.pattern == "hybrid-reranked"


def _timeout(*args):
    raise rerank.RerankFailure("timeout")


@pytest.mark.parametrize("failing", [_timeout, lambda *a: [1.0]])  # second: incomplete scores
def test_provider_failure_never_returns_the_hybrid_list(monkeypatch, failing):
    fake_hybrid(monkeypatch)
    result = rerank.retrieve(request(), settings=settings(), rerank=failing)
    assert result.status == "unavailable" and result.results == []
    assert result.trace["reason"] == "rerank_failed" and result.omitted_candidates == []
    assert "No re-ranking was performed" in result.message


class FakeResponse:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code, self._payload, self._bad = status, payload, bad_json

    def json(self):
        if self._bad:
            raise ValueError("x")
        return self._payload


def _data(*pairs):
    return FakeResponse(payload={"data": [{"index": i, "relevance_score": s} for i, s in pairs]})


@pytest.mark.parametrize(
    "outcome, cause",
    [
        (httpx.ReadTimeout("t"), "timeout"),
        (httpx.ConnectError("c"), "connection_error"),
        (FakeResponse(status=500), "http_500"),
        (FakeResponse(bad_json=True), "malformed_response"),
        (_data((0, 0.5)), "malformed_response"),  # one document has no score
        (_data((0, 1), (0, 1)), "malformed_response"),  # duplicate index
        (_data((0, 1), (5, 1)), "malformed_response"),  # unknown index
    ],
)
def test_call_rerank_failures_have_coarse_causes(monkeypatch, outcome, cause):
    def post(url, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(rerank.RerankFailure) as exc:
        rerank.call_rerank("q", ["a", "b"], settings())
    assert exc.value.cause == cause


def test_call_rerank_request_and_response_mapping(monkeypatch):
    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json, timeout=timeout, auth=headers["Authorization"])
        return _data((1, 0.9), (0, 0.1))

    monkeypatch.setattr(httpx, "post", post)
    s = settings(rerank_api_base_url="https://p.example/v1/", rerank_request_timeout_seconds=7)
    assert rerank.call_rerank("q", ["a", "b"], s) == [0.1, 0.9]  # back in input order
    assert seen["url"] == "https://p.example/v1/rerank" and seen["timeout"] == 7
    assert seen["json"] == {"model": "rerank-2.5", "query": "q", "documents": ["a", "b"], "top_k": 2}
    assert seen["auth"] == "Bearer rk"


def test_shared_path_one_retrieval_one_generation_only_kept_passages(monkeypatch):
    seen = {"retrieve": 0, "generate": 0}

    def fake_retrieve(req):
        seen["retrieve"] += 1
        kept = chunk(4).model_copy(update={"rerank_rank": 1, "rerank_score": 0.9})
        return QueryResult(pattern="hybrid-reranked", status="ok", message="m", results=[kept])

    def fake_generate(result, question, stream=False):
        seen["generate"] += 1
        assert [p.chunk_id for p in result.results] == ["bns:4#0"]
        yield ("final", result)

    monkeypatch.setattr(rerank, "retrieve", fake_retrieve)
    monkeypatch.setattr(hybrid, "retrieve", no_call)
    monkeypatch.setattr(semantic, "retrieve", no_call)
    monkeypatch.setattr(answer, "generate_events", fake_generate)
    final = service.run_pattern(request(generate_answer=True), answer_generation=True)
    assert final.pattern == "hybrid-reranked" and seen == {"retrieve": 1, "generate": 1}


def test_other_modes_keep_rerank_fields_null():
    item = chunk(1)
    assert (item.rerank_score, item.rerank_rank) == (None, None)
