"""Story 4.1: hybrid retrieval fusion, keyword stage, outcomes and shared dispatch. Offline."""

import pytest

from building_with_rag import service
from building_with_rag.contracts import QueryRequest, RetrievedChunk, SemanticFilters
from building_with_rag.generation import answer
from building_with_rag.retrieval import hybrid, semantic
from building_with_rag.settings import Settings

SETTINGS = Settings(_env_file=None, mongodb_uri="mongodb://x", voyage_api_key="k")


def chunk(n: int, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"bns:{n}#0",
        section_id=f"bns:{n}",
        act="BNS_2023",
        text=f"Murder is punishable with death or imprisonment for life, section {n}.",
        heading=f"Heading {n}",
        score=score,
        source={"section_label": f"BNS section {n}", "source_pdf": "bns.pdf"},
    )


def test_fusion_rule_and_fields():
    semantic_route = [chunk(1), chunk(2), chunk(3)]
    keyword_route = [chunk(2), chunk(4)]
    fused, omitted = hybrid.fuse(semantic_route, keyword_route, limit=3)

    assert [c.chunk_id for c in fused] == ["bns:2#0", "bns:1#0", "bns:4#0"]  # in both routes first
    assert [c.fused_rank for c in fused] == [1, 2, 3]
    assert all(c.score == c.fused_score for c in fused)
    assert fused[0].fused_score == pytest.approx(1 / (hybrid.RRF_K + 2) + 1 / (hybrid.RRF_K + 1))
    assert (fused[0].semantic_rank, fused[0].keyword_rank) == (2, 1)
    assert hybrid.route_label(fused[0]) == "both"
    only_semantic, only_keyword = fused[1], fused[2]
    assert (only_semantic.keyword_rank, only_semantic.keyword_score) == (None, None)
    assert (only_keyword.semantic_rank, only_keyword.semantic_score) == (None, None)
    assert [c.chunk_id for c in omitted] == ["bns:3#0"]  # the limit applies after fusion
    assert omitted[0].omitted_reason == "below_limit"


def test_fusion_ties_break_by_chunk_id():
    fused, _ = hybrid.fuse([chunk(9)], [chunk(1)], limit=5)  # equal 1/(k+1) contributions
    assert [c.chunk_id for c in fused] == ["bns:1#0", "bns:9#0"]


def test_keyword_stage_scope_comes_from_validated_lists():
    scope = semantic.effective_scope(SemanticFilters(act=["BNS_2023"]))
    stage = hybrid.keyword_stage("culpable homicide", scope, 20)
    compound = stage[0]["$search"]["compound"]
    assert compound["must"] == [{"text": {"query": "culpable homicide", "path": "text"}}]
    assert {"in": {"path": "access_level", "value": ["public"]}} in compound["filter"]
    assert {"in": {"path": "act", "value": ["BNS_2023"]}} in compound["filter"]
    assert stage[1] == {"$limit": 20}


def test_empty_scope_is_no_result_without_any_call():
    def boom(*args, **kwargs):
        raise AssertionError("must not be called")

    request = QueryRequest(
        question="x", pattern="hybrid", filters=SemanticFilters(access_level=["restricted"])
    )
    result = hybrid.retrieve(request, settings=SETTINGS, db=boom, embed=boom)
    assert result.status == "no_results" and result.results == []


def _fake_routes(monkeypatch, *, keyword_ready=True):
    monkeypatch.setattr(hybrid, "_ensure_ready", lambda db, settings: None)

    def keyword_ready_check(db, settings):
        if not keyword_ready:
            raise hybrid._keyword_unavailable("keyword_index_missing")

    monkeypatch.setattr(hybrid, "_ensure_keyword_ready", keyword_ready_check)
    monkeypatch.setattr(hybrid, "vector_hits", lambda *a: [{"chunk_id": "bns:1#0", "score": 0.9}])
    monkeypatch.setattr(
        hybrid,
        "keyword_hits",
        lambda *a: [{"chunk_id": "bns:2#0", "score": 7.0}, {"chunk_id": "bns:1#0", "score": 3.0}],
    )
    by_id = {c.chunk_id: c for c in (chunk(1), chunk(2))}
    monkeypatch.setattr(
        hybrid, "resolve_hits", lambda db, hits: [(h, by_id[h["chunk_id"]]) for h in hits]
    )


def test_hybrid_result_shows_both_routes(monkeypatch):
    _fake_routes(monkeypatch)
    request = QueryRequest(question="murder", pattern="hybrid")
    result = hybrid.retrieve(request, settings=SETTINGS, db=object(), embed=lambda q, s: [0.0])
    assert result.status == "ok" and result.pattern == "hybrid"
    assert [c.chunk_id for c in result.results] == ["bns:1#0", "bns:2#0"]
    assert result.trace["route_overlap"] == 1 and result.trace["keyword_only"] == 1
    assert result.trace["semantic_only"] == 0
    assert result.trace["fusion"]["name"] == "reciprocal_rank_fusion"


def test_missing_keyword_index_is_unavailable_not_semantic_only(monkeypatch):
    _fake_routes(monkeypatch, keyword_ready=False)
    request = QueryRequest(question="murder", pattern="hybrid")
    result = hybrid.retrieve(request, settings=SETTINGS, db=object(), embed=lambda q, s: [0.0])
    assert result.status == "unavailable" and result.results == []
    assert result.trace["reason"] == "keyword_index_missing"


def test_semantic_results_keep_hybrid_fields_null(monkeypatch):
    hits = [{"chunk_id": "bns:1#0", "score": 0.9}]
    monkeypatch.setattr(semantic, "vector_hits", lambda *a: hits)
    monkeypatch.setattr(semantic, "resolve_hits", lambda db, h: [(x, chunk(1, 0.9)) for x in h])
    results, _, _ = semantic.search(object(), [0.0], {"access_level": ["public"]}, 5)
    item = results[0]
    assert (item.semantic_score, item.semantic_rank, item.keyword_score) == (None, None, None)
    assert (item.keyword_rank, item.fused_score, item.fused_rank) == (None, None, None)


def test_hybrid_uses_the_shared_path_with_one_retrieval_and_one_generation(monkeypatch):
    seen = {"retrieve": 0, "generate": 0}

    def fake_retrieve(request):
        seen["retrieve"] += 1
        return hybrid.QueryResult(
            pattern="hybrid", status="ok", message="m", results=[chunk(1)]
        )

    def fake_generate(result, question, stream=False):
        seen["generate"] += 1
        assert [p.chunk_id for p in result.results] == ["bns:1#0"]
        yield ("text", "t")
        yield ("final", result)

    monkeypatch.setattr(hybrid, "retrieve", fake_retrieve)
    monkeypatch.setattr(semantic, "retrieve", lambda r: pytest.fail("semantic must not run"))
    monkeypatch.setattr(answer, "generate_events", fake_generate)
    request = QueryRequest(question="q", pattern="hybrid", generate_answer=True)
    final = service.run_pattern(request, answer_generation=True)
    assert final.pattern == "hybrid" and seen == {"retrieve": 1, "generate": 1}
