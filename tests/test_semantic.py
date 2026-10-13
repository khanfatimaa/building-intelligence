import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from building_with_rag.contracts import QueryRequest, SemanticFilters
from building_with_rag.main import app
from building_with_rag.retrieval import semantic
from building_with_rag.service import run_pattern
from building_with_rag.settings import Settings


def test_filters_accept_lists_and_reject_the_rest():
    ok = SemanticFilters(act=["BNS_2023"], status=["in_force"], access_level=["public"])
    assert ok.act == ["BNS_2023"]
    for bad in (
        {"chapter": ["I"]},  # unknown field
        {"act": "BNS_2023"},  # not a list
        {"act": [{"$ne": "x"}]},  # operator shaped
        {"act": ["$where"]},  # fragment-like value
        {"act": []},
    ):
        with pytest.raises(ValidationError):
            SemanticFilters(**bad)


def test_blank_question_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="   ", pattern="semantic")


def test_scope_narrows_never_widens():
    assert semantic.effective_scope(None) == {"access_level": ["public"]}
    scope = semantic.effective_scope(SemanticFilters(act=["IPC_1860"], access_level=["public"]))
    assert semantic.build_filter(scope) == {
        "$and": [{"access_level": {"$in": ["public"]}}, {"act": {"$in": ["IPC_1860"]}}]
    }
    assert semantic.effective_scope(SemanticFilters(access_level=["restricted"])) is None
    assert semantic.num_candidates(20) >= 20


def test_empty_scope_is_no_result_without_any_call():
    def boom(*args, **kwargs):
        raise AssertionError("must not be called")

    request = QueryRequest(
        question="x", pattern="semantic", filters=SemanticFilters(access_level=["restricted"])
    )
    result = semantic.retrieve(request, settings=Settings(), db=boom, embed=boom)
    assert result.status == "no_results"
    assert result.results == []


def test_missing_configuration_is_unavailable_not_no_result():
    request = QueryRequest(question="What is theft?", pattern="semantic")
    result = semantic.retrieve(request, settings=Settings(mongodb_uri="", voyage_api_key=""))
    assert result.status == "unavailable"
    assert "MONGODB_URI" in result.message


def test_generate_answer_keeps_placeholder():
    request = QueryRequest(question="What is theft?", pattern="semantic", generate_answer=True)
    result = run_pattern(request)
    assert result.status == "not_implemented"
    assert result.results == []


def test_query_route_maps_unavailable_to_503_and_bad_filter_to_422(monkeypatch):
    monkeypatch.setattr(semantic, "get_settings", lambda: Settings(mongodb_uri=""))
    client = TestClient(app)
    body = {"question": "What is theft?", "pattern": "semantic"}
    assert client.post("/v1/query", json={**body, "filters": {"$where": ["x"]}}).status_code == 422
    response = client.post("/v1/query", json=body)
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
