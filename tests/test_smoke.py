from fastapi.testclient import TestClient

from building_with_rag.contracts import QueryResult
from building_with_rag.main import app
from building_with_rag.retrieval import semantic

client = TestClient(app)
MODELS = [
    "rag-semantic",
    "rag-hybrid",
    "rag-hybrid-reranked",
    "rag-structured",
    "rag-decomposition",
    "rag-hyde",
]


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_query_placeholder():
    r = client.post("/v1/query", json={"question": "What is theft?", "pattern": "decomposition"})
    assert r.status_code == 200
    assert r.json()["status"] == "not_implemented"
    assert r.json()["results"] == []


def test_models():
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == MODELS


def test_chat_json_and_sse(monkeypatch):
    def no_results(request):
        return QueryResult(pattern="semantic", status="no_results", message="nothing matched")

    monkeypatch.setattr(semantic, "retrieve", no_results)
    body = {
        "model": "rag-semantic",
        "messages": [{"role": "user", "content": "hi"}],
        "rag_options": {"pattern": "semantic", "limit": 5},
    }
    r = client.post("/v1/chat/completions", json=body)
    assert "not give enough evidence" in r.json()["choices"][0]["message"]["content"]
    s = client.post("/v1/chat/completions", json={**body, "stream": True})
    assert s.text.rstrip().endswith("data: [DONE]")


def test_chat_unknown_model_error_envelope():
    r = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404 and "error" in r.json()
