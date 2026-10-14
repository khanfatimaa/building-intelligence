import json

import httpx
import pytest
from fastapi.testclient import TestClient

from building_with_rag import service
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.generation import answer, context
from building_with_rag.main import app
from building_with_rag.retrieval import semantic
from building_with_rag.settings import Settings

CONFIGURED = Settings(generation_api_base_url="http://proxy.invalid/v1", generation_api_key="k")


def chunk(
    n: int,
    text: str = "Murder is punishable with death or imprisonment for life.",
    act: str = "BNS_2023",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"bns:{n}#0",
        section_id=f"bns:{n}",
        act=act,
        text=text,
        heading=f"Heading {n}",
        score=0.9,
        source={
            "section_label": f"BNS section {n}",
            "source_id": "source:BNS_2023",
            "source_pdf": "bns.pdf",
            "page_start": n,
            "page_end": n,
        },
    )


def retrieval(chunks: list[RetrievedChunk]) -> QueryResult:
    return QueryResult(
        pattern="semantic",
        status="ok" if chunks else "no_results",
        message="m",
        trace={"query": "q"},
        results=chunks,
    )


def model_says(payload) -> answer.ModelCall:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return lambda messages, settings: (text, "proxy-model")


def never_called(*args, **kwargs):
    raise AssertionError("the model must not be called")


# --- context assembly ---------------------------------------------------------------------------


def test_context_is_bounded_in_count_and_characters_without_cutting(monkeypatch):
    monkeypatch.setattr(context, "MAX_PASSAGES", 3)
    ctx = context.assemble([chunk(n) for n in range(1, 6)])
    assert [p.label for p in ctx.passages] == ["E1", "E2", "E3"]
    assert ctx.outcome == "truncated"
    assert ctx.omitted == [
        {"chunk_id": "bns:4#0", "reason": "passage_limit"},
        {"chunk_id": "bns:5#0", "reason": "passage_limit"},
    ]

    monkeypatch.setattr(context, "MAX_PASSAGES", 10)
    monkeypatch.setattr(context, "MAX_CONTEXT_CHARS", 25)
    ctx = context.assemble([chunk(1, "a" * 10), chunk(2, "b" * 20), chunk(3, "c" * 5)])
    # The second passage does not fit, so it and the (smaller) third are left out.
    assert [p.chunk.text for p in ctx.passages] == ["a" * 10]
    assert ctx.chars == 10
    assert [o["reason"] for o in ctx.omitted] == ["character_budget", "character_budget"]


def test_context_labels_keep_ids_and_source_details():
    ctx = context.assemble([chunk(103), chunk(302)])
    assert ctx.outcome == "ok"
    first = ctx.by_label["E1"]
    assert (first.chunk.chunk_id, first.chunk.section_id) == ("bns:103#0", "bns:103")
    assert first.section_label == "BNS section 103"
    assert first.source_details()["source_pdf"] == "bns.pdf"
    assert ctx.trace()["retrieved"] == 2 and ctx.trace()["kept"] == 2


def test_no_passages_means_no_model_call():
    result = answer.generate(retrieval([]), "q", settings=CONFIGURED, call=never_called)
    assert result.status == "no_results"
    assert result.generation.outcome == "insufficient_evidence"
    assert result.generation.context_outcome == "no_evidence"
    assert result.generation.answer is None


# --- validation and resolution ------------------------------------------------------------------

GOOD = {
    "outcome": "answered",
    "answer": "Murder is punished under BNS section 103 [E1].",
    "claims": [{"text": "Murder is punishable.", "evidence": ["E1", "E1"]}],
    "reason": "",
}


def run(payload, chunks=None):
    return answer.generate(
        retrieval(chunks or [chunk(103), chunk(104)]), "q", settings=CONFIGURED,
        call=model_says(payload),
    )  # fmt: skip


def test_valid_answer_resolves_to_act_qualified_citations():
    result = run(f"```json\n{json.dumps(GOOD)}\n```")
    generation = result.generation
    assert generation.outcome == "answered"
    assert generation.answer == GOOD["answer"]
    assert generation.claims == [{"text": "Murder is punishable.", "labels": ["E1"]}]
    (citation,) = generation.citations
    assert citation["citation"] == "BNS section 103"
    assert citation["section_id"] == "bns:103"
    assert citation["chunk_id"] == "bns:103#0"
    assert citation["source_pdf"] == "bns.pdf"
    assert [p.chunk_id for p in generation.supporting_passages] == ["bns:103#0"]
    assert generation.model == "proxy-model"
    assert generation.trace["supplied_labels"] == ["E1", "E2"]
    assert generation.trace["cited_labels"] == ["E1"]
    assert result.status == "ok" and len(result.results) == 2


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({**GOOD, "claims": [{"text": "x", "evidence": ["E9"]}]}, "unknown_label"),
        ({**GOOD, "claims": []}, "no_claims"),
        ({**GOOD, "claims": [{"text": "x", "evidence": []}]}, "claim_without_label"),
        ({**GOOD, "answer": " "}, "empty_answer"),
        ("The answer is life imprisonment.", "not_json"),
        ("[1, 2]", "not_an_object"),
    ],
)
def test_invalid_model_output_is_malformed_without_a_fallback_answer(payload, reason):
    generation = run(payload).generation
    assert generation.outcome == "malformed"
    assert generation.trace["reason"] == reason
    assert generation.answer is None and generation.citations == []


def test_model_insufficient_evidence_returns_no_answer():
    payload = {
        "outcome": "insufficient_evidence",
        "answer": "guess",
        "claims": [],
        "reason": "off topic",
    }
    result = run(payload)
    assert result.generation.outcome == "insufficient_evidence"
    assert result.generation.answer is None
    assert result.generation.trace["model_reason"] == "off topic"
    assert result.generation.supporting_passages == []


def test_empty_settings_are_unavailable_naming_the_setting():
    settings = Settings(generation_api_base_url="http://proxy.invalid/v1", generation_api_key="")
    result = answer.generate(retrieval([chunk(1)]), "q", settings=settings)
    assert result.generation.outcome == "unavailable"
    assert result.generation.trace["reason"] == "missing_generation_settings: GENERATION_API_KEY"
    assert result.generation.answer is None


def test_timeout_is_unavailable(monkeypatch):
    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("slow", request=None)

    monkeypatch.setattr(answer.httpx, "post", timeout)
    result = answer.generate(retrieval([chunk(1)]), "q", settings=CONFIGURED)
    assert result.generation.outcome == "unavailable"
    assert result.generation.trace["reason"] == "timeout"
    assert "proxy.invalid" not in result.message


# --- routing ------------------------------------------------------------------------------------


def test_other_modes_keep_the_placeholder_in_chat(monkeypatch):
    monkeypatch.setattr(semantic, "retrieve", never_called)
    monkeypatch.setattr(answer, "generate_events", never_called)
    body = {"model": "rag-hybrid", "messages": [{"role": "user", "content": "What is theft?"}]}
    text = TestClient(app).post("/v1/chat/completions", json=body).json()["choices"][0]
    assert "not implemented" in text["message"]["content"]


def test_query_without_generate_answer_has_no_generation(monkeypatch):
    calls = []

    def fake_retrieve(request):
        calls.append(request.question)
        return retrieval([chunk(1)])

    monkeypatch.setattr(semantic, "retrieve", fake_retrieve)
    monkeypatch.setattr(answer, "generate", never_called)
    response = TestClient(app).post(
        "/v1/query", json={"question": "What is theft?", "pattern": "semantic"}
    )
    assert response.status_code == 200
    assert response.json()["generation"] is None
    assert calls == ["What is theft?"]


def test_query_with_generate_answer_retrieves_once_and_generates(monkeypatch):
    calls = []

    def fake_retrieve(request):
        calls.append(request.question)
        return retrieval([chunk(103)])

    monkeypatch.setattr(semantic, "retrieve", fake_retrieve)
    monkeypatch.setattr(answer, "get_settings", lambda: CONFIGURED)
    monkeypatch.setattr(answer, "call_model", model_says(GOOD))
    response = TestClient(app).post(
        "/v1/query", json={"question": "Murder?", "pattern": "semantic", "generate_answer": True}
    )
    assert response.status_code == 200
    assert response.json()["generation"]["outcome"] == "answered"
    assert calls == ["Murder?"]


def test_unavailable_retrieval_skips_generation(monkeypatch):
    unavailable = QueryResult(pattern="semantic", status="unavailable", message="m")
    monkeypatch.setattr(semantic, "retrieve", lambda request: unavailable)
    monkeypatch.setattr(answer, "generate", never_called)
    request = QueryRequest(question="q", pattern="semantic", generate_answer=True)
    result = service.run_pattern(request, answer_generation=True)
    assert result.status == "unavailable" and result.generation is None
