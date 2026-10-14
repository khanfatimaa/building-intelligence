"""Story 3.2: streamed answers, attempts, confidence and chat auth. Fully offline."""

import json

from fastapi.testclient import TestClient

from building_with_rag import main, service
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.generation import answer
from building_with_rag.generation.answer import GenerationFailure
from building_with_rag.retrieval import semantic
from building_with_rag.settings import Settings

CONFIGURED = Settings(
    _env_file=None, generation_api_base_url="http://proxy.invalid/v1", generation_api_key="k"
)
CHAT = {
    "model": "rag-semantic",
    "messages": [{"role": "user", "content": "What is the punishment for murder?"}],
}
GOOD = {
    "outcome": "answered",
    "answer": "Murder is punishable under BNS section 103.",
    "claims": [{"text": "Murder is punishable.", "evidence": ["E1"]}],
    "reason": "",
}
BAD_LABEL = {**GOOD, "claims": [{"text": "Murder is punishable.", "evidence": ["E9"]}]}


def passage() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="bns:103#0",
        section_id="bns:103",
        act="BNS_2023",
        text="Murder is punishable with death or imprisonment for life.",
        heading="Punishment for murder",
        score=0.9,
        source={
            "section_label": "BNS section 103",
            "source_id": "source:BNS_2023",
            "source_pdf": "bns.pdf",
            "page_start": 45,
            "page_end": 46,
        },
    )


class Harness:
    """Fakes retrieval and the model; counts calls so one operation per request can be asserted."""

    def __init__(self, monkeypatch, attempts):
        self.retrievals = 0
        self.model_calls = 0
        self.attempts = list(attempts)  # each: dict/str payload, or a list of pieces, or Exception

        def retrieve(request):
            self.retrievals += 1
            return QueryResult(
                pattern="semantic", status="ok", message="m", trace={}, results=[passage()]
            )

        def pieces():
            self.model_calls += 1
            item = self.attempts.pop(0)
            if isinstance(item, list):
                return item
            text = item if isinstance(item, str) else json.dumps(item)
            return [text[i : i + 7] for i in range(0, len(text), 7)]

        def stream(messages, settings, info):
            info["model"] = "proxy-model"
            for piece in pieces():
                if isinstance(piece, Exception):
                    raise piece
                yield piece

        def one_shot(messages, settings):
            parts = pieces()
            return "".join(parts), "proxy-model"

        monkeypatch.setattr(semantic, "retrieve", retrieve)
        monkeypatch.setattr(answer, "get_settings", lambda: CONFIGURED)
        monkeypatch.setattr(answer, "call_model_stream", stream)
        monkeypatch.setattr(answer, "call_model", one_shot)


def sse_frames(response):
    lines = [x for x in response.text.split("\n\n") if x]
    assert lines[-1] == "data: [DONE]"
    return [json.loads(x[len("data: ") :]) for x in lines[:-1]]


def streamed_text(frames):
    return "".join(f["choices"][0]["delta"].get("content") or "" for f in frames)


def test_validated_first_attempt_streams_a_draft_then_high_confidence(monkeypatch):
    h = Harness(monkeypatch, [GOOD] * 3)
    client = TestClient(main.app)
    response = client.post("/v1/chat/completions", json={**CHAT, "stream": True})
    frames = sse_frames(response)
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(f["object"] == "chat.completion.chunk" for f in frames)
    text = streamed_text(frames)
    assert text.startswith("**DRAFT — checking evidence**")
    assert "Murder is punishable under BNS section 103." in text
    assert "Checks passed. Confidence: high." in text
    assert "BNS section 103 (bns.pdf, pages 45-46)" in text
    assert (h.retrievals, h.model_calls) == (1, 1)
    h.retrievals = h.model_calls = 0

    plain = client.post("/v1/chat/completions", json=CHAT).json()
    assert plain["choices"][0]["message"]["content"] == text  # same rendering, same operation

    final = list(service.run_pattern_events(_request(), answer_generation=True, stream=True))[-1][1]
    generation = final.generation
    assert generation.outcome == "answered"
    assert generation.confidence == answer.HIGH_CONFIDENCE and generation.attempts == 1
    assert generation.issues == [] and generation.draft_answer is None
    assert [c["citation"] for c in generation.citations] == ["BNS section 103"]


def _request():
    return QueryRequest(question="q", pattern="semantic", generate_answer=True)


def test_failed_check_retries_once_and_keeps_the_failure(monkeypatch):
    h = Harness(monkeypatch, [BAD_LABEL, GOOD])
    events = list(service.run_pattern_events(_request(), answer_generation=True, stream=True))
    text = "".join(v for k, v in events if k == "text")
    generation = events[-1][1].generation
    assert h.model_calls == 2 and generation.attempts == 2
    assert generation.outcome == "answered" and generation.confidence == answer.HIGH_CONFIDENCE
    assert generation.issues == ["attempt 1: claim 1 cites E9, which was not supplied"]
    log = generation.trace["attempt_log"]
    assert (
        log[0]["cited_labels"] == ["E9"] and log[0]["failed_checks"][0]["check"] == "unknown_label"
    )
    assert log[1]["failed_checks"] == []
    assert "**Check failed:** claim 1 cites E9" in text
    assert "Retrying once (attempt 2 of 2)" in text
    assert text.count("**DRAFT — checking evidence**") == 2


def test_two_failed_attempts_end_as_a_labelled_low_confidence_draft(monkeypatch):
    Harness(monkeypatch, [BAD_LABEL, BAD_LABEL])
    events = list(service.run_pattern_events(_request(), answer_generation=True, stream=True))
    text = "".join(v for k, v in events if k == "text")
    result = events[-1][1]
    generation = result.generation
    assert generation.outcome == "malformed" and generation.answer is None
    assert generation.confidence == answer.LOW_CONFIDENCE and generation.attempts == 2
    assert generation.draft_answer == GOOD["answer"]
    assert generation.low_confidence_reason
    assert len(generation.issues) == 2
    assert generation.citations == [] and generation.supporting_passages == []
    assert generation.trace["draft_cited_labels"] == ["E9"]
    assert generation.trace["attempt_log"][1]["cited_labels"] == ["E9"]
    assert (
        text.rstrip()
        .rsplit("\n\n---\n", 1)[1]
        .startswith("**DRAFT — low confidence, not the final answer.**")
    )
    assert "Checks passed" not in text
    assert result.status == "ok" and len(result.results) == 1  # retrieval untouched


def test_unsupported_claim_is_caught_by_the_support_check(monkeypatch):
    claim = {"text": "Theft carries 7 years of rigorous imprisonment.", "evidence": ["E1"]}
    Harness(monkeypatch, [{**GOOD, "claims": [claim]}] * 2)
    generation = list(service.run_pattern_events(_request(), answer_generation=True))[-1][1]
    assert generation.generation.confidence == answer.LOW_CONFIDENCE
    checks = {
        f["check"] for a in generation.generation.trace["attempt_log"] for f in a["failed_checks"]
    }
    assert {"claim_not_supported", "number_not_in_evidence"} <= checks


def test_provider_failure_after_text_began_still_closes_the_stream(monkeypatch):
    prefix = '{"outcome": "answered", "answer": "Murder is pun'
    failure = GenerationFailure(
        "unavailable", "connection_error", "the model service could not be reached"
    )
    Harness(monkeypatch, [[prefix, failure]])
    response = TestClient(main.app).post("/v1/chat/completions", json={**CHAT, "stream": True})
    frames = sse_frames(response)
    text = streamed_text(frames)
    assert "Murder is pun" in text
    assert "**UNAVAILABLE — the text above is an incomplete draft, not a final answer.**" in text
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert "proxy.invalid" not in text


def test_unparseable_output_has_no_confidence(monkeypatch):
    Harness(monkeypatch, ["I think the answer is life imprisonment."])
    events = list(service.run_pattern_events(_request(), answer_generation=True, stream=True))
    generation = events[-1][1].generation
    assert generation.outcome == "malformed" and generation.trace["reason"] == "not_json"
    assert generation.confidence is None and generation.draft_answer is None
    assert generation.answer is None and generation.attempts == 1


def test_insufficient_evidence_streams_no_draft(monkeypatch):
    payload = {"outcome": "insufficient_evidence", "answer": None, "claims": [], "reason": "off"}
    Harness(monkeypatch, [payload])
    events = list(service.run_pattern_events(_request(), answer_generation=True, stream=True))
    text = "".join(v for k, v in events if k == "text")
    assert "DRAFT" not in text and "not give enough evidence" in text
    assert events[-1][1].generation.outcome == "insufficient_evidence"
    assert events[-1][1].generation.confidence is None


def test_chat_requires_the_capstone_key_only_when_it_is_set(monkeypatch):
    h = Harness(monkeypatch, [GOOD, GOOD])
    client = TestClient(main.app)
    monkeypatch.setattr(
        main, "get_settings", lambda: Settings(_env_file=None, capstone_api_key="s3cret")
    )
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}):
        response = client.post("/v1/chat/completions", json=CHAT, headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_api_key"
        assert "s3cret" not in response.text
    assert (h.retrievals, h.model_calls) == (0, 0)
    ok = client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer s3cret"})
    assert (
        ok.status_code == 200
        and "Confidence: high" in ok.json()["choices"][0]["message"]["content"]
    )
    monkeypatch.setattr(main, "get_settings", lambda: Settings(_env_file=None))
    assert client.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_query_and_chat_share_one_path_and_result(monkeypatch):
    h = Harness(monkeypatch, [GOOD, GOOD])
    client = TestClient(main.app)
    body = client.post(
        "/v1/query", json={"question": "q", "pattern": "semantic", "generate_answer": True}
    ).json()
    chat_final = list(service.run_pattern_events(_request(), answer_generation=True))[-1][1]
    assert body["generation"] == chat_final.model_dump()["generation"]
    assert body["generation"]["confidence"] == answer.HIGH_CONFIDENCE
    assert h.retrievals == 2 and h.model_calls == 2  # one of each per request
