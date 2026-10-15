import json

import pytest
from fastapi.testclient import TestClient

from building_with_rag import service
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.generation import answer
from building_with_rag.main import app
from building_with_rag.retrieval import decomposition, hyde, semantic
from building_with_rag.settings import Settings

CONFIGURED = Settings(
    mongodb_uri="mongodb://fake",
    voyage_api_key="v",
    generation_api_base_url="http://proxy.invalid/v1",
    generation_api_key="k",
)
QUESTION = "Compare the punishment for murder under the BNS with the punishment for murder under the IPC."
BNS_SUB = "What is the punishment for murder under the BNS?"
IPC_SUB = "What is the punishment for murder under the IPC?"
HYPO = "A person who falsely claims an official post to deceive another commits an offence."


@pytest.fixture(autouse=True)
def no_step_delay(monkeypatch):
    monkeypatch.setattr(decomposition, "STEP_DELAY_SECONDS", 0)


def chunk(chunk_id: str, text: str = "Whoever commits murder shall be punished.") -> RetrievedChunk:
    act, _, number = chunk_id.partition(":")
    number = number.split("#")[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        section_id=f"{act}:{number}",
        act="BNS_2023" if act == "bns" else "IPC_1860",
        text=text,
        heading="Punishment for murder",
        score=0.9,
        source={"section_label": f"{act.upper()} section {number}"},
    )


def calls(payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return lambda messages, settings: (text, "proxy-model")


def never(*args, **kwargs):
    raise AssertionError("must not be called")


def retrieval(chunks, status="ok"):
    return QueryResult(pattern="semantic", status=status, message="m", results=chunks)


def request(pattern, **kw):
    return QueryRequest(question=QUESTION, pattern=pattern, **kw)


# --- decomposition: plan validation -------------------------------------------------------------


def test_good_plan_is_accepted(monkeypatch):
    monkeypatch.setattr(semantic, "retrieve", lambda *a, **k: retrieval([chunk("bns:103#0")]))
    result = decomposition.retrieve(
        request("decomposition"), settings=CONFIGURED, call=calls({"subquestions": [BNS_SUB, IPC_SUB]})
    )
    assert [s.subquestion for s in result.subquestions] == [BNS_SUB, IPC_SUB]


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"subquestions": []}, "plan_empty"),
        ("this is not json", "plan_malformed"),
        ({"subquestions": [BNS_SUB, IPC_SUB, "What is murder under the BNS?", "Murder under IPC?"]}, "plan_too_many"),
        ({"subquestions": [BNS_SUB, BNS_SUB.lower()]}, "plan_duplicate"),
        (
            {"subquestions": ["What is the punishment for murder under the Motor Vehicles Act?"]},
            "plan_out_of_scope",
        ),
    ],
)
def test_bad_plans_clarify_without_retrieval(monkeypatch, payload, reason):
    monkeypatch.setattr(semantic, "retrieve", never)
    result = decomposition.retrieve(request("decomposition"), settings=CONFIGURED, call=calls(payload))
    assert (result.status, result.trace["reason"]) == ("clarify", reason)
    assert result.results == [] and result.subquestions == []


# --- decomposition: evidence --------------------------------------------------------------------

SHARED = chunk("bns:1#0", "Murder and its punishment are described here.")


def fake_steps(by_act):
    seen = []

    def fake(step_request, **kwargs):
        seen.append((step_request.question, step_request.filters.act))
        return retrieval(by_act[step_request.filters.act[0]])

    return fake, seen


def test_every_step_evidenced_is_ok_with_deduplicated_union(monkeypatch):
    fake, seen = fake_steps(
        {"BNS_2023": [chunk("bns:103#0"), SHARED], "IPC_1860": [chunk("ipc:302#0"), SHARED]}
    )
    monkeypatch.setattr(semantic, "retrieve", fake)
    plan = {"subquestions": [BNS_SUB, IPC_SUB]}
    result = decomposition.retrieve(request("decomposition"), settings=CONFIGURED, call=calls(plan))
    assert result.status == "ok"
    assert [c.chunk_id for c in result.results] == ["bns:103#0", "bns:1#0", "ipc:302#0"]
    assert all(s.status == "evidenced" for s in result.subquestions)
    assert [c.chunk_id for c in result.subquestions[0].results] == ["bns:103#0", "bns:1#0"]
    assert [c.chunk_id for c in result.subquestions[1].results] == ["ipc:302#0", "bns:1#0"]
    assert seen == [(BNS_SUB, ["BNS_2023"]), (IPC_SUB, ["IPC_1860"])]  # each step narrowed to its act
    assert result.trace["decomposition"]["evidenced_count"] == 2


def test_pauses_between_searches_but_not_before_the_first(monkeypatch):
    monkeypatch.setattr(decomposition, "STEP_DELAY_SECONDS", 20)
    monkeypatch.setattr(semantic, "retrieve", lambda *a, **k: retrieval([chunk("bns:103#0")]))
    pauses = []
    decomposition.retrieve(
        request("decomposition"),
        settings=CONFIGURED,
        call=calls({"subquestions": [BNS_SUB, IPC_SUB]}),
        sleep=pauses.append,
    )
    assert pauses == [20]


def test_missing_evidence_is_partial_or_clarify_not_ok(monkeypatch):
    plan = {"subquestions": [BNS_SUB, IPC_SUB]}
    fake, _ = fake_steps({"BNS_2023": [chunk("bns:103#0")], "IPC_1860": []})
    monkeypatch.setattr(semantic, "retrieve", fake)
    partial = decomposition.retrieve(request("decomposition"), settings=CONFIGURED, call=calls(plan))
    assert partial.status == "partial_answer"
    assert [s.status for s in partial.subquestions] == ["evidenced", "no_evidence"]
    assert partial.subquestions[1].reason == "no_passages"
    assert partial.trace["decomposition"]["reason"] == "partial_evidence"
    assert "no complete comparison" in partial.message

    fake, _ = fake_steps({"BNS_2023": [], "IPC_1860": []})
    monkeypatch.setattr(semantic, "retrieve", fake)
    none = decomposition.retrieve(request("decomposition"), settings=CONFIGURED, call=calls(plan))
    assert none.status == "clarify"
    assert none.trace["decomposition"]["reason"] == "no_subquestion_evidence"


# --- HyDE -------------------------------------------------------------------------------------


def hyde_fixture(monkeypatch):
    embedded = []

    def fake_embed(text, settings):
        embedded.append(text)
        return text  # the "vector" is the text, so the fake search can tell the routes apart

    def fake_retrieve(step_request, *, settings=None, db=None, embed=None):
        searched = embed(step_request.question, settings)
        if searched == HYPO:
            return retrieval([chunk("ipc:170#0", "Personating a public servant."), chunk("bns:204#0")])
        return retrieval([chunk("bns:204#0"), chunk("ipc:99#0", "Another passage.")])

    monkeypatch.setattr(semantic, "retrieve", fake_retrieve)
    return fake_embed, embedded


def test_hyde_returns_real_passages_and_keeps_text_in_debug_field_only(monkeypatch):
    fake_embed, embedded = hyde_fixture(monkeypatch)
    result = hyde.retrieve(
        request("hyde"), settings=CONFIGURED, embed=fake_embed, call=calls({"hypothetical_text": HYPO})
    )
    assert result.status == "ok"
    assert [c.chunk_id for c in result.hyde_direct_candidates] == ["bns:204#0", "ipc:99#0"]
    assert [c.chunk_id for c in result.hyde_query_candidates] == ["ipc:170#0", "bns:204#0"]
    assert [c.chunk_id for c in result.results] == ["bns:204#0", "ipc:170#0", "ipc:99#0"]
    assert result.hyde_hypothetical_text_debug == HYPO
    assert embedded == [QUESTION, HYPO]
    assert result.trace["hyde"]["hypothetical_only"] == ["ipc:170#0"]
    dumped = result.model_dump()
    dumped.pop("hyde_hypothetical_text_debug")
    assert HYPO not in json.dumps(dumped)  # not in message, trace, results or candidates

    sent = []
    final = None
    for kind, value in answer.generate_events(
        result, QUESTION, settings=CONFIGURED, call=lambda m, s: (sent.append(m) or "{}", None)
    ):
        final = value if kind == "final" else final
    assert HYPO not in json.dumps(sent)  # never in a prompt to the answer generator
    assert HYPO not in final.generation.model_dump_json()


@pytest.mark.parametrize(
    "text", ["", "not json", json.dumps({"hypothetical_text": "  "}), json.dumps({"x": 1})]
)
def test_unusable_hypothetical_is_hyde_unavailable(monkeypatch, text):
    monkeypatch.setattr(semantic, "retrieve", never)
    result = hyde.retrieve(request("hyde"), settings=CONFIGURED, embed=never, call=calls(text))
    assert result.status == "hyde_unavailable"
    assert result.results == [] and result.hyde_hypothetical_text_debug is None
    assert result.hyde_direct_candidates == [] and result.hyde_query_candidates == []


# --- configuration and the shared generation path ----------------------------------------------


@pytest.mark.parametrize("module", [decomposition, hyde])
@pytest.mark.parametrize(
    ("overrides", "name"),
    [
        ({"generation_api_key": ""}, "GENERATION_API_KEY"),
        ({"generation_api_base_url": ""}, "GENERATION_API_BASE_URL"),
        ({"voyage_api_key": ""}, "VOYAGE_API_KEY"),
        ({"mongodb_uri": ""}, "MONGODB_URI"),
    ],
)
def test_missing_configuration_is_reported(monkeypatch, module, overrides, name):
    monkeypatch.setattr(semantic, "retrieve", never)
    monkeypatch.setattr(answer, "call_model", never)
    settings = CONFIGURED.model_copy(update=overrides)
    result = module.retrieve(request(module.PATTERN), settings=settings)
    assert result.status == "unavailable" and result.results == []
    assert result.trace["reason"] == "missing_setting" and name in result.message


@pytest.mark.parametrize(
    ("pattern", "status", "generations"),
    [
        ("decomposition", "ok", 1), ("decomposition", "partial_answer", 0),
        ("decomposition", "clarify", 0), ("hyde", "ok", 1), ("hyde", "hyde_unavailable", 0),
    ],
)  # fmt: skip
def test_generation_runs_only_for_ok(monkeypatch, pattern, status, generations):
    module = {"decomposition": decomposition, "hyde": hyde}[pattern]
    fake = QueryResult(pattern=pattern, status=status, message="m", results=[chunk("bns:103#0")])
    monkeypatch.setattr(module, "retrieve", lambda r: fake)
    ran = []

    def fake_generate(result, question, **kwargs):
        ran.append(result.results)
        yield ("final", result)

    monkeypatch.setattr(answer, "generate_events", fake_generate)
    service.run_pattern(request(pattern, generate_answer=True), answer_generation=True)
    assert len(ran) == generations
    assert all(r == fake.results for r in ran)


@pytest.mark.parametrize(
    ("model", "module", "status"),
    [("rag-decomposition", decomposition, "clarify"), ("rag-hyde", hyde, "hyde_unavailable")],
)
def test_streamed_chat_ends_with_done(monkeypatch, model, module, status):
    fake = QueryResult(pattern=model[4:], status=status, message="Not available here.")
    monkeypatch.setattr(module, "retrieve", lambda r: fake)
    body = {"model": model, "stream": True, "messages": [{"role": "user", "content": "q"}]}
    response = TestClient(app).post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert "Not available here." in response.text
    assert response.text.endswith("data: [DONE]\n\n")
