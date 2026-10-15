import pytest
from fastapi.testclient import TestClient

from building_with_rag.contracts import QueryRequest, SemanticFilters
from building_with_rag.generation import answer
from building_with_rag.main import app
from building_with_rag.retrieval import structured
from building_with_rag.retrieval.signals import classify
from building_with_rag.service import run_pattern
from building_with_rag.settings import Settings

SECTION = {
    "_id": "bns:103", "section_id": "bns:103", "act": "BNS_2023", "act_label": "BNS, 2023",
    "status": "in_force", "chapter": "VI", "section_number": 103, "heading": "Punishment for murder",
    "source_id": "source:BNS_2023", "access_level": "public", "page_start": 5, "page_end": 5,
    "source_status_version": "v1",
}
CHUNK = {"_id": "bns:103#0", "chunk_id": "bns:103#0", "section_id": "bns:103",
         "chunk_index": 0, "text": "Whoever commits murder..."}
SOURCE = {"_id": "source:BNS_2023", "source_pdf": "BNS.pdf", "source_sha256": "abc"}


class FakeCollection:
    def __init__(self, db, name, docs):
        self.db, self.name, self.docs = db, name, docs

    def find(self, query=None):
        self.db.calls.append((self.name, "find", query))
        query = query or {}
        return [d for d in self.docs if all(d.get(k) == v for k, v in query.items()
                                            if not isinstance(v, dict))]

    def __getattr__(self, item):  # any write or other method is a test failure
        raise AssertionError(f"unexpected call: {item}")


class FakeDb:
    def __init__(self, sections=(SECTION,)):
        self.calls = []
        self.data = {"sections": list(sections), "chunks": [CHUNK], "sources": [SOURCE]}

    def __getitem__(self, name):
        return FakeCollection(self, name, self.data[name])


def req(question, **kw):
    return QueryRequest(question=question, pattern="structured", **kw)


SETTINGS = Settings(mongodb_uri="mongodb://fake")


def test_classifier_exact_and_unsupported():
    for text, act, number in (("BNS section 103", "BNS", "103"),
                              ("section 302 of the IPC", "IPC", "302")):
        c = classify(text)
        assert c.kind == "exact" and c.signals.intent == "exact_lookup"
        assert (c.signals.act, c.signals.section_number) == (act, number)
    for text, reason in (("What does section 103 say?", "missing_act"),
                         ("BNS and IPC section 302", "ambiguous_act"),
                         ("IPC 498A", "lettered_section_unsupported"),
                         ("BNS sections 100 to 105", "multiple_sections"),
                         ("BNS section 5000", "section_out_of_range"),
                         ("What is the punishment for murder?", "not_structured_request")):
        c = classify(text)
        assert c.kind != "exact" and c.reason == reason, text
    assert classify("list the sections in chapter VI of the BNS").signals.intent == "filter"
    assert classify("how many sections are in the IPC").signals.intent == "aggregation"


def test_no_guess_means_no_database_call():
    db = FakeDb()
    for q in ("section 103", "BNS and IPC section 302", "IPC 498A", "tell me about theft"):
        result = structured.retrieve(req(q), settings=SETTINGS, db=db)
        assert result.status in ("clarification_needed", "recommendation") and not result.results
        assert result.trace["database_called"] is False
    assert db.calls == []


def test_exact_lookup_is_read_only_and_uses_validated_signals_only():
    db = FakeDb()
    q = 'BNS section 103 {"$gt": ""} $where'
    result = structured.retrieve(req(q), settings=SETTINGS, db=db)
    assert result.status == "ok" and result.results[0].section_id == "bns:103"
    assert result.results[0].score is None and result.results[0].source["status"] == "in_force"
    sections_calls = [c for c in db.calls if c[0] == "sections"]
    assert len(sections_calls) == 1
    assert sections_calls[0][2] == {"act": "BNS_2023", "section_number": 103,
                                    "access_level": "public"}
    assert result.trace["predicate"] == sections_calls[0][2]


def test_narrowing_and_missing_record():
    db = FakeDb()
    r = structured.retrieve(req("BNS section 103", chapter="VI",
                                filters=SemanticFilters(status=["in_force"])),
                            settings=SETTINGS, db=db)
    assert db.calls[0][2]["chapter"] == "VI" and db.calls[0][2]["status"] == {"$in": ["in_force"]}
    assert r.status == "ok"
    excluded = structured.retrieve(req("BNS section 103", filters=SemanticFilters(act=["IPC_1860"])),
                                   settings=SETTINGS, db=FakeDb())
    assert excluded.status == "not_found" and excluded.trace["database_called"] is False
    missing = structured.retrieve(req("IPC section 600"), settings=SETTINGS, db=FakeDb())
    assert missing.status == "not_found" and "not in the loaded corpus" in missing.message


def test_http_statuses(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(structured, "get_settings", lambda: Settings(mongodb_uri=""))
    r = client.post("/v1/query", json={"question": "BNS section 103", "pattern": "structured"})
    assert r.status_code == 503 and r.json()["trace"]["reason"] == "missing_setting"
    assert "MONGODB_URI" in r.json()["message"] and r.json()["results"] == []
    r = client.post("/v1/query", json={"question": "section 103", "pattern": "structured"})
    assert r.status_code == 200 and r.json()["status"] == "clarification_needed"


@pytest.mark.parametrize("status_question, expect_calls", [
    ("BNS section 103", 1), ("IPC section 600", 0), ("section 103", 0)])
def test_generation_only_for_ok(monkeypatch, status_question, expect_calls):
    db = FakeDb()
    monkeypatch.setattr(structured, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(structured.semantic, "_database", lambda s: db)
    seen = []

    def fake_generate(result, question, stream=False):
        seen.append([c.chunk_id for c in result.results])
        yield ("text", "answer")
        yield ("final", result)

    monkeypatch.setattr(answer, "generate_events", fake_generate)
    run_pattern(req(status_question), answer_generation=True)  # generate_answer false
    assert seen == []
    run_pattern(req(status_question, generate_answer=True), answer_generation=True)
    assert len(seen) == expect_calls
    if expect_calls:
        assert seen == [["bns:103#0"]]
