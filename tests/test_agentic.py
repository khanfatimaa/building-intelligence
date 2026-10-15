from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk, SemanticFilters
from building_with_rag.retrieval import agentic
from building_with_rag.settings import Settings

SETTINGS = Settings(agentic_deadline_seconds=45)
HYPO = "SECRET HYPOTHETICAL TEXT"
LAY = "What is the offence of pretending to be a government official?"


def chunk(text="Whoever personates a public servant shall be punished.", act="IPC_1860"):
    return RetrievedChunk(chunk_id="ipc:170#0", section_id="ipc:170", act=act, text=text)


class Fake:
    def __init__(self, pattern, status="ok", chunks=None, **fields):
        self.pattern, self.status, self.fields = pattern, status, fields
        self.chunks = [chunk()] if chunks is None else chunks
        self.requests = []

    def __call__(self, request, settings):
        self.requests.append(request)
        return QueryResult(
            pattern=self.pattern, status=self.status, message="x", results=self.chunks,
            trace={"leak": HYPO}, **self.fields,
        )  # fmt: skip


def run(question, adapters, clock=None, **request_fields):
    request = QueryRequest(question=question, pattern="agentic", **request_fields)
    return agentic.retrieve(request, settings=SETTINGS, adapters=adapters, clock=clock)


def adapters(**overrides):
    base = {
        "structuredLookup": Fake("structured"),
        "semanticRetrieve": Fake("semantic", chunks=[chunk("Culpable homicide is the causing of death.")]),
        "hydeRetrieve": Fake("hyde", hyde_hypothetical_text_debug=HYPO),
    }
    return {**base, **overrides}


def test_validation_before_any_call():
    a = adapters()
    result = run("section 103", a)
    assert (result.status, result.trace["agentic"]["call_count"]) == ("clarify", 0)
    assert result.trace["agentic"]["validation"]["reason"] == "missing_act"
    assert "103" not in str(result.trace) and all(not f.requests for f in a.values())


def test_exact_lookup_one_call_and_scope_propagation():
    a = adapters()
    filters = SemanticFilters(act=["BNS_2023", "IPC_1860"])
    result = run("What does IPC section 170 say?", a, filters=filters, limit=3)
    assert (result.status, result.trace["agentic"]["stop_reason"]) == ("ok", "ready_for_synthesis")
    assert [c["adapter"] for c in result.trace["agentic"]["calls"]] == ["structuredLookup"]
    sent = a["structuredLookup"].requests[0]
    assert (sent.filters, sent.limit, sent.pattern, sent.generate_answer) == (filters, 3, "structured", False)


def test_missing_section_is_clarify_without_fallback():
    a = adapters(structuredLookup=Fake("structured", "not_found", chunks=[]))
    result = run("What does BNS section 999 say?", a)
    assert result.status == "clarify" and not a["semanticRetrieve"].requests


def test_ordinary_question_one_semantic_call():
    result = run("What is culpable homicide?", adapters())
    assert result.status == "ok" and result.trace["agentic"]["call_count"] == 1


def test_vocabulary_mismatch_calls_hyde_once_and_hides_hypothetical():
    a = adapters(
        semanticRetrieve=Fake("semantic", chunks=[chunk("Whoever cheats shall be punished.")])
    )
    result = run(LAY, a)
    agentic_trace = result.trace["agentic"]
    assert [c["adapter"] for c in agentic_trace["calls"]] == ["semanticRetrieve", "hydeRetrieve"]
    assert (agentic_trace["hyde_calls"], result.status) == (1, "ok")
    assert result.hyde_hypothetical_text_debug is None and not result.hyde_query_candidates
    assert HYPO not in result.model_dump_json()


def test_vocabulary_direct_adequate_skips_hyde():
    a = adapters()
    a["semanticRetrieve"] = Fake("semantic")
    result = run(LAY, a)
    assert result.trace["agentic"]["call_count"] == 1 and not a["hydeRetrieve"].requests


def test_tool_unavailable_and_bad_result_stop():
    a = adapters(semanticRetrieve=Fake("semantic", "unavailable", chunks=[]))
    assert run("What is theft?", a).trace["agentic"]["stop_reason"] == "tool_unavailable"
    a = adapters(semanticRetrieve=Fake("semantic", chunks=[chunk(act="OTHER")]))
    assert run("What is theft?", a).trace["agentic"]["stop_reason"] == "bad_tool_result"


def test_deadline_stops_without_further_call():
    ticks = iter([0, 0, 100, 100, 100])
    a = adapters(semanticRetrieve=Fake("semantic", chunks=[chunk("Whoever cheats.")]))
    result = run(LAY, a, clock=lambda: next(ticks))
    assert result.trace["agentic"]["stop_reason"] == "deadline_exceeded"
    assert result.status == "partial_result" and not a["hydeRetrieve"].requests


def test_only_allowlisted_adapters_can_be_overridden():
    a = adapters(shellTool=Fake("semantic"))
    assert run("What is theft?", a).status == "ok"
    assert set(agentic.ADAPTERS) == {"structuredLookup", "semanticRetrieve", "hydeRetrieve"}
