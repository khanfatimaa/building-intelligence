"""Bounded agentic RAG: a deterministic, read-only controller over three existing adapters.

It runs only when the user selects the `agentic` mode. The fixed allowlist is structuredLookup,
semanticRetrieve and hydeRetrieve; there is no model call in the controller, no planner, no
retry, no parallel work and no write. At most two adapter calls (at most one HyDE call) run within
a short application deadline. Real retrieved corpus passages are the only evidence; an adapter's
trace and any HyDE hypothetical text are never copied.

Diagnostic command (from the repository root, with an untracked .env):
    uv run python -m building_with_rag.retrieval.agentic "What does BNS section 103 say?"

Output is compact: status, path, calls, adequacy, stop reason, ids and lengths. It never prints
keys, passage text or any hypothetical text.
"""

import argparse
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.signals import classify
from building_with_rag.settings import Settings, get_settings

PATTERN = "agentic"
CONTROLLER = "agentic-rules-v1"
MAX_CALLS = 2
MAX_HYDE_CALLS = 1
MIN_DEADLINE, MAX_DEADLINE = 1.0, 120.0
ALLOWED_ACTS = ("BNS_2023", "IPC_1860")

# Prepared vocabulary-mismatch table (at most five entries): lay phrasing -> statutory terms.
# Terms were confirmed against data/processed/*.jsonl ("personates", "personation", "public servant").
VOCABULARY: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"\bpretend(?:s|ed|ing)?\s+to\s+be\s+(?:a\s+|an\s+)?"
            r"(?:government|public)\s+(?:official|servant|officer|employee)\b",
            re.IGNORECASE,
        ),
        ("personat", "public servant"),
    ),
    (
        re.compile(
            r"\bpretend(?:s|ed|ing)?\s+to\s+be\s+(?:someone|somebody|another\s+person)\b",
            re.IGNORECASE,
        ),
        ("personat",),
    ),
)

# Statuses each allowlisted adapter documents.
ADAPTER_STATUSES = {
    "structuredLookup": {"ok", "not_found", "clarification_needed", "recommendation", "unavailable"},
    "semanticRetrieve": {"ok", "no_results", "unavailable"},
    "hydeRetrieve": {"ok", "no_results", "hyde_unavailable", "unavailable"},
}
ADAPTER_PATTERN = {
    "structuredLookup": "structured",
    "semanticRetrieve": "semantic",
    "hydeRetrieve": "hyde",
}

Adapter = Callable[[QueryRequest, Settings], QueryResult]


def _structured(request: QueryRequest, settings: Settings) -> QueryResult:
    from building_with_rag.retrieval import structured

    return structured.retrieve(request, settings=settings)


def _semantic(request: QueryRequest, settings: Settings) -> QueryResult:
    return semantic.retrieve(request, settings=settings)


def _hyde(request: QueryRequest, settings: Settings) -> QueryResult:
    from building_with_rag.retrieval import hyde

    return hyde.retrieve(request, settings=settings)


ADAPTERS: dict[str, Adapter] = {
    "structuredLookup": _structured,
    "semanticRetrieve": _semantic,
    "hydeRetrieve": _hyde,
}

CLARIFY_MESSAGES = {
    "missing_act": "A section number was found but no act. Say BNS or IPC.",
    "missing_section": "An act was found but no section number.",
    "ambiguous_act": "Both BNS and IPC were named. Ask about one act at a time.",
    "multiple_sections": "More than one section or a range was named. Ask for one section.",
    "lettered_section_unsupported": "Lettered sections (for example 498A) are not supported.",
    "section_out_of_range": "The section number must be between 1 and 999.",
    "invalid_chapter": "The chapter must be a Roman numeral such as VI.",
    "unsupported_intent": "Listing or counting sections is not supported in agentic mode.",
    "no_letters": "The question has no words. Ask a question about the BNS or IPC.",
}
EXAMPLE = " Try: 'What does BNS section 103 say?' or ask a plain question about the BNS or IPC."


class _Stop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def _clamp(seconds: float) -> float:
    try:
        return min(max(float(seconds), MIN_DEADLINE), MAX_DEADLINE)
    except (TypeError, ValueError):
        return 45.0


def _statutory_terms(question: str) -> tuple[str, ...] | None:
    for pattern, terms in VOCABULARY:
        if pattern.search(question):
            return terms
    return None


def _usable(chunk: RetrievedChunk, scope: dict[str, list[str]]) -> bool:
    if not chunk.chunk_id or not chunk.text.strip() or chunk.act not in ALLOWED_ACTS:
        return False
    if scope.get("act") and chunk.act not in scope["act"]:
        return False
    source = chunk.source or {}
    for key in ("status", "access_level"):
        if scope.get(key) and key in source and source[key] not in scope[key]:
            return False
    return True


def retrieve(
    request: QueryRequest,
    *,
    settings: Settings | None = None,
    adapters: dict[str, Adapter] | None = None,
    clock: Callable[[], float] | None = None,
) -> QueryResult:
    """Statuses: ok, partial_result, clarify, unavailable."""
    settings = settings or get_settings()
    clock = clock or time.monotonic
    table = dict(ADAPTERS)
    for name, fn in (adapters or {}).items():
        if name in table:  # only the three allowlisted names can be overridden
            table[name] = fn
    deadline_seconds = _clamp(settings.agentic_deadline_seconds)
    started = clock()
    scope = semantic.effective_scope(request.filters)
    agentic: dict[str, Any] = {
        "controller": CONTROLLER,
        "validation": {"outcome": "pending", "reason": None},
        "path": None,
        "calls": [],
        "call_count": 0,
        "hyde_calls": 0,
        "deadline_seconds": deadline_seconds,
        "stop_reason": None,
    }
    trace: dict[str, Any] = {
        "caller_id": settings.webui_demo_caller_id,
        "ignored_fields": [n for n in ("required_acts",) if getattr(request, n)],
        "effective_filters": scope,
        "agentic": agentic,
    }

    def finish(status: str, stop: str, message: str, results: list[RetrievedChunk] | None = None):
        agentic["stop_reason"] = stop
        return QueryResult(
            pattern=PATTERN, status=status, message=message, trace=trace, results=results or []
        )

    # --- 1. validate before any call ---
    question = request.question.strip()
    outcome = classify(question, request.chapter)
    reason = None
    if outcome.kind == "clarification":
        reason = outcome.reason
    elif outcome.kind == "recommendation" and outcome.reason == "intent_not_executed":
        reason = "unsupported_intent"
    elif not re.search(r"[A-Za-z]", question):
        reason = "no_letters"
    if reason:
        agentic["validation"] = {"outcome": "clarify", "reason": reason}
        return finish("clarify", "clarify", CLARIFY_MESSAGES[reason] + EXAMPLE)
    if scope is None:
        agentic["validation"] = {"outcome": "empty_scope", "reason": "empty_scope"}
        return finish(
            "partial_result",
            "partial_result",
            "No passages can match: the requested filters leave an empty scope. "
            "No retrieval was run.",
        )
    agentic["validation"] = {"outcome": "valid", "reason": None}

    terms = None if outcome.kind == "exact" else _statutory_terms(question)
    agentic["path"] = (
        "exact_lookup"
        if outcome.kind == "exact"
        else "vocabulary_mismatch"
        if terms
        else "ordinary"
    )

    def expired() -> bool:
        return clock() - started >= deadline_seconds

    def call(name: str) -> tuple[QueryResult, list[RetrievedChunk]]:
        """One guarded, inspected adapter call. Raises _Stop; records the call in the trace."""
        if agentic["call_count"] >= MAX_CALLS or (
            name == "hydeRetrieve" and agentic["hyde_calls"] >= MAX_HYDE_CALLS
        ):
            raise _Stop("call_budget_exhausted")
        if expired():
            raise _Stop("deadline_exceeded")
        agentic["call_count"] += 1
        agentic["hyde_calls"] += name == "hydeRetrieve"
        entry: dict[str, Any] = {"adapter": name, "status": None, "passages": 0, "usable": 0}
        agentic["calls"].append(entry)
        # Scope, limit, chapter and caller propagate unchanged; only pattern and generation change.
        adapter_request = request.model_copy(
            update={"pattern": ADAPTER_PATTERN[name], "generate_answer": False}
        )
        try:
            result = table[name](adapter_request, settings)
        except Exception:  # noqa: BLE001 - an adapter failure is a stop, never raw text
            entry["status"] = "error"
            raise _Stop("tool_unavailable") from None
        entry["status"] = result.status
        entry["passages"] = len(result.results)
        if result.pattern != ADAPTER_PATTERN[name] or result.status not in ADAPTER_STATUSES[name]:
            raise _Stop("bad_tool_result")
        if result.status in ("unavailable", "hyde_unavailable"):
            raise _Stop("tool_unavailable")
        usable = [c for c in result.results if _usable(c, scope)]
        if len(usable) != len(result.results):
            raise _Stop("bad_tool_result")
        entry["usable"] = len(usable)
        return result, usable

    def decide(entry_index: int, adequate: bool) -> None:
        agentic["calls"][entry_index]["adequate"] = adequate

    retained: list[RetrievedChunk] = []
    try:
        if outcome.kind == "exact":
            result, usable = call("structuredLookup")
            decide(0, bool(usable))
            if result.status != "ok" or not usable:
                return finish(
                    "clarify",
                    "clarify",
                    "That section is not in the loaded corpus, so nothing is guessed. "
                    "Check the act and section number." + EXAMPLE,
                )
            retained = usable
        else:
            result, usable = call("semanticRetrieve")
            retained = usable
            adequate = bool(usable)
            if terms:
                adequate = any(t in c.text.lower() for c in usable for t in terms)
            decide(0, adequate)
            if expired():
                raise _Stop("deadline_exceeded")
            if not adequate and terms:
                result, usable = call("hydeRetrieve")
                decide(1, bool(usable))
                if usable:
                    retained = usable
                else:
                    return _partial_or_clarify(finish, retained)
            elif not usable:
                return finish(
                    "clarify",
                    "clarify",
                    "No passage in the loaded corpus matched this question within the "
                    "requested scope. Try rephrasing it." + EXAMPLE,
                )
        if expired():
            raise _Stop("deadline_exceeded")
    except _Stop as stop:
        if stop.reason in ("tool_unavailable", "bad_tool_result"):
            return finish(
                "unavailable",
                stop.reason,
                "Agentic retrieval is unavailable: "
                + (
                    "a retrieval step failed."
                    if stop.reason == "tool_unavailable"
                    else "a retrieval step returned an unusable result."
                ),
            )
        # deadline_exceeded or call_budget_exhausted: keep only inspected real passages.
        if retained:
            return finish(
                "partial_result",
                stop.reason,
                f"Stopped early ({stop.reason}); {len(retained)} real passage(s) were retained. "
                "No answer is generated from a stopped run.",
                retained,
            )
        return finish(
            "unavailable",
            stop.reason,
            f"Agentic retrieval stopped early ({stop.reason}) before any usable passage.",
        )
    return finish(
        "ok",
        "ready_for_synthesis",
        f"Returned {len(retained)} real source passage(s) with adequate evidence.",
        retained,
    )


def _partial_or_clarify(finish: Callable[..., QueryResult], retained: list[RetrievedChunk]):
    if retained:
        return finish(
            "partial_result",
            "partial_result",
            f"Only {len(retained)} loosely related real passage(s) were found and they do not "
            "clearly answer the question. No answer is generated.",
            retained,
        )
    return finish(
        "clarify",
        "clarify",
        "No passage in the loaded corpus matched this question. Try rephrasing it." + EXAMPLE,
    )


# --- diagnostic command -----------------------------------------------------------------------


def _print_result(result: QueryResult) -> None:
    agentic = result.trace["agentic"]
    print(f"status:  {result.status}")
    print(f"message: {result.message}")
    print(f"mode:    {result.pattern}")
    print(f"validation: {agentic['validation']}")
    print(f"path: {agentic['path']}")
    for call in agentic["calls"]:
        print(f"  call {call}")
    print(f"call_count={agentic['call_count']} hyde_calls={agentic['hyde_calls']}")
    print(f"stop_reason: {agentic['stop_reason']}")
    for item in result.results:
        print(
            f"  chunk_id={item.chunk_id} section_id={item.section_id} act={item.act} "
            f"text_chars={len(item.text)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic controller diagnostic (compact output).")
    parser.add_argument("question")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        request = QueryRequest(
            question=args.question,
            pattern=PATTERN,
            caller_id=settings.webui_demo_caller_id,
            limit=args.limit,
        )
    except ValidationError as exc:
        print(f"Invalid input: {[e['loc'] for e in exc.errors()]}")
        return 2
    result = retrieve(request, settings=settings)
    _print_result(result)
    return 1 if result.status == "unavailable" else 0


if __name__ == "__main__":
    sys.exit(main())
