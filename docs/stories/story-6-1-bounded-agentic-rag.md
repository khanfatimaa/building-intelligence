# Story 6.1: Bounded Agentic RAG

## Purpose

Add the seventh and last course mode, `agentic` (model ID `rag-agentic`), to the shared registry. It is deterministic application code that runs **only** after a user explicitly selects it (the `agentic` pattern, the `rag-agentic` model ID or `rag_options.pattern`). It is a small read-only controller over three existing adapters, with a fixed path, a call budget and a deadline. It is not a planner, not an automatic router and not a new retrieval method.

The controller may call only this fixed allowlist: `structuredLookup` (`retrieval/structured.retrieve`), `semanticRetrieve` (`retrieval/semantic.retrieve`) and `hydeRetrieve` (`retrieval/hyde.retrieve`). Real retrieved corpus passages stay the only source of evidence, answer text and citations. This story adds a mode additively; `semantic`, `hybrid`, `hybrid-reranked`, `structured`, `decomposition` and `hyde` behave exactly as today, as do `POST /v1/query`, `GET /v1/models`, `POST /v1/chat/completions`, every response shape, citations and Open WebUI streaming. The supplied Open WebUI Functions are not changed.

Not in this story: an LLM planner, free-form or dynamically registered tools, raw database, shell or HTTP tools, writes, browsing, GraphRAG, controller re-ranking, retries, parallel, background or recursive work, memory, automatic routing from any other mode, authorization systems, production security or observability, deployment, broad tests, an evaluation harness, new provider variables, new top-level response fields.

## Prerequisites

- Stories 1.1 to 5.2 are complete. Run `uv run pytest` and `uv run ruff check .` first; if they fail, stop and report. Do not fix other stories here.
- Read first: `docs/architecture.md`, every story in `docs/stories/`, `registry.py`, `contracts.py` (`Pattern`, `QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`), `service.py`, `main.py` (`/v1/query`, `/v1/models`, the chat adapter and its streaming), `settings.py`, `retrieval/signals.py`, `retrieval/structured.py`, `retrieval/semantic.py` (`effective_scope`, `retrieve`), `retrieval/hyde.py`, `generation/context.py`, `generation/answer.py`, `docs/manual-tests.md`, `docs/master-prompt-update-manual-tests.md`, `.env.example`, and the Open WebUI Function files supplied with the course bundle (read only).
- Live checks need the existing `.env` (`MONGODB_URI`, `VOYAGE_API_KEY`, and the generation settings). The vocabulary-mismatch path and answers also need the generation proxy. `hybrid-reranked` in the smoke check needs `RERANK_API_KEY`.
- Use existing `Settings` fields only, plus the one application setting below. Confirm stored act values (`BNS_2023`, `IPC_1860`) and the corpus wording used by the vocabulary table in `data/processed/*.jsonl` before coding.

## Work to do

### 1. Register the mode (additive)

- Add `agentic` to `Pattern` in `contracts.py` and to the `MODES` tuple in `registry.py` (ID `rag-agentic`). `/v1/models` then lists seven IDs. No other contract, route or request field changes.
- Add one application setting, `AGENTIC_DEADLINE_SECONDS` (`agentic_deadline_seconds`, default `45`), to `Settings` and `.env.example` with a one-line comment. It is not a provider variable. The controller clamps the value to 1 to 120 seconds. No other setting is added, renamed or re-defaulted.

### 2. The controller

Add `src/building_with_rag/retrieval/agentic.py` with `retrieve(request, *, settings=None, adapters=None, clock=None)`. `adapters` and `clock` are injectable for tests, and `adapters` can only override the three allowlisted names. Dispatch it from `run_pattern_events` in `service.py` through the same single path as the other modes. Generation runs only for status `ok`, over `QueryResult.results` and the original question, as for structured, decomposition and hyde.

**Fixed rules (code only, no model call in the controller itself):**

1. **Validate before any call.** Use the structured classifier (`signals.classify`) and basic request checks (the chapter, if given, is valid; the question contains letters). A bare section number (no act), a malformed or ambiguous request (both acts, several sections or a range, lettered or out-of-range section, invalid chapter, no letters) or an unsupported `filter`/`aggregation` phrase stops with `clarify`. The trace records only the validation outcome and a reason code, never the question. An empty effective scope (`semantic.effective_scope` is `None`) stops with `partial_result` and no call.
2. **Act-qualified section request** (classifier kind `exact`): call `structuredLookup` once. Inspect its result. A found section with usable passages is adequate. A missing record (`not_found`) is `clarify`, never a guess and never a fallback to semantic search.
3. **Non-exact valid question, prepared vocabulary-mismatch rule.** Keep a short fixed table (five entries at most) in the module that maps lay phrasings to statutory terms, for example "pretending to be a government official" to "personat" and "public servant". The rule applies only when a table phrase matches the question. Then call `semanticRetrieve` once. If no returned usable passage contains a table's statutory term, the direct result fails adequacy and the controller calls `hydeRetrieve` **once** after it. Inspect the HyDE result. At least one usable passage is adequate.
4. **Otherwise (ordinary valid question):** call `semanticRetrieve` once. Adequate means at least one usable real passage under the existing scope.

**Usable passage and inspection (every returned result, before the next step):** the result's `pattern` matches the adapter called, its `status` is one the adapter documents, and the passage has non-empty `text`, a `chunk_id`, an act of `BNS_2023` or `IPC_1860` and sits inside the request's effective scope (`filters.act`, `status`, `access_level`). A result that breaks these checks is a bad tool result: the controller stops and returns `unavailable`. Passages are never edited, scored or re-ranked by the controller.

**Bounds:**

- At most two adapter calls in total and at most one `hydeRetrieve` call. The guard runs before each call; tripping it stops with `call_budget_exhausted`. The fixed path never needs a third call. (`hydeRetrieve` runs its own two internal searches; they do not count as controller calls.)
- Every adapter call receives a copy of the request with only `pattern` and `generate_answer=false` changed, so the effective demo/public scope (`filters`, access level), `limit`, `chapter` and `caller_id` propagate unchanged. `required_acts` stays ignored and is listed in `ignored_fields`.
- The deadline is checked before each call and after each call returns. The controller does no sleeping, retrying, threading or background work; an in-flight adapter is bounded by its own timeouts. Expiry stops with `deadline_exceeded`. The deadline covers the controller only, not the later answer generation.

**Outcomes** (`QueryResult.status`; HTTP 200 except `unavailable`):

| Situation | `status` | `trace.agentic.stop_reason` |
|---|---|---|
| Adequate real evidence | `ok` | `ready_for_synthesis` |
| Real passages retained but not adequate (vocabulary path after HyDE, or empty scope) | `partial_result` | `partial_result` |
| Ambiguous or invalid input; section not found; no usable passage at all | `clarify` | `clarify` |
| Adapter `unavailable`, `hyde_unavailable` or a bad tool result | `unavailable` (503) | `tool_unavailable` or `bad_tool_result` |
| Deadline expired | `partial_result` with retained usable passages, else `unavailable` (503) | `deadline_exceeded` |
| Budget guard tripped | same as deadline | `call_budget_exhausted` |

`partial_result`, `clarify` and `unavailable` make no model call; the `message` is the visible text. Messages are fixed text and never echo the question. `QueryResult.pattern` is `agentic`.

### 3. Evidence boundary and trace

- `QueryResult.results` holds only usable real passages returned by the adapters, unchanged. The controller builds its `QueryResult` from the whitelisted fields `results`, an adapter's status and a short reason code. It never copies an adapter's `trace`, `hyde_hypothetical_text_debug`, `hyde_direct_candidates` or `hyde_query_candidates`; those stay empty/null for `agentic`.
- The HyDE hypothetical text must not appear anywhere in the controller's trace, message, context, prompt, answer, citation, log or diagnostic output.
- `trace["agentic"]` (additive, under the existing `trace` dict, because the shared generation step overwrites `trace["reason"]`) records: `controller` name, `validation` (`outcome`, `reason`), `path` (`exact_lookup`, `ordinary`, `vocabulary_mismatch` or `null`), `calls` (adapter name, adapter status, passage count, usable count, adequacy decision, in order), `call_count`, `hyde_calls`, `deadline_seconds` and `stop_reason`. Beside it keep `caller_id`, `ignored_fields` and `effective_filters`. It holds no raw request text, no credential, no passage text and no hypothetical text.
- No new top-level `QueryResult` field and no new `RetrievedChunk` field. If a shared generation or context helper assumes the older modes, change it additively without changing their output.

### 4. Diagnostic command

Add a `main()` to the module in the style of `hyde.py` (compact: status, path, calls, adequacy, stop reason, ids, act and text lengths; never keys, passage text or any hypothetical text; non-zero exit for `unavailable`):

```
uv run python -m building_with_rag.retrieval.agentic "What does BNS section 103 say?"
```

### 5. Tests and docs (light)

- Update the existing assertions that count six modes or list six IDs (for example in `tests/test_smoke.py`) to seven. `tests/test_semantic.py`, `tests/test_hybrid.py`, `tests/test_rerank.py` and the other mode tests pass unmodified.
- Add one small `tests/test_agentic.py` with fake adapters covering only: the allowlist, validation-before-call, the call budget (including one HyDE call at most), scope propagation, the deadline stop with a fake clock, and that a hypothetical text never reaches the result. No broad suite and no evaluation harness.
- `docs/architecture.md`: add `agentic` / `rag-agentic` to the modes table and an "Agentic RAG" section (the fixed rules and outcome table above, the trace shape, the diagnostic command, the new setting), update the Commands and Limitations text, and replace "agentic RAG" in "Out of scope" with the remaining exclusions. Limitations to state: the path is fixed and the user must pick the mode; the vocabulary table is tiny and only for prepared phrasings; adequacy is a weak floor, not proof of relevance; the deadline is cooperative and cannot interrupt an in-flight adapter; HyDE adds a model call and a second embedding call and may hit the Voyage free-key rate limit (no retry); no memory, retries or re-ranking by the controller.
- `docs/manual-tests.md`: add a `## Story 6.1 — Bounded Agentic RAG` section following `docs/master-prompt-update-manual-tests.md`, with the checks below.

## Completion checks

Run `uv run pytest` and `uv run ruff check .`; both must pass. Then, with the API running (`uv run uvicorn building_with_rag.main:app --port 8000`), call `/v1/query` with `"pattern": "agentic"` and `generate_answer` false. Wait about 20 seconds between commands that call Voyage. For each case inspect the call sequence, `trace.agentic`, the safe outcome and that `results` holds only real corpus passages.

1. **Act-qualified lookup.** "What does BNS section 103 say?" Expect one `structuredLookup` call, `call_count` 1, `ok` / `ready_for_synthesis`, the BNS section 103 passages. Also try "What does BNS section 9999 say?": expect `clarify`, no semantic fallback.
2. **Ordinary explanation.** "What is culpable homicide?" Expect one `semanticRetrieve` call, an adequate decision, `ok` / `ready_for_synthesis`, real passages.
3. **Prepared vocabulary mismatch.** "What is the offence of pretending to be a government official?" Expect path `vocabulary_mismatch`, `semanticRetrieve` first; `hydeRetrieve` once only if the direct result lacks the statutory terms; `call_count` at most 2. Report honestly which happened. Confirm `hyde_hypothetical_text_debug` is null and the hypothetical text is nowhere in the response.
4. **Ambiguous input.** "section 103" (and optionally "BNS and IPC section 302"). Expect `clarify`, `call_count` 0, a trace with only the validation outcome and reason, and no question text.
5. **Deadline.** Set `AGENTIC_DEADLINE_SECONDS=1` temporarily and repeat check 3 or 2: expect `deadline_exceeded` in the trace and no further adapter call. Restore the value.
6. **Open WebUI stream.** `curl -N` `rag-agentic` with `"stream": true` and the check 1 or 2 question: expect SSE with a grounded draft, confidence and sources citing real passages, ending with `data: [DONE]`. In Open WebUI, confirm the same text and citations. Do not edit the supplied Functions.
7. **Seven-mode smoke.** `GET /v1/models` lists all seven IDs. Through `/v1/query` and then chat, run one short question per mode: `semantic`, `hybrid`, `hybrid-reranked`, `structured`, `decomposition`, `hyde` and `agentic`. Confirm only the contract: HTTP 200, each response's own `pattern`, its usual status and result shape.

## Handover

**The implementer fills in:**

- Files created or changed.
- The controller as implemented: validation reasons, the vocabulary table entries, the usable-passage and adequacy rules, the stop reasons and the status mapping.
- Where the hypothetical text is excluded (which fields are whitelisted).
- Commands actually run, with results; which live checks ran and which are pending, and why.
- Commands for you to run: the diagnostic command and the `/v1/query` and chat curls in `docs/manual-tests.md`.
- Deviations and open issues.

**You fill in after running it:** what each of checks 1 to 7 showed (call sequence, trace, outcome, section ids, whether HyDE ran and helped, the seven-mode smoke result, the Open WebUI display), and the known limitations listed above.

Course end: all seven modes are available behind the same `run_pattern`, `QueryResult` and `RetrievedChunk`, additively.

## Handover record (implementer)

- **Files created or changed:** `src/building_with_rag/retrieval/agentic.py` (new), `contracts.py` (`Pattern`), `registry.py`, `settings.py` and `.env.example` (`AGENTIC_DEADLINE_SECONDS`), `service.py` (dispatch; generation for `ok` only), `tests/test_agentic.py` (new, 9 tests), `tests/test_smoke.py` (seven modes), `docs/architecture.md`, `docs/manual-tests.md`, this record.
- **Controller as implemented:** validation reasons are the structured classifier's (`missing_act`, `missing_section`, `ambiguous_act`, `multiple_sections`, `lettered_section_unsupported`, `section_out_of_range`, `invalid_chapter`) plus `unsupported_intent` and `no_letters`. Vocabulary table has two entries ("pretending to be a government official" -> `personat`, `public servant`; "pretending to be someone/another person" -> `personat`), confirmed in the corpus. Adequacy: at least one usable passage; on the vocabulary path the direct result must also contain a statutory term, else HyDE runs once (any usable HyDE passage is adequate). Exceptions raised by an adapter become `tool_unavailable`.
- **Hypothetical text exclusion:** the result is built only from `results`, a status and fixed messages; adapter traces and all `hyde_*` fields are dropped (covered by a test).
- **Commands actually run:** `uv run pytest` (126 passed) and `uv run ruff check .` (clean). Live `/v1/query`, chat, Open WebUI and the seven-mode smoke checks were **not run** (need `.env` credentials and the running API).
- **Commands for you to run:** the diagnostic command in `docs/architecture.md` and the Story 6.1 curls in `docs/manual-tests.md`.
- **Deviations and open issues:** none. A deadline that expires after an otherwise adequate final call still stops with `deadline_exceeded` (`partial_result`), per the story.
