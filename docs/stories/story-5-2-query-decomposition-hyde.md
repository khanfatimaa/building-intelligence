# Story 5.2: Query Decomposition and HyDE

## Purpose

Make the last two modes real: `decomposition` (`rag-decomposition`) and `hyde` (`rag-hyde`). Both are small retrieval extensions placed in front of the retrieval and grounded-answer path that already exists.

- **Decomposition** handles a compound comparison question, for example "Compare the punishment for murder under the BNS with the punishment for murder under the IPC". One bounded model step splits it into one to three clear subquestions. Each accepted subquestion is **required**: it must find real corpus evidence of its own. If one does not, the mode says so (`partial_answer` or `clarify`) instead of writing a complete comparison.
- **HyDE** handles a vocabulary mismatch, where the user's words differ from the statute's (a lay "pretending to be a government official" against statutory "personates a public servant"). One bounded model step writes a short **hypothetical retrieval text**; that text is embedded and used only to find real passages. It is never an answer source, a citation or evidence.

Everything else is shared with the other four modes: `QueryRequest`/`QueryResult`/`RetrievedChunk`, the registry, `/v1/query`, the chat adapter, context assembly, grounded generation, citations, confidence and Open WebUI streaming. No new endpoint, UI, setting, provider variable or application. These are not agents, not automatic routing, not graph construction.

The mode is chosen only by the `decomposition` / `hyde` pattern, the `rag-decomposition` / `rag-hyde` model ID or `rag_options.pattern`. The application never picks or switches modes.

Not in this story: automatic mode choice or fallback between modes, agentic or open-ended loops (one planning or one hypothetical-text call, no retry), more than three subquestions, follow-up or conversation memory, a general evaluation harness, GraphRAG or any graph, new settings or renamed provider variables, changes to the corpus, chunking, embeddings or indexes, re-ranking or hybrid inside these modes, production security.

## Prerequisites

- Stories 1.1 to 5.1 are complete. First run `uv run pytest` and `uv run ruff check .`; if they fail, stop and report. Do not fix other stories here.
- Read first: `docs/architecture.md`, every story in `docs/stories/`, `contracts.py` (`QueryResult.subquestions`, `SubquestionEvidence`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug` already exist from Story 1.1), `registry.py`, `service.py`, `main.py` (the `/v1/query` route and chat adapter), `settings.py`, `retrieval/semantic.py` (`retrieve`, `embed_query`, `search`, `effective_scope`, `Unavailable`), `retrieval/signals.py` (act detection), `generation/context.py` (`MAX_PASSAGES`), `generation/answer.py` (`call_model`, `GenerationFailure`, `generate_events`, and that `_finish` copies the retrieval result so extra fields survive), `docs/manual-tests.md`, `docs/master-prompt-update-manual-tests.md`, and `.env.example`.
- The database is loaded and the vector index is ready (`uv run python -m building_with_rag.retrieval.semantic --check`). Both modes use **only** the existing settings through `Settings`: `MONGODB_URI`, `MONGODB_DB_NAME`, `VOYAGE_API_KEY` (embeddings, `voyage-3.5`, 1,024 dimensions), `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME` (the LiteLLM proxy) and `WEBUI_DEMO_CALLER_ID`. Do not rename, add, remove or re-default any setting or edit `.env.example`. No re-rank or keyword index is needed.
- A live check needs those values in the untracked `.env`. A free Voyage key is rate limited: wait at least 20 seconds between steps that call Voyage (decomposition makes up to three embedding calls per question, HyDE two).

## Contracts to preserve

Do not rename, replace or remove: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `SubquestionEvidence`, `SemanticFilters`, `GenerationResult`, `/v1/query` and its response model, the registry (six modes, `rag-<pattern>` IDs; `rag-decomposition` and `rag-hyde` already exist), `/v1/models`, the chat adapter with its streaming and `CAPSTONE_API_KEY` behavior, context assembly and citation rules, the `semantic`, `hybrid`, `hybrid-reranked` and `structured` behavior and outputs, MongoDB collections and validators, or `.env.example`. Changes are additive only: **no new contract field**, because every field this story fills already exists.

- `tests/test_semantic.py`, `tests/test_hybrid.py`, `tests/test_rerank.py` and `tests/test_structured.py` pass unmodified.
- `unavailable` stays HTTP 503 with empty `results` and no generation; every other status here is HTTP 200. `generation` outcomes never change the retrieval `status`.
- The generation system prompt, validation checks and confidence values (`HIGH_CONFIDENCE`, `LOW_CONFIDENCE`) are reused unchanged.

## Work to do

### 1. Decomposition (`retrieval/decomposition.py`)

`retrieve(request, *, settings=None, db=None, embed=None, call=None) -> QueryResult`, injectable like semantic so tests pass fakes.

1. **Settings check first.** Empty `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL` or `GENERATION_API_KEY` gives `unavailable` (503) with `trace["reason"]="missing_setting"` and a message naming only the variable. No model, Voyage or MongoDB call has been made at that point.
2. **Planning step: one model call, no retry.** Reuse `generation.answer.call_model` (the existing LiteLLM settings, bounded timeout). The system prompt treats the question as data, asks for one JSON object `{"subquestions": ["...", ...]}` with one to three self-contained subquestions that each name the act (BNS or IPC) they concern, and forbids anything else. Provider failure (timeout, connection, HTTP) is `unavailable` (503, reason such as `planner_failed`), never guessed.
3. **Validate the plan in code.** Accept only a list of one to three strings that are, after trimming and whitespace collapsing: non-empty, at most `MAX_SUBQUESTION_CHARS` (named constant, about 300), **unique** (case-insensitive after normalisation), inside the corpus scope (mention no act other than BNS or IPC, and share at least one significant word with the original question, so unrelated text is out of scope), and free of instruction-like or markup text (for example URLs, code fences, "ignore previous"). Anything else is `clarify` (HTTP 200, empty `results`, no retrieval) with `trace["reason"]` such as `plan_empty`, `plan_malformed`, `plan_too_many`, `plan_duplicate` or `plan_out_of_scope`, and a message asking the user to restate the comparison as one or two plain questions. The rejected output is never retrieved, never used as evidence and not echoed beyond the reason.
4. **Retrieve evidence for every accepted subquestion.** For each, in order, call the existing semantic retrieval (embed with Voyage `voyage-3.5`, `input_type="query"`, the existing vector index, the request's `filters` narrowing as today). If a subquestion names exactly one act, narrow that step's `act` filter to it (an empty intersection with the request's filters is `no_evidence`, reason `scope_excludes_request`, with no search). Per-step limit is `min(request.limit, MAX_PASSAGES // number_of_subquestions)` so the union never exceeds the context budget and no subquestion's evidence is silently cut. A semantic failure on any step is `unavailable` (503), with the step number in the reason and no partial result presented as complete.
5. **Per-step evidence rule.** A subquestion is `evidenced` only if its step returned at least one real, resolved passage in scope **and** at least one of them passes a small deterministic lexical check against the subquestion (reuse the idea and the named threshold style of `SUPPORT_MIN_OVERLAP`; keep it a named constant in the new module and do not change generation). Otherwise it is `no_evidence` with `reason` `no_passages` or `weak_overlap`. Because semantic search has no score threshold, this is a weak floor, not proof of relevance, and the docs must say so.
6. **Status.**

| `status` | When | HTTP |
|---|---|---|
| `ok` | every accepted subquestion is `evidenced` | 200 |
| `partial_answer` | at least one `evidenced` and at least one `no_evidence` | 200 |
| `clarify` | the plan was rejected, or no subquestion is `evidenced` (`trace` reason `no_subquestion_evidence`) | 200 |
| `unavailable` | missing setting, planner failure, retrieval failure | 503 |

7. **Result shape (additive, existing fields only).**
   - `QueryResult.subquestions`: one `SubquestionEvidence` per accepted subquestion, in plan order (`subquestion`, `status` `evidenced`/`no_evidence`, that step's `results`, `reason` for `no_evidence`).
   - `QueryResult.results`: the union of every real passage from every step, deduplicated by `chunk_id`, in plan order then step rank, so each passage appears once. Real passages from a `no_evidence` step are kept here and in its step so they can be inspected, but they are weak neighbours, not support (see the answer boundary).
   - `trace["decomposition"]`: `planner` (model name, never a key or URL), `accepted` (the subquestions), `rejected_reason` when clarified, per-step `{subquestion, status, reason, retrieval_status, passages, chunk_ids, act_filter}` (no passage text), `evidenced_count`, `required_count`, and `reason` (`partial_evidence` or `no_subquestion_evidence` when not `ok`). Keep the partial/insufficient reason under `trace["decomposition"]`, because the shared generation step overwrites top-level `trace["reason"]`. Also fill `caller_id` and `ignored_fields`.
   - `message`: for `partial_answer` and `clarify`, plain readable text listing each subquestion as evidenced (with its act-qualified sections, such as "BNS section 103") or missing, and stating that no complete comparison is given. For `ok`, a short step summary.
8. **Answer boundary.** Generation runs only with `generate_answer=true` **and** status `ok`, and goes through the existing `generate_events` over exactly `QueryResult.results`. For `partial_answer`, `clarify` and `unavailable` **no model call is made** and the `message` is the visible text; a complete comparison is never synthesized, and a later story may add a partial synthesis. For `ok`, emit one short leading `text` event listing the subquestions and their evidence counts, then the unchanged generation events (DRAFT, checks, confidence, sources). The answer must still cite real passages, and citation and confidence rules stay as they are.

### 2. HyDE (`retrieval/hyde.py`)

`retrieve(request, *, settings=None, db=None, embed=None, call=None) -> QueryResult`, injectable the same way.

1. **Settings check first.** Same four settings as above. Missing configuration gives `unavailable` (503), `trace["reason"]="missing_setting"`, the variable named, and **no hypothetical text is invented** and no comparison is run.
2. **Hypothetical text: one model call, no retry.** Reuse `call_model`. The prompt asks for one JSON object `{"hypothetical_text": "..."}`: one short passage, in the plain legal style of the BNS/IPC, that would answer the question. It is a search aid only. Provider failure is `unavailable` (503, `hypothetical_generation_failed`).
3. **Validate.** The text must be a non-empty string after trimming, at most `MAX_HYPOTHETICAL_CHARS` (named constant, about 1,200) and not code, markup or a bare echo of the question. Empty or malformed output is `hyde_unavailable` (HTTP 200, empty `results`, empty candidate lists, `trace["reason"]` such as `hypothetical_empty` or `hypothetical_malformed`). It does **not** fall back to a direct-only result under this mode, and the message says no HyDE comparison ran.
4. **Two real retrievals, same scope and index.** (a) Direct: the question through the existing semantic retrieval, giving `hyde_direct_candidates`. (b) Hypothetical: embed the hypothetical text once with the existing retrieval model (`embed_query`, `voyage-3.5`, 1,024 dimensions) and run the same `$vectorSearch` with the same scope, giving `hyde_query_candidates`. Each list holds at most `request.limit` real `RetrievedChunk`s with their own route's score. Retrieval failure in either route is `unavailable` (503). The hypothetical text never becomes a filter, a MongoDB predicate or a `section_id`.
5. **Final passages.** `QueryResult.results` is the deduplicated union of the two candidate lists, interleaved by route rank (direct 1, hypothetical 1, direct 2, ...), skipping a `chunk_id` already taken, ties broken by `chunk_id`, cut to `request.limit`. Every item is a real corpus passage. `trace["hyde"]` records `direct_count`, `hypothetical_count`, `overlap`, `direct_only`, `hypothetical_only` (chunk ids), per-passage provenance (`both`, `direct`, `hypothetical`) and the hypothetical text length only.
6. **Hypothetical text containment.** The text appears **only** in `QueryResult.hyde_hypothetical_text_debug` (so `/v1/query` can show it for inspection). It must not appear in `message`, `trace`, logs, context assembly, any prompt to the answer generator, the label map, citations, supporting passages, or Open WebUI text. The `/v1/query` `message` says it is a retrieval aid, not evidence.
7. **Status.** `ok` (200), `no_results` (200, empty scope, with no model or Voyage call), `hyde_unavailable` (200), `unavailable` (503).
8. **Answer boundary.** With `generate_answer=true` and status `ok`, the same `QueryResult` goes through `generate_events` over `QueryResult.results` only, with the user's original question. For every other status no model call is made and the `message` is the visible text. Confidence, citations, DRAFT/sources rendering and Open WebUI streaming are unchanged.

### 3. Make the modes selectable (`service.py`)

- Change `run_pattern_events` so `decomposition` and `hyde` run their `retrieve` in the same single path as the other modes (one retrieval operation, then at most one `generate_events` operation over its result). The existing `generate_answer` and `answer_generation` condition stays. Generation runs for `ok` only in both new modes (decomposition `partial_answer`/`clarify` and HyDE `hyde_unavailable` show their `message`). Other patterns behave exactly as today.
- `rag-decomposition` and `rag-hyde` are already in the registry and `/v1/models`; add or rename nothing. `/v1/chat/completions` already maps the model ID to the same `QueryRequest`, so Open WebUI streams these modes through the existing adapter. Check that nothing in `main.py` or `answer.py` assumes the pattern is one of the first four, and change it only additively.
- The `not_implemented` branch in `run_pattern_events` is now unreachable for registered modes; keep it as a defensive default.

### 4. Diagnostic commands

```
uv run python -m building_with_rag.retrieval.decomposition "<question>" [--limit N]
uv run python -m building_with_rag.retrieval.hyde "<question>" [--limit N]
```

Each calls the same `retrieve` function `run_pattern` uses and prints compact output: status, message, the reason, and per route or per subquestion the status, counts, `chunk_id`, `section_id`, `act` and text length. Decomposition prints the accepted subquestions (they are short model-written questions, not passages). HyDE prints the hypothetical text **length and a "debug field present" flag only**. No passage text, keys, URLs or vectors. Exit non-zero for `unavailable`.

### 5. Tests (a few, offline)

No network, MongoDB, Voyage, model provider or server. Fake the model `call`, the `embed` function and `db`. Cover only:

- **Decomposition plan validation:** a good two-subquestion plan is accepted; empty, malformed JSON, four items, a duplicate pair and an out-of-scope item each give `clarify` with the right reason and zero retrieval calls.
- **Decomposition evidence:** all steps evidenced gives `ok`, `subquestions` in plan order, and top-level `results` equal to the deduplicated union (a passage returned by two steps appears once at the top level and in both steps); one step with no passages gives `partial_answer`; no evidenced step gives `clarify`; `trace["decomposition"]` carries the reason; a step naming one act is narrowed to that act.
- **HyDE:** a good fake text gives `hyde_direct_candidates`, `hyde_query_candidates`, deduplicated `results` and the text only in `hyde_hypothetical_text_debug`; empty and malformed text give `hyde_unavailable` with empty results and no embedding of the hypothetical; the hypothetical text appears nowhere in `message`, `trace` or the context given to generation.
- **Configuration and generation:** a missing `GENERATION_API_KEY` (and `VOYAGE_API_KEY`) is `unavailable` naming the variable with no model call; `run_pattern` with generation on makes one generation operation for `ok` and none for `partial_answer`, `clarify` or `hyde_unavailable`; a streamed chat request for each mode ends with `data: [DONE]`.

Existing tests that asserted `decomposition` or `hyde` is `not_implemented` (`tests/test_smoke.py`, `tests/test_generation.py`) are changed to assert that all six registered patterns dispatch to a real retrieval function (faked); no `not_implemented` assertion remains. No broad suite and no evaluation harness.

### 6. Documentation

- `docs/architecture.md`: keep existing text. Add a "Query decomposition" and a "HyDE" section (or one combined "Query decomposition and HyDE" section with two subsections) covering: the flow and bounds (one model call, one to three subquestions, per-step limit, no retry, no loop); the validation and `clarify` rules; the per-step evidence rule and statuses table; the result fields (`subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`, `trace["decomposition"]` and `trace["hyde"]`); the containment rule that hypothetical text is never evidence, a citation, context or chat text; the answer boundary (generation for `ok` only, no complete comparison without per-subquestion evidence); the diagnostic **commands**; and **Limitations** (the evidence rule is a weak lexical floor over a search with no score threshold, so "evidenced" is not proof of relevance; planning and hypothetical text come from one model call that can be wrong or leak the question's bias; a `partial_answer` is an evidence report, not a synthesized answer; HyDE can pull in plausible but wrong passages, it embeds the text with the existing query-type embedding, and it adds one model call and a second embedding call, so latency, cost and rate-limit pressure go up; at most three subquestions; the user chooses the mode; no memory; both depend on the generation proxy). Update only the lines that call these modes placeholders, including the "remaining modes (`decomposition`, `hyde`) still reply that they are not implemented" line under Using Open WebUI, and the Fixed choices note if it implies they are not built.
- `docs/manual-tests.md`: add a `## Story 5.2 — Query Decomposition and HyDE` section following `docs/master-prompt-update-manual-tests.md`: a `/v1/query` curl for `decomposition` with the prepared BNS-versus-IPC question, a `clarify` or partial edge curl, a `/v1/query` curl for `hyde` with the prepared question, a missing-configuration note or edge, and a `rag-decomposition` and a `rag-hyde` chat curl each with `"stream": true` ending in `data: [DONE]`. Update the earlier lines that list `decomposition` and `hyde` as `not_implemented` (the Story 1.1 curls and the note under them). Do not run them as part of this item.

## Token-safe checks

Every check prints only statuses, counts, ids, headings, lengths and booleans. Never print keys, URLs, vectors or full passage text in scripts, logs, tests or the handover. For the hypothetical text, print its length and whether it is present. Subquestions are short model-written questions and may be shown. If a check needs passage text, compare it in code and print booleans.

## Completion checks

### Checks the implementer runs (no services)

1. `uv sync`, `uv run ruff check .` and `uv run pytest` pass, including the item 5 tests and the unmodified semantic, hybrid, re-rank and structured tests.
2. With the model or Voyage settings empty, both modes give `unavailable` naming the variable with empty `results`, and no hypothetical text or plan is produced.
3. Reading the code shows one dispatch path in `run_pattern_events`, at most one planning call and one hypothetical-text call per request with no retry or loop, `generate_events` consuming only `QueryResult.results`, and the hypothetical text referenced only by the debug field.
4. `.env.example`, `data/raw/`, `data/processed/`, the indexes, the registry, `contracts.py` and `Settings` are unchanged.

### Direct checks (need the populated database, Voyage and the generation proxy)

Run if services are available; otherwise report them **pending** with what is missing. Never claim a pass that was not run. Inspect the `/v1/query` evidence **before** judging the chat answer.

1. **Decomposition, prepared comparison.** Question: "Compare the punishment for murder under the BNS with the punishment for murder under the IPC." Through the diagnostic command and `/v1/query` (`pattern: "decomposition"`, `generate_answer` false, then true). Inspect: the accepted subquestions (one to three, unique, each naming an act), each step's status, reason and `section_id`s (expect a BNS passage for the BNS step and an IPC passage for the IPC step; confirm against the corpus rather than assuming), that top-level `results` is their deduplicated union, and `trace["decomposition"]`. With `generate_answer` true and status `ok`, confirm the answer cites passages from both acts. Then show it in Open WebUI (**Building with RAG**, **decomposition**).
2. **Decomposition not complete.** Ask a comparison where one side has no evidence, for example "Compare the punishment for murder under the BNS with the punishment for murder under the Motor Vehicles Act." Expect `clarify` (out-of-scope subquestion) or `partial_answer`, a message naming what is missing, and no complete comparison and no model call for the answer. Report what actually happened.
3. **HyDE, vocabulary mismatch.** Prepared question: "What is the offence of pretending to be a government official?" (statutory wording is closer to "personates a public servant"). Through the diagnostic command and `/v1/query` (`pattern: "hyde"`). Inspect `hyde_direct_candidates`, `hyde_query_candidates` and `results` (real `section_id`s only), `hyde_hypothetical_text_debug` present, and whether the hypothetical route added at least one passage the direct route missed. Report the observation honestly; if it did not help, record that, and optionally try one alternative prepared question, without tuning to get a better result. Confirm the hypothetical text is absent from the answer, the sources and the chat text. Then show it in Open WebUI (**hyde**).
4. **Smoke all six modes.** One short question each, through `/v1/query` and then Open WebUI (select each mode in the RAG-options chip): `semantic` ("What is the punishment for murder under the BNS?"), `hybrid` ("What is culpable homicide?"), `hybrid-reranked` (the same, needs `RERANK_API_KEY`), `structured` ("What does BNS section 103 say?"), `decomposition` (check 1), `hyde` (check 3). Confirm only the contract: `GET /v1/models` lists the six IDs; each `/v1/query` returns HTTP 200 with its own `pattern`, a sensible `status` and `results` shaped as before (hybrid fused fields, re-rank fields, `source` metadata, `score` null for structured); each chat reply streams a `DRAFT` label, `Checks passed. Confidence: high.` and sources (or the mode's own message) and ends with `data: [DONE]`. Wait 20 seconds between Voyage calls. This is a contract check on six questions, not an evaluation.

## Handover

**The implementer fills in:**

- **Files created or changed.**
- **Decomposition as implemented:** the planner prompt contract, validation rules and reasons, the per-step limit and evidence rule, the statuses and when generation runs.
- **HyDE as implemented:** the hypothetical-text contract and validation, how the two routes are searched, the interleaving rule, and where the hypothetical text appears.
- **Commands actually run,** with results. State which direct checks ran and which are pending and why.
- **Commands for you to run:** the two diagnostic commands, the `/v1/query` curls and the two chat curls.
- **Deviations and open issues.**

**You fill in after running it:**

- **Checks 1 to 4:** what you saw (subquestions, step statuses, section ids, the HyDE candidate lists and whether the hypothetical route helped, the six-mode smoke results, the Open WebUI display).
- **Known limitations:** the evidence rule is weak and not proof of relevance; a `partial_answer` is an evidence report; HyDE can retrieve plausible but wrong passages and adds latency and cost; at most three subquestions; the user chooses the mode; no follow-up memory.

Session 5 ends here: all six modes are available behind the same `run_pattern`, `QueryResult` and `RetrievedChunk`, additively.

## Out of scope

Automatic mode choice or routing, fallback between modes, agentic or open-ended loops, retries or a second planning pass, follow-up or conversation memory, a general evaluation harness or benchmark, GraphRAG or graph construction, new settings, renamed provider variables or a new provider, changes to embeddings, chunking, the corpus or indexes, multi-user authorization and production security, a new endpoint, UI or second application.

## Handover record (implementer)

- **Files created or changed:** `src/building_with_rag/retrieval/decomposition.py`, `src/building_with_rag/retrieval/hyde.py`, `src/building_with_rag/retrieval/signals.py` (one small public `detect_act` wrapper, classifier behavior unchanged), `src/building_with_rag/service.py` (dispatch for both modes; generation for `ok` only in the new modes and structured), `tests/test_decomposition_hyde.py`, `tests/test_smoke.py` and `tests/test_generation.py` (the `not_implemented` assertions replaced), `docs/architecture.md`, `docs/manual-tests.md`, this record. `contracts.py`, `registry.py`, `settings.py` and `.env.example` are unchanged.
- **Decomposition as implemented:** one planning call through `generation.answer.call_model`, prompt asks for `{"subquestions": [...]}`. Reasons: `plan_empty`, `plan_malformed`, `plan_too_many`, `plan_duplicate`, `plan_out_of_scope` (over 300 characters, markup or instruction-like text, a capitalised Act/Code/Sanhita other than BNS/IPC, or no key word shared with the question), all `clarify` with no retrieval. Per-step limit `min(limit, 8 // subquestions)`; a subquestion naming exactly one act is narrowed to it. Evidence rule: at least one passage and one passage with at least 40% of the subquestion's key words (`EVIDENCE_MIN_OVERLAP`). Generation only for `ok`; `partial_answer` and `clarify` show the message with no model call.
- **HyDE as implemented:** one call returning `{"hypothetical_text": "..."}`; empty, non-JSON, over 1,200 characters, markup or an echo of the question is `hyde_unavailable`. Direct and hypothetical retrievals both use `semantic.retrieve` (the hypothetical one by embedding the hypothetical text instead of the question), interleaved by rank and cut to the limit. The text appears only in `hyde_hypothetical_text_debug`; the trace keeps its length.
- **Commands actually run:** `uv run pytest` (116 passed, 28 new or changed) and `uv run ruff check .` (clean). Live diagnostic commands against the real database and proxy: decomposition on the prepared BNS-versus-IPC question gave `ok` with three subquestions, each evidenced (BNS 103/104, IPC 302/303), union of 4 passages. HyDE on the prepared question gave `ok`: direct and hypothetical lists of 5 each overlapping on 4, final `ipc:170`, `bns:204`, `ipc:171`, `bns:205`, `ipc:140`.
- **Pending (not run):** `/v1/query` and chat curls, Open WebUI display, check 2 (the Motor Vehicles Act comparison live), and the six-mode smoke check (`hybrid-reranked` also needs `RERANK_API_KEY`). Only unit tests covered the `clarify`/`partial_answer` paths.
- **Commands for you to run:** the two diagnostic commands in `docs/architecture.md`, and the curls under Story 5.2 in `docs/manual-tests.md`.
- **Deviations and open issues:**
  - HyDE's hypothetical-text call returned empty content on 3 of about 6 live attempts (the proxy model `openai/gpt-oss-20b` intermittently returns an empty message), giving `hyde_unavailable` with reason `hypothetical_empty`. With no retry allowed this is expected behavior, but it will show up in class; just run the question again.
  - A Voyage rate-limit hit (`embedding_rate_limited`) appeared when HyDE ran about 20 seconds after another command; it is reported as `unavailable`, not retried.
  - The decomposition planner can add a third "how do they differ" subquestion naming both acts; it is not act-narrowed and is evidenced by the other steps' nearest passages, which the weak evidence rule cannot tell apart.
  - On the prepared HyDE question the direct search already returned the right sections (IPC 170, BNS 204); the hypothetical route added only one passage (`ipc:167`) outside the final list. Not a demonstration that HyDE helps here.
