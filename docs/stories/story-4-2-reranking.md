# Story 4.2: Hybrid Re-ranking

## Purpose

Make the `hybrid-reranked` mode (`rag-hybrid-reranked`) real. Hybrid (Story 4.1) fuses two rankings by rank only, so a passage that answers the question can sit below a passage that merely shares words or sounds similar. Re-ranking takes a small, bounded set of hybrid candidates and has a re-ranking model read each candidate against the question, then keeps the best few. It is one extra step between hybrid retrieval and context assembly.

Everything else is shared with `semantic` and `hybrid`: the corpus, the `RetrievedChunk` evidence shape, context assembly, grounded generation, citations, confidence, `/v1/query`, the chat adapter, Open WebUI streaming, and the `.env.example` contract. There is no new endpoint, UI, answer path or application. `semantic` and `hybrid` must behave exactly as they do today.

A re-rank score is a relevance ranking from one model, not proof that a passage is correct or sufficient. Nothing in this story may present it as one.

Not in this story: automatic mode choice, an evaluation harness or full benchmark, query rewriting, HyDE, decomposition, structured routing, GraphRAG, agentic behavior, a second re-ranking pass, a local or alternative re-ranker, a new provider contract, weight tuning, a new HTTP framework or SDK, or any change to the corpus, chunking, embeddings or indexes.

## Prerequisites

- Stories 1.1, 2.1, 2.2, 2.3, 3.1, 3.2 and 4.1 are complete. First run `uv run pytest` and `uv run ruff check .`; if they fail, stop and report. Do not fix other stories here.
- Read first: `docs/architecture.md` (Fixed choices, Semantic retrieval, Grounded answer generation, Streamed answers and confidence, Hybrid retrieval), every story in `docs/stories/`, `contracts.py`, `registry.py`, `service.py`, `main.py` (the chat adapter), `settings.py`, `retrieval/semantic.py`, `retrieval/hybrid.py` (`retrieve`, `fuse`, `route_candidates`, the diagnostic command), `generation/context.py` (`assemble` keeps retrieval order, `MAX_PASSAGES`, `MAX_CONTEXT_CHARS`), `generation/answer.py` (`generate_events` consumes `result.results`), `docs/manual-tests.md`, `docs/master-prompt-update-manual-tests.md`, and `.env.example`.
- Story 4.1's keyword index exists and `uv run python -m building_with_rag.retrieval.hybrid --check` reports ready, because this mode sits on top of hybrid. If it does not, stop and say to fix Story 4.1 first. This story does not reload, re-chunk, re-embed or reindex anything.
- Settings already exist in `.env.example` and `Settings`. Use these exact names and defaults through `Settings`, and do not rename, add, remove or re-default any setting or edit `.env.example`:

| Setting | Default | Use in this story |
|---|---|---|
| `RERANK_API_BASE_URL` | `https://api.voyageai.com/v1` | base URL of the re-rank request |
| `RERANK_API_KEY` | empty | bearer key; empty means the explicit missing-key outcome |
| `RERANK_MODEL_NAME` | `rerank-2.5` | model name sent in the request |
| `RERANK_REQUEST_TIMEOUT_SECONDS` | `30` | timeout of the one re-rank request |
| `RERANK_CANDIDATE_LIMIT` | `20` | how many hybrid candidates are taken |
| `RERANK_SEND_LIMIT` | `10` | how many of those are sent to the re-ranker |
| `RERANK_RETURN_LIMIT` | `5` | how many re-ranked passages are kept |

  Plus the existing hybrid, generation and `CAPSTONE_API_KEY` settings for the answer path. A live check needs `RERANK_API_KEY` in the untracked `.env`.

## Contracts to preserve

Do not rename, replace or remove: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `OmittedCandidate`, `SemanticFilters`, `GenerationResult`, the `/v1/query` route and response model, the registry (six modes, `rag-<pattern>` IDs, `hybrid-reranked` / `rag-hybrid-reranked` already present), `/v1/models`, the chat adapter with its streaming and `CAPSTONE_API_KEY` behavior, context assembly and citation rules, the `retrieval.semantic` and `retrieval.hybrid` behavior and outputs, and `.env.example`. Changes are additive only.

- `semantic` and `hybrid` results keep `rerank_score` and `rerank_rank` as `null`. `tests/test_semantic.py` and `tests/test_hybrid.py` pass unmodified.
- The other three modes (`structured`, `decomposition`, `hyde`) stay `not_implemented`.
- Retrieval `unavailable` stays HTTP 503 with empty `results` and no generation; `generation` outcomes never change the retrieval `status`.

## Work to do

### 1. The re-ranking step (one documented rule)

Add `src/building_with_rag/retrieval/rerank.py` holding the mode's retrieval function (`retrieve(request, *, settings=None, db=None, embed=None, rerank=None)` or the same shape as hybrid's, with the provider call injectable for tests). It reuses hybrid's `retrieve` as it is; do not copy its routes or fusion, and do not change hybrid's behavior. If a tiny helper must be exposed for reuse, do it without changing hybrid's or semantic's output.

The pipeline is exactly this, with each stage bounded by the settings above:

1. **Candidates.** Run hybrid retrieval for the same question, filters and scope, asking for `RERANK_CANDIDATE_LIMIT` fused candidates (ignoring the request's own `limit` for this stage). Candidates are in `fused_rank` order. Validate the three limits at use: each is an integer of at least 1, `SEND <= CANDIDATE`, and `RETURN <= SEND`; a violation is `unavailable` with reason `invalid_rerank_settings` naming the offending setting names only. The request `limit` (1 to 20, default 5) can only lower the final count: **final count = min(request.limit, RERANK_RETURN_LIMIT)**. Record this rule in the architecture.
2. **Send.** Take the first `RERANK_SEND_LIMIT` candidates in fused order. Candidates beyond that are cut before re-ranking.
3. **Re-rank.** One provider request (item 2) scores each sent candidate's `text` against the trimmed question. The order of the response by score, descending, is the re-rank order; ties break by `fused_rank`.
4. **Keep.** Keep the first *final count* passages. The rest of the sent candidates are cut after re-ranking.
5. The kept passages, in re-rank order, are `QueryResult.results`. This is the only evidence context assembly and the answer generator receive.

If hybrid retrieval returns `unavailable` or `no_results`, the reranked result carries that same status, reason and message (with `pattern="hybrid-reranked"`), and no re-rank call is made. An empty scope is `no_results` with no Voyage, MongoDB or re-rank call. One question makes at most one re-rank request: no retry loop, no second pass.

### 2. The provider call (existing contract, no new framework)

- One `httpx` `POST` to `{RERANK_API_BASE_URL}/rerank` (trailing slash trimmed), `Authorization: Bearer RERANK_API_KEY`, JSON body with `model` (`RERANK_MODEL_NAME`), `query` (the trimmed question), `documents` (the sent candidates' `text`, in fused order) and `top_k` (the number of sent documents, so every sent candidate gets a score; the keep cut is applied by the application, not the provider). Use `RERANK_REQUEST_TIMEOUT_SECONDS` as the timeout. `httpx` is already a dependency; add no SDK, framework or dependency.
- Before writing code, check the current Voyage rerank documentation for the endpoint path, field names and response shape (expected: a `data` list of `{index, relevance_score}`), and record any difference from the above in the handover. Do not invent a different provider contract and do not change the base URL, model name or the names above.
- Map each response entry to a sent candidate by `index`. Validate the response: it is a list with an integer `index` inside range, a numeric score, no duplicate index. An incomplete or malformed response (a sent candidate without a score, an unknown index, wrong shape) is a failure, not a partial result. The application never fabricates, estimates or fills in a score.
- Passage text is evidence sent to the provider as data only. Never log, trace or return the key, base URL, request body, raw response, raw exception text or passage text from this step.

### 3. Outcomes (clear and safe; no silent fallback)

Use the existing `QueryResult` statuses and the `Unavailable`-style reasons already used by semantic and hybrid.

| Situation | `status` | HTTP (`/v1/query`) | `trace.reason` |
|---|---|---|---|
| Re-ranked, at least one passage kept | `ok` | 200 | none |
| Hybrid found nothing, or the scope is empty | `no_results` | 200 | as hybrid |
| `RERANK_API_KEY` empty | `unavailable` | 503 | `missing_setting`, message names `RERANK_API_KEY is not set` |
| `RERANK_API_BASE_URL` or `RERANK_MODEL_NAME` empty | `unavailable` | 503 | `missing_setting`, message names the setting |
| Timeout, connection error, non-2xx status, or malformed or incomplete response | `unavailable` | 503 | `rerank_failed`, with a coarse cause such as `timeout`, `http_<status>` or `malformed_response` |
| Invalid limits | `unavailable` | 503 | `invalid_rerank_settings` |
| Hybrid itself unavailable | `unavailable` | 503 | hybrid's reason |

Hard rules:

- **No silent fallback.** If re-ranking cannot run or fails, return `unavailable` with empty `results`. Never return the un-re-ranked hybrid list under `pattern="hybrid-reranked"`, never switch to `hybrid` or `semantic`, and never return partial scores. The message says plainly that no re-ranking was performed and no passages are returned.
- The missing-key check happens before any MongoDB, Voyage embedding or hybrid call, so a missing key costs nothing and the message is explicit. The retrieval-side missing settings (`MONGODB_URI`, `VOYAGE_API_KEY`) keep their existing outcome and wording.
- `message` on `ok` states that results are hybrid candidates re-ordered by one re-ranking model, that scores are relevance rankings, and that this is not proof a passage answers the question. No score threshold is applied, so an unrelated question still returns its best-ranked passages.
- No secret, URI, base URL or raw exception text in any message, trace or log.

### 4. Result shape and diagnostics (additive)

- Add to `RetrievedChunk`, all optional with `None` defaults: `rerank_score: float | None` and `rerank_rank: int | None`. Add nothing else to `RetrievedChunk`, and nothing to `QueryResult` or `OmittedCandidate`. Update the comment above the hybrid fields in `contracts.py` to describe the new pair.
- Each returned chunk keeps all its hybrid fields (`semantic_*`, `keyword_*`, `fused_score`, `fused_rank`) so its position and score before re-ranking stay readable. It sets `rerank_score` (the provider's score) and `rerank_rank` (1-based position in re-rank order; contiguous 1..n for returned chunks). `score` equals `rerank_score`, as `score` equalled `fused_score` in hybrid.
- `QueryResult.omitted_candidates` lists every candidate that did not reach the final evidence, using the existing `{chunk_id, omitted_reason}` shape, with these reasons: `below_limit` (hybrid's existing reason for a fused candidate cut by the candidate limit, if any is reported), `below_send_limit` (cut before re-ranking, beyond `RERANK_SEND_LIMIT`) and `below_return_limit` (re-ranked but cut by the final count). Each candidate appears once with one reason.
- Omitted candidates carry no scores in that shape, so the order and scores must also be readable in `trace`, using only ids, ranks and numbers (no passage text):
  - `trace["rerank"]`: `model` (`RERANK_MODEL_NAME`), the four counts (`candidates_taken`, `sent`, `returned`, `omitted`), the effective limits (`candidate_limit`, `send_limit`, `return_limit`, `final_count`), and `timeout_seconds`.
  - `trace["rerank"]["before"]`: the sent candidates in fused order as `{chunk_id, fused_rank, fused_score}`.
  - `trace["rerank"]["after"]`: the same sent candidates in re-rank order as `{chunk_id, rerank_rank, rerank_score}`, including those later cut, so the ranking of every sent candidate is inspectable.
  - `trace["rerank"]["selected"]`: the chunk ids that became final evidence, in order.
  - Plus everything hybrid's trace holds (`query`, `limit`, `effective_filters`, embedding and keyword index details, `candidates_per_route`, `route_overlap`, `fusion`, `ignored_fields`, `caller_id`, `unresolved_hits`).
- The diagnostic response must therefore show, without generation: candidate order and scores before re-ranking, order and scores after re-ranking, the final selected evidence, and each cut candidate with its reason.
- `QueryResult.pattern="hybrid-reranked"`. `source` is built exactly as in hybrid and semantic.

### 5. Make the mode selectable (no new path, no routing)

- `hybrid-reranked` / `rag-hybrid-reranked` already exist in the registry and `/v1/models`; do not add or rename modes. Change `run_pattern_events` in `service.py` so `hybrid-reranked` is dispatched to the new retrieval function in the same branch and with the same generation hand-off as `semantic` and `hybrid`: one retrieval, then one `generate_events` operation over that same result. The generation condition (`generate_answer`, `answer_generation`) stays as it is.
- The answer generator receives only the final selected evidence: `generate_events` reads `QueryResult.results`, which are the kept passages in re-rank order. Candidates cut before or after re-ranking never reach context assembly, the prompt, the label map or citations. Context assembly, the `MAX_PASSAGES` and `MAX_CONTEXT_CHARS` budget, `E1`..`En` labels (in re-rank order), citation resolution, validation, confidence and the DRAFT, sources and low-confidence rendering are unchanged and still apply. Answers must still cite the evidence. Update `generation/answer.py` only if something assumes the pattern is `semantic` or `hybrid`, additively and without changing their output.
- Selection is only by `"pattern": "hybrid-reranked"` on `/v1/query`, the `rag-hybrid-reranked` model ID, or `rag_options.pattern` in chat. There is no new endpoint, UI, query classifier, automatic choice between modes, or fallback from one mode to another.

### 6. Diagnostic command

`uv run python -m building_with_rag.retrieval.rerank "<question>" [--limit N] [--act X] [--status X] [--access-level X]`, plus `--check` that reports whether the re-rank settings are present and valid (names of any missing or invalid setting, never values) and then hybrid readiness. It calls the same function `run_pattern` uses. Compact output: status, message, mode, query, the rerank counts and limits, then the **before** table (fused rank, fused score, `chunk_id`), the **after** table (re-rank rank, re-rank score, `chunk_id`, fused rank so a move is visible), the final selection with `section_id`, `act`, heading, source pages and text length, and omitted candidates with reasons. No passage text, keys or vectors. Exit non-zero for `unavailable`.

### 7. Tests (a few, offline)

No network, MongoDB, Voyage, re-rank provider, model provider or server. Fake hybrid retrieval and the provider call. Cover only:

- the pipeline bounds: a fake candidate list longer than the send limit, then the return limit and the `request.limit` rule, checking `rerank_rank` 1..n, `score == rerank_score`, hybrid fields preserved, and that every cut candidate appears once in `omitted_candidates` with the correct reason;
- order change: a fake provider that reverses the order produces results in re-rank order, with `trace["rerank"]["before"]` and `["after"]` showing both orders and scores, including a sent-then-cut candidate;
- no silent fallback: empty `RERANK_API_KEY` gives `unavailable` naming `RERANK_API_KEY` with empty `results` and no hybrid or provider call; a timeout, a non-2xx status, and a malformed or incomplete response each give `unavailable` with `rerank_failed` and empty `results`, never the hybrid list; invalid limits give `invalid_rerank_settings`;
- `run_pattern` with `pattern="hybrid-reranked"` and generation on makes one retrieval and one generation operation, and the context given to generation contains only the kept passages in re-rank order; `semantic` and `hybrid` results still have `rerank_score` and `rerank_rank` `null`.

Existing tests keep passing unmodified except where a test asserted that `hybrid-reranked` is a `not_implemented` placeholder (`tests/test_smoke.py`, `tests/test_generation.py`); move those assertions to `structured`. No broad suite or evaluation harness.

### 8. Documentation

- `docs/architecture.md`: keep existing text. Add a "Hybrid re-ranking" section covering: the pipeline (candidates, send, re-rank, keep) and the final-count rule; the provider call and the exact `RERANK_*` settings with defaults; the `rerank_score` and `rerank_rank` fields and that `score` is the re-rank score; the `omitted_candidates` reasons and the `trace["rerank"]` fields; the outcomes table, especially the explicit missing-key outcome and the no-silent-fallback rule; that context assembly and generation take only the final selected evidence through the same path; the diagnostic command; and the lightweight comparison finding after it is run. Update only the lines that call `hybrid-reranked` a placeholder, and add the `RERANK_*` settings to the settings description. Add a **Limitations** paragraph: re-rank scores are one model's relevance ranking, not proof; the re-ranker only sees the sent candidates, so a relevant passage hybrid ranked below `RERANK_SEND_LIMIT` (or below the candidate limit) is never seen; re-ranking adds one external call, so latency and cost go up and the mode depends on a third service; passages are sent to the provider; no score threshold, so an unrelated question still returns its best-ranked passages; no caching; the bounds are defaults, not tuned; and the legal text sent is chunk text only (section numbers and headings are not in `text`, as noted for hybrid).
- `docs/manual-tests.md`: add a `## Story 4.2 — Hybrid Re-ranking` section following `docs/master-prompt-update-manual-tests.md`: a `/v1/query` curl with `pattern: "hybrid-reranked"`, one failure or edge case (for example an empty scope via `access_level: ["restricted"]`, and the 503 `RERANK_API_KEY is not set` outcome when the key is blank), and a `rag-hybrid-reranked` chat curl with `"stream": true` ending in `data: [DONE]`. Do not run them as part of this item. Also update the earlier line in that file that shows `hybrid-reranked` as `not_implemented`, if it still says so.

## Token-safe checks

Every check prints only statuses, counts, ids, ranks, scores, headings, lengths and booleans. Never print keys, vectors, request bodies or full passage text in scripts, logs, tests or the handover. If a check needs passage text, compare it in code and print booleans. Wait at least 20 seconds between steps that call Voyage.

## Out of scope

Automatic mode choice or routing, a query classifier, a fallback from one mode to another, an evaluation harness or full benchmark, a broad regression suite, latency or cost measurement tooling, advanced RAG patterns (query rewriting, HyDE, decomposition, GraphRAG, agentic behavior), a second or local re-ranker, a different provider contract or SDK, new settings, changes to hybrid fusion or its constants, a score threshold, changes to embeddings, chunking, the corpus, indexes or `.env.example`, multi-user authorization, a new endpoint, UI or second application.

## Completion checks

### Checks the implementer runs (no services)

1. `uv sync`, `uv run ruff check .` and `uv run pytest` pass, including the item 7 tests and the unmodified semantic and hybrid tests.
2. With `RERANK_API_KEY` empty, `uv run python -m building_with_rag.retrieval.rerank --check` reports `unavailable` naming `RERANK_API_KEY` and shows no credentials, and a `/v1/query` call for `hybrid-reranked` returns 503 `unavailable` with empty `results` and no hybrid call made.
3. `.env.example`, `data/raw/`, `data/processed/`, both search indexes and `load.py` behavior are unchanged, `.env` is untracked, and `Settings` has the same seven `RERANK_*` fields with the same defaults.
4. Reading the code shows one dispatch path: `hybrid-reranked` goes through `run_pattern_events`, hybrid retrieval is reused rather than copied, and generation consumes only `QueryResult.results` from the re-ranked result.
5. The new mode never returns hybrid results under `pattern="hybrid-reranked"` without a successful re-rank call.

### Direct checks (need the populated database, Voyage and re-rank keys, the proxy settings)

Run these if services are available; otherwise report them **pending** with what is missing. Never claim a pass that was not run.

1. **Readiness and a real question.** `--check`, then the re-rank command with one question. Expect `ok`, the counts and limits, a before table, an after table, a final selection with act-qualified `section_id`s, `score` equal to `rerank_score`, and `rerank_rank` 1..n.
2. **Inspectable ordering.** For one run, confirm from the output that the before order matches `fused_rank`, the after order matches `rerank_rank`, the selected ids equal the returned `results`, every omitted candidate has a reason, and the counts add up (`candidates_taken` = `returned` + `omitted`).
3. **Edge cases.** `--access-level restricted` gives `no_results` with no re-rank call; a blanked `RERANK_API_KEY` gives `unavailable` naming it with empty results; an unknown filter field on `/v1/query` gives 422; `pattern: "hybrid"` and `"semantic"` for the same question keep their previous shape with `rerank_score` and `rerank_rank` `null`.
4. **Provider failure.** With a deliberately wrong `RERANK_MODEL_NAME` or a very small timeout in a temporary environment (not committed), confirm `unavailable` with `rerank_failed`, empty results, and no hybrid list returned.
5. **Answer path.** `POST /v1/query` with `pattern: "hybrid-reranked"` and `generate_answer: true`, and the same in Open WebUI (select **Building with RAG**, choose **hybrid-reranked** in the RAG-options chip). Expect the same behavior as hybrid: draft label, confidence, sources, citations only to the final selected passages (labels `E1`..`En` in re-rank order), and retrieval `unavailable` as 503 on `/v1/query`.
6. **Comparison (required, lightweight).** Pick **one** question where ordering plausibly matters: for example a question whose best passage may sit just below a similar sibling (such as a question that distinguishes culpable homicide from murder, or a BNS versus IPC wording difference), or one where hybrid's top hit shares the words but not the point. Run it in `hybrid` and then `hybrid-reranked` with the same `limit`, then once more in each with `generate_answer: true`. Record and compare:
   - **Order:** hybrid's top results (rank, `chunk_id`, `section_id`, fused score) against the re-ranked results (rank, `chunk_id`, `section_id`, `rerank_score`, former `fused_rank`); which passages moved, entered or left the final set, and which sent candidates were cut.
   - **Sources:** read the top passages in each mode and say which looks more relevant to the question and why.
   - **Answer:** whether the two answers, citations and confidence differ, and whether the final cited evidence changed.
   - **Latency and cost, qualitatively:** note if visible (for example "the reranked request took noticeably longer", one extra provider call, up to `RERANK_SEND_LIMIT` passages sent) without a timing harness. If no provider timing or usage was observable, say so rather than guess.

   State the finding as an observation from one question, including when re-ranking made no difference or made things worse. Do not claim a general winner, and do not build a benchmark.

## Handover

**The implementer fills in:**

- **Files created or changed.**
- **Provider contract:** the exact endpoint path, request fields and response shape used, and any difference from item 2 found in the current documentation.
- **Commands actually run,** with results. State which direct checks were run and which are pending and why.
- **Pipeline bounds as implemented:** the effective candidate, send, return and final-count rules, the `omitted_reason` values, and the `trace["rerank"]` fields.
- **Commands for you to run:** the re-rank diagnostic with `--check` and with a question, and the `/v1/query` and chat calls.
- **Deviations and open issues.**

**You fill in after running it:**

- **Readiness line** and the missing-key outcome you saw.
- **Checks 1 to 5:** what you saw (statuses, counts, before and after order, scores, omitted reasons, the Open WebUI display).
- **Comparison (check 6):** the one question, hybrid and hybrid-reranked order, sources and answers, the observed difference, and qualitative latency and cost.
- **Known limitations:** re-rank scores are one model's ranking, not correctness; only sent candidates are seen, so a relevant passage ranked below the send limit by hybrid is lost; one extra external call adds latency and cost and a third-service dependency; passage text leaves the system to the provider; no score threshold; bounds are untuned defaults.

Later stories add the remaining modes behind the same `run_pattern`, `QueryResult` and `RetrievedChunk`, additively.
