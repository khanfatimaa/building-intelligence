# Story 4.1: Hybrid Search

## Purpose

Make the `hybrid` mode (`rag-hybrid`) real. Some questions hinge on exact legal wording ("culpable homicide", "dowry death", a phrase copied from a section) that semantic similarity can blur. Hybrid adds a keyword route over the same chunk text, runs it beside the semantic route, and fuses the two candidate lists with one explicit, documented rule.

Everything else is shared with `semantic`: the same corpus and MongoDB deployment, the same `RetrievedChunk` evidence shape, the same context assembly, citations, `/v1/query`, model mapping, Open WebUI streaming path, and `.env.example` contract. There is no second application and no new answer path. The working `semantic` mode must behave exactly as it does today.

A fused score is a ranking, not proof that a passage is correct or sufficient. Nothing in this story may present it as one.

Not in this story: automatic routing, re-ranking (`hybrid-reranked` stays a placeholder), an evaluation harness, a broad regression suite, GraphRAG, agentic behavior, a change of embedding model, or regenerating the corpus.

## Prerequisites

- Stories 1.1, 2.1, 2.2, 2.3, 3.1 and 3.2 are complete. First run `uv run pytest` and `uv run ruff check .`; if they fail, stop and report. Do not fix other stories here.
- Read first: `docs/architecture.md`, all prior stories in `docs/stories/`, `contracts.py`, `registry.py`, `service.py`, `main.py` (the chat adapter), `settings.py`, `retrieval/semantic.py` (the pattern to follow and reuse), `ingestion/mongodb_schema.py`, `ingestion/load.py` (its capability check and index helpers), `generation/context.py`, `generation/answer.py`, `docs/manual-tests.md`, and `.env.example`. There is no `docs/config.yaml`; use the repository root.
- The Story 2.2 load has been run and semantic retrieval works against the populated database. This story does not reload, re-chunk or re-embed anything. If semantic readiness (`uv run python -m building_with_rag.retrieval.semantic --check`) fails, stop and say to fix that first.
- An untracked `.env` made from `.env.example`. Use only existing settings through `Settings`: `MONGODB_URI`, `MONGODB_DB_NAME`, `VOYAGE_API_KEY`, `WEBUI_DEMO_CALLER_ID`, plus the generation and `CAPSTONE_API_KEY` settings for the answer path. Do not rename, add, or remove settings and do not change `.env.example`. The unused `RERANK_*` settings stay unused.

## Contracts to preserve

Do not rename, replace, or remove: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `SemanticFilters`, `GenerationResult`, the `/v1/query` route and response model, the registry (six modes, `rag-<pattern>` IDs), `/v1/models`, the chat adapter and its streaming and `CAPSTONE_API_KEY` behavior, the context-assembly and citation rules, the constants and index in `mongodb_schema.py`, the `embeddings_vector_index`, and `.env.example`. Changes are additive only. `semantic` results keep the new hybrid fields `null`. The other four modes stay `not_implemented`.

## Work to do

### 1. Identify the keyword-search capability (before writing any retrieval code)

The configured deployment is MongoDB Atlas (story 2.2 verified `list_search_indexes` and `$vectorSearch`). Its keyword capability is Atlas Search: a `search`-type index queried with `$search`. Do not assume it; prove it:

1. Connect with a short timeout and `ping`.
2. Call `list_search_indexes` on `chunks` (the collection that holds the chunk text). If it fails because the deployment is not Atlas or the user lacks the privilege, **stop and report the missing prerequisite**. Do not fall back to a regex or `$text` scan, a client-side keyword match, or another search engine. `semantic` stays unchanged and `hybrid` reports `unavailable` (item 5).
3. Print one line recording the capability, for example "Atlas Search available; search index type used for keyword retrieval", and record the same exact wording in `docs/architecture.md`.
4. Check the current Atlas Search documentation for the index definition and query syntax in items 2 and 4 below before relying on them; the shapes shown here are the intended design, and the implementer records any difference. An M0 cluster allows only a few Atlas Search indexes in total (the vector index already uses one); stop and report if creating this one would exceed the limit.

### 2. Keyword field and index (one additive setup step)

Record the exact field and index requirement:

- **Field searched:** `chunks.text`, the plain chunk text. It is not `embedding_input_text`. Consequence to document: the section number and heading are in `embedding_input_text`, not in `text`, so a question that only names "section 302" matches chunks whose text happens to contain that number (for example a cross-reference), not necessarily section 302 itself. Record this as a known limitation. Do not change what is indexed to work around it without asking.
- **Index:** a `search`-type Atlas Search index on `chunks` named by a new constant `KEYWORD_INDEX_NAME` (for example `chunks_text_index`), with a static mapping (`dynamic: false`): `text` as `string` with a named analyzer constant (start with `lucene.standard`, which does not stem, so exact terms and section numbers stay literal; record the analyzer in the architecture), and `act`, `status`, `access_level` as `token` so the scope filter runs inside `$search`.
- Put the names and a `keyword_index_definition()` beside the vector-index code in `ingestion/mongodb_schema.py`, additively. Import them everywhere; do not repeat names. Do not change `ensure_vector_index`, the vector index, `load.py`'s behavior, or any document.
- Add a small setup command, `uv run python -m building_with_rag.ingestion.keyword_index`, that runs item 1, then creates the index with the documented PyMongo search-index API, or reuses it if an index of that name exists with a matching type and definition. If one exists with a different definition or type, report the difference and stop; never delete, drop, recreate or rename an index. After creating it, wait a bounded time (for example 5 minutes, polling every 10 seconds) for `queryable`; if not ready, say "index not ready yet" and treat verification as pending. Finish with one real `$search` smoke query that prints only the number of hits and the top `chunk_id` and score. Never print the URI, keys or passage text.

### 3. The keyword route and the semantic route

Add `src/building_with_rag/retrieval/hybrid.py`, following `retrieval/semantic.py`. Reuse, do not copy, its helpers (scope and filter building, embedding, resolution of `chunk_id` to `chunks`, `sections` and `sources`, source-dict building, Voyage error mapping). If a small helper must be exposed or moved for reuse, do it without changing semantic behavior, and `tests/test_semantic.py` must pass unmodified.

- **Validation, scope and filters:** identical to semantic: question trimmed to 1 to 4,000 characters, `limit` 1 to 20, `SemanticFilters` rules, fixed demo caller, `public` base scope that filters only narrow, `chapter` and `required_acts` ignored and listed in `trace["ignored_fields"]`, and an empty scope is `no_results` with no Voyage or MongoDB search call.
- **Semantic route:** the same `$vectorSearch` as semantic (Voyage `voyage-3.5`, 1,024 dimensions, `input_type="query"`, same filter, same `numCandidates` rule), asking for the per-route candidate count below.
- **Keyword route:** an Atlas `$search` on `KEYWORD_INDEX_NAME` over `text`, using the `text` operator with the trimmed question, inside a `compound` whose `filter` clauses are built from the same validated scope lists (database-side; never filter after retrieval and never pass a caller value except as a validated list item). Then `$limit` to the per-route candidate count and `$project` `chunk_id` and `{"$meta": "searchScore"}`. Keyword scores are on a different scale from cosine scores and are never added to or compared with them directly.
- **Per-route candidate count:** named constants in the module, no new settings: a floor, a multiplier of `limit`, and a cap (for example at least 20, never above 50, always at least `limit`). Both routes use the same count.
- Resolve hits from both routes to `RetrievedChunk`s through the same path. A hit that does not resolve is left out and counted per route in the trace.

### 4. Fusion rule (explicit and documented)

Use **Reciprocal Rank Fusion (RRF)** with equal route weights:

```
fused_score(chunk) = sum over routes containing the chunk of 1 / (RRF_K + rank_in_route)
```

- `RRF_K = 60` and equal weights are named constants. Ranks are 1-based and come from each route's own ordering (semantic by vector score, keyword by search score; ties inside a route broken by `chunk_id`). Raw scores are not used by the fusion, which is why different score scales are safe.
- A chunk found by both routes sums both contributions; a chunk found by one route gets that route's contribution only.
- Order by `fused_score` descending, ties by `chunk_id`, and assign `fused_rank` 1..n. Return at most `limit`. Do not de-duplicate across chunks, merge passages, rewrite text, or re-rank beyond this rule.
- Say in the architecture why RRF was chosen (no tuning, no score normalization, works with two unlike scores) and its limitations: equal weights are a judgment, not tuned; a chunk seen by only one route can still rank high; rank fusion ignores how strong a score was; no score threshold, so an unrelated question still returns its nearest passages.

### 5. Result shape, source metadata and outcomes

- Add to `RetrievedChunk`, additively and all optional with `None` defaults: `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`. Add nothing else (no `rerank_*` fields). `QueryResult` is unchanged.
- For a hybrid result, `score` equals `fused_score`. A route that did not return the chunk leaves its score and rank `null`, so which route contributed each candidate is readable straight from the result: both set, semantic only, or keyword only. `source` is built exactly as in semantic.
- `QueryResult.pattern="hybrid"`. `trace` holds everything semantic's trace holds (`query`, `limit`, `effective_filters`, embedding model and version, `ignored_fields`, `caller_id`, `unresolved_hits`) plus: the keyword index name, field and analyzer; the vector index; `candidates_per_route`; `semantic_candidates_returned`, `keyword_candidates_returned`; `route_overlap` (chunks found by both), `semantic_only` and `keyword_only` counts; and the fusion rule (`name`, `k`, `weights`). The diagnostic response must make it clear, before generation, why each passage is there. `omitted_candidates` may list fused candidates cut by `limit` using the existing `OmittedCandidate` shape and a reason such as `below_limit`.
- Outcomes and HTTP codes match semantic: `ok` (200), `no_results` (200, both routes empty or empty scope), `unavailable` (503, empty results), invalid input (422). `unavailable` covers a missing `VOYAGE_API_KEY` or `MONGODB_URI`, Voyage or MongoDB failure, semantic readiness failure, or a keyword index that is missing, wrongly defined or not yet queryable (reason such as `keyword_index_missing`, `keyword_index_incompatible`, `keyword_index_not_ready`). Hybrid never quietly degrades to one route and still calls itself hybrid. A keyword-side failure never changes `semantic` behavior. No secret, URI or raw exception text in any message or trace. A short readiness check for the keyword index (type, definition, queryable) is cached like semantic's.
- `message` states plainly that scores are rankings, that results fuse a semantic and a keyword ranking, and that this is not proof a passage answers the question.

### 6. Make hybrid selectable (no routing)

- `hybrid` / `rag-hybrid` already exist in the registry and `/v1/models`; do not add or rename modes. `run_pattern` / `run_pattern_events` dispatch `hybrid` to the new retrieval function exactly as `semantic` is dispatched, with generation reading the same `QueryResult.results` through the same context assembly, citations, confidence, and streaming events. If the generation path has anything that assumes `pattern == "semantic"`, fix it additively without changing semantic's output.
- Hybrid is chosen only by the model ID (`rag-hybrid`) or `rag_options.pattern` in chat, and `pattern: "hybrid"` on `/v1/query`. There is no automatic choice between modes, no query classifier, and no fallback from one mode to another.
- The generation context keeps passages in fused order. Citations may only name retrieved passages, as today.
- Chat text, `DRAFT`, confidence, and sources rendering are unchanged. Update the architecture and handover note that earlier said only `rag-semantic` runs real generation.

### 7. Diagnostic command

`uv run python -m building_with_rag.retrieval.hybrid "<question>" [--limit N] [--act X] [--status X] [--access-level X]`, plus `--check` for readiness (semantic readiness plus keyword index status). It calls the same function `run_pattern` uses. Compact output: status, message, mode, query, filters, route counts and overlap, and per result: fused rank and score, semantic rank and score, keyword rank and score (4 decimals, `-` when absent), the route label (`both`, `semantic`, `keyword`), `chunk_id`, `section_id`, heading, `act`, source pages, and passage length. No passage text, vectors or keys. Exit non-zero for `unavailable`.

### 8. Tests (a few, offline)

No network, MongoDB, Voyage, model provider or server; fake both routes and the embedding call. Cover only:

- the fusion rule: a chunk in both routes outranks single-route chunks at similar ranks, a keyword-only and a semantic-only chunk each keep the other route's fields `null`, ties break by `chunk_id`, `score == fused_score`, `fused_rank` is 1..n, and the limit is applied after fusion;
- the built `$search` stage: scope clauses come only from validated lists, `public` is always present, an empty scope returns `no_results` with no search call;
- an unavailable keyword index gives `unavailable` (not a semantic-only result), and `semantic` results still have the six new fields `null`;
- `run_pattern` with `pattern="hybrid"` and generation on uses the same path and context assembly as semantic (one retrieval, one generation operation).

Existing tests keep passing unmodified except where a test asserted that `hybrid` is a `not_implemented` placeholder. Update only those assertions. No broad suite or evaluation harness.

### 9. Documentation

- `docs/architecture.md`: keep existing text. Add a "Hybrid retrieval" section covering: the capability line from item 1; the keyword field (`chunks.text`), index name, analyzer and definition; the two routes and the shared scope and filters; the fusion rule (RRF, `RRF_K`, equal weights, per-route candidate count); the six `RetrievedChunk` fields and that `score` is the fused score; the trace fields; outcomes; that hybrid reuses the same context assembly, generation and streaming path; the diagnostic and setup commands; and the limitations from item 4 and the `text` field note in item 2. Update only the lines that say `hybrid` is a placeholder or that only `rag-semantic` runs real generation.
- `docs/manual-tests.md`: add a `## Story 4.1 — Hybrid Search` section following `docs/master-prompt-update-manual-tests.md` (a `/v1/query` curl with `pattern: "hybrid"`, one failure or edge case, and a `rag-hybrid` chat curl with `"stream": true` ending in `data: [DONE]`). Do not run them as part of this item.

## Token-safe checks

Every check prints only statuses, counts, ids, ranks, scores, headings, lengths and booleans. Never print vectors, keys, or full passage text in scripts, logs, tests or the handover. If a check needs passage text, compare it in code and print booleans. Wait at least 20 seconds between steps that call Voyage.

## Out of scope

Automatic routing, query classification, re-ranking or `rerank_*` fields, weight tuning, an evaluation harness, a broad regression suite, GraphRAG, agentic behavior, query rewriting, a score threshold, a change of embedding model, chunking or corpus, indexing `embedding_input_text`, changes to `load.py` behavior, the vector index, `.env.example`, existing contracts, routes or the registry, multi-user authorization, a second application or UI.

## Completion checks

### Checks the implementer runs (no services)

1. `uv sync`, `uv run ruff check .` and `uv run pytest` pass, including the item 8 tests and unmodified semantic tests.
2. With `MONGODB_URI` or `VOYAGE_API_KEY` empty, `uv run python -m building_with_rag.retrieval.hybrid --check` reports `unavailable` naming the missing setting and shows no credentials. `uv run python -m building_with_rag.ingestion.keyword_index --help` works without touching MongoDB.
3. `.env.example`, `data/raw/`, `data/processed/`, the vector index definition and `load.py` behavior are unchanged; `.env` is untracked.
4. Reading the code shows one dispatch path: `hybrid` goes through `run_pattern_events`, and generation consumes the same `QueryResult`, context assembly and events as `semantic`.

### Direct checks (need the populated database, a Voyage key, the proxy settings)

Run these if services are available; otherwise report them **pending** with what is missing. Never claim a pass that was not run.

1. **Capability and index.** Run `uv run python -m building_with_rag.ingestion.keyword_index`. Expect the capability line, the index created or reused, `queryable`, and the smoke query's hit count. If the deployment lacks the capability, expect the prerequisite report, `uv run python -m building_with_rag.retrieval.hybrid --check` reporting `unavailable`, and semantic still working.
2. **Readiness and a real question.** `--check`, then the hybrid command with one question. Expect `ok`, each result with fused, semantic and keyword fields as applicable, a route label, an act-qualified `section_id`, and `score` equal to the fused score.
3. **Fusion is inspectable.** For one run, confirm from the output that at least the overlap, semantic-only and keyword-only counts match the per-result route labels, and compute one fused score by hand from the printed ranks and `RRF_K`.
4. **Edge cases.** `--access-level restricted` gives `no_results` with no Voyage call; a blanked `VOYAGE_API_KEY` gives `unavailable` naming it; an unknown filter field on `/v1/query` gives 422; `pattern: "semantic"` for the same question still returns its previous shape with the six new fields `null`.
5. **Semantic unchanged.** Run the Story 2.3 example question in semantic mode and confirm the status, result count, ids and scores are as before.
6. **Answer path.** `POST /v1/query` with `pattern: "hybrid"` and `generate_answer: true`, and the same in Open WebUI (select **Building with RAG**, choose **hybrid** in the RAG-options chip). Expect the same behavior as semantic: draft label, confidence, sources, citations only to retrieved passages, retrieval `unavailable` as 503 on `/v1/query`.
7. **Comparison (required).** Pick two questions and run each in `semantic` and then `hybrid` with the same `limit`:
   - one **paraphrased** question that avoids the section's own wording (for example describing killing with intent without saying "murder");
   - one built around an **exact legal term or section-language phrase** that appears in the corpus text (for example "culpable homicide not amounting to murder" or "dowry death"). If you also try a bare section-number question, note the `text` field limitation from item 2.

   For each, record the retrieved sources (ids, acts, sections, ranks, route labels, scores), read the top passages, and read the generated answer (`/v1/query` with `generate_answer: true`). Document the observed difference in the handover: which passages only one mode returned, where the keyword route added or did not add something relevant, whether the answers differed, and where hybrid was no better or worse. Do not claim a universal winner; two questions show two observations.

## Handover

**The implementer fills in:**

- **Files created or changed.**
- **Capability and index:** the exact capability line, field, index name, analyzer and definition used, and any difference from the design above.
- **Commands actually run,** with results. State which direct checks were run and which are pending.
- **Constants chosen:** `RRF_K`, weights, per-route candidate floor, multiplier and cap, readiness cache time.
- **Commands for you to run:** the keyword index setup command and the hybrid diagnostic command and `--check`.
- **Deviations and open issues.**

**You fill in after running it:**

- **Readiness and index line.**
- **Checks 1 to 6:** what you saw (statuses, route counts, ranks, scores, the hand-computed fused score, the Open WebUI display).
- **Comparison (check 7):** for each of the two questions, the two modes' sources and answers and your observed difference, stated as an observation, not a general ranking of the modes.
- **Known limitations:** fused scores are rankings, not correctness; RRF equal weights are untuned; the keyword route searches `chunks.text` only, so section numbers and headings are not keyword-searchable; an unrelated question still returns nearest passages; a keyword hit on a common phrase can add a weak passage; a new index may take a short time to become queryable; an M0 cluster limits Atlas Search indexes.

Later stories add re-ranking and other modes behind the same `run_pattern`, `QueryResult` and `RetrievedChunk`, and may add their own fields (such as `rerank_score` and `rerank_rank`) additively.

## Handover record (implementer)

- **Files created or changed:** `retrieval/hybrid.py` (new), `ingestion/keyword_index.py` (new), `ingestion/mongodb_schema.py` (keyword constants, `keyword_index_definition()`, `keyword_index_matches()`), `retrieval/semantic.py` (the search split into `vector_hits`, `resolve_hits` and `search` for reuse; behavior unchanged), `contracts.py` (six optional `RetrievedChunk` fields), `service.py` (dispatch of `hybrid`), `tests/test_hybrid.py` (new), `tests/test_smoke.py` and `tests/test_generation.py` (placeholder assertions moved from `hybrid` to `hybrid-reranked`), `docs/architecture.md`, `docs/manual-tests.md`.
- **Capability and index:** designed as in item 2: line "Atlas Search available; search index type used for keyword retrieval", field `chunks.text`, index `chunks_text_index` (type `search`, `dynamic: false`, `text` string with `lucene.standard`, `act`/`status`/`access_level` token). Not yet proved against a real cluster: Atlas's stored definition shape (the compatibility check compares field types and analyzer, not the whole document) and the `in` operator on token fields.
- **Commands actually run:** `uv run pytest -q` (54 passed before; 62 passed after), `uv run ruff check .` (passed), `uv run python -m building_with_rag.retrieval.hybrid --check` with `MONGODB_URI` and `VOYAGE_API_KEY` blanked (`unavailable`, names `MONGODB_URI`, no credentials), `uv run python -m building_with_rag.ingestion.keyword_index --help` (works). **Direct checks 1 to 7 are pending**: the live setup command, real questions and the semantic/hybrid comparison were not run, and no index was created in Atlas.
- **Constants chosen:** `RRF_K=60`, weights 1.0 / 1.0, per-route candidates floor 20, multiplier 4, cap 50, readiness cache 60 s.
- **Commands for you to run:** `uv run python -m building_with_rag.ingestion.keyword_index`, then `uv run python -m building_with_rag.retrieval.hybrid --check` and `uv run python -m building_with_rag.retrieval.hybrid "<question>"`.
- **Deviations and open issues:** none from the story. Existing tests that asserted `hybrid` is a placeholder now use `hybrid-reranked`. `ruff format --check` reports 6 files; this was not enforced before and was left alone.
