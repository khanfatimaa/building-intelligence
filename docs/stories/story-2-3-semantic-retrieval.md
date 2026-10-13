# Story 2.3: Semantic Retrieval

## Purpose

Build the first real course mode: `semantic`. A question is embedded with the same Voyage model used for the stored chunks, MongoDB Atlas Vector Search finds the closest chunks, and the matching source passages come back in `QueryResult.results` with enough detail to inspect them.

This story returns passages only. Answer generation, Open WebUI answer behavior, hybrid search, re-ranking, an evaluation harness, and broad testing are later or out of scope. A similarity score says a passage is close to the question. It never proves the passage is correct or sufficient, and nothing in this story may present it that way.

Story 2.2's handover says "Story 3.1 will add semantic retrieval". This story is that retrieval, numbered 2.3. Story 3.1 keeps its name for answer generation.

## Prerequisites

- Stories 1.1, 2.1 and 2.2 are complete. Read first: `docs/architecture.md`, the three story files, `src/building_with_rag/contracts.py`, `registry.py`, `service.py`, `main.py`, `settings.py`, `ingestion/mongodb_schema.py`, and `ingestion/load.py`. There is no `docs/config.yaml`; use the repository root.
- The 2.2 load command has been run for real: `sections`, `chunks` and `embeddings` are populated and `embeddings_vector_index` is queryable. This story does not load, re-embed, or change data or the index. If readiness (work item 1) fails, stop and say to run `uv run python -m building_with_rag.ingestion.load` first.
- An untracked `.env` made from `.env.example`. Use only existing settings through `Settings`: `MONGODB_URI`, `MONGODB_DB_NAME`, `VOYAGE_API_KEY`, `WEBUI_DEMO_CALLER_ID`. Do not rename, add, or remove settings and do not change `.env.example`.

## Contracts to preserve

Do not rename, replace, or remove: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `SemanticFilters`, the `/v1/query` route and its response model, the registry (`semantic` / `rag-semantic`), the model-ID mapping, the MongoDB document model and constants in `mongodb_schema.py`, `/healthz`, `/v1/models`, and `.env.example`. Changes to contracts are additive only. The other five modes stay as the Story 1.1 placeholder. Import every collection name, field name, model name, dimension, index name and filter field from `mongodb_schema.py`; do not repeat them.

## Work to do

### 1. Readiness check (before any retrieval)

Add one small module, for example `src/building_with_rag/retrieval/semantic.py`, with a readiness function that proves the data can be searched. Counts and metadata only:

- `sections` has documents for both acts (`BNS_2023`, `IPC_1860`); `chunks` is non-empty; every chunk has an embedding (`embeddings` count equals `chunks` count).
- Embeddings carry `embedding_model` and `embedding_version` equal to the schema constants, and `embedding_dimensions` is 1,024 (check metadata and one stored vector's length; never print a vector).
- `embeddings_vector_index` exists, is type `vectorSearch`, is queryable, and its definition matches `vector_index_definition()`.

Run it before retrieving. The API path may cache a successful result for a short time so it does not count collections on every request. If anything is missing, return the "unavailable" outcome (item 5) with a short reason such as `no_embeddings` or `index_not_ready`. Never query a partly loaded database and return thin results as if they were normal.

### 2. Validate the request

- `question`: trim whitespace, then require 1 to 4,000 characters. A blank or whitespace-only question is rejected (HTTP 422 through the existing validation handler). Do not change the contract's field names or limits.
- `limit`: keep the contract's range (1 to 20, default 5). Put the vector-search candidate count in named constants in the semantic module (for example a multiplier and a floor with a hard upper cap, always at least `limit`). No new environment settings.
- `SemanticFilters`: forbid extra fields (`extra="forbid"`, an additive change). The contract's own fields `act`, `status` and `access_level` stay accepted, only as lists of short plain strings (bounded list length and string length). Reject unknown field names, non-list values, operator-shaped values (a dict, or a key starting with `$`), and anything shaped like a MongoDB fragment. The same rules apply to filters that arrive through `rag_options` on `/v1/chat/completions`.
- Fields this mode does not use (`chapter`, `required_acts`) are ignored and listed in `trace["ignored_fields"]`. They are not turned into filters.
- This course has one fixed local demo caller and the public access level. Use `WEBUI_DEMO_CALLER_ID` as the caller and `public` as the base scope. Do not add users, roles, tokens or per-user authorization. A caller-supplied `caller_id` never widens scope.

### 3. Embed the question

Embed the trimmed question with the same model settings as the documents: `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` and `output_dimension`, with `input_type="query"` (documents used `"document"`). Check the returned vector has 1,024 numbers. Reuse the approach in `load.py`'s `embed_texts`, but this is a request path: one attempt with a bounded timeout and no long sleep-and-retry loops. Do not change `load.py` behavior. A missing `VOYAGE_API_KEY`, timeout, rate limit or provider error is the "unavailable" outcome and never prints the key, headers, or the vector.

### 4. Query MongoDB and build results

- Run `$vectorSearch` on `embeddings` with `VECTOR_INDEX_NAME`, `VECTOR_FIELD`, the query vector, the bounded `numCandidates`, and `limit`.
- Build the `filter` on the server from the validated lists, using only the index's filter fields. The effective scope is the base scope narrowed by the caller: `access_level` is the intersection of `["public"]` with any caller list, and `act` and `status` apply only when supplied. Filters narrow; they never widen. The filter is part of the `$vectorSearch` stage so the database applies it. Never filter after retrieval, and never pass a caller value into the query except as a validated list item.
- If the intersection of a field is empty (for example `access_level=["restricted"]`), nothing can match: return the no-result outcome without calling Voyage or MongoDB search.
- Resolve each hit's `chunk_id` to `chunks` and then `section_id` to `sections`, as `docs/architecture.md` describes. A hit that does not resolve is left out and counted in `trace["unresolved_hits"]`.
- Return results ordered by score, highest first, with ties broken by `chunk_id`, at most `limit` items. Do not merge, de-duplicate, rewrite, or summarise passages.
- Each `RetrievedChunk` carries: `chunk_id`, `section_id` (act-qualified, for example `bns:103`), `act`, `text` (the chunk text from `chunks`), `heading` (the section heading, `null` if the section has none; do not invent a title), `score` (the vector search score), and `source` (a dict). `source` holds, copied from `chunks`, `sections` and `sources` and left `null` when absent: `section_label` (for example "BNS section 103"), `act_label`, `section_number`, `chapter`, `chapter_title`, `status`, `source_id`, `source_pdf`, `source_sha256`, `page_start`, `page_end`, `needs_review`, `chunk_index`. Never fill in a missing value.

### 5. Result shape and outcomes

Return the existing `QueryResult` with `pattern="semantic"` (the selected mode). Use these statuses, which must be easy to tell apart:

| Situation | `status` | HTTP | `results` |
|---|---|---|---|
| Passages found | `ok` | 200 | the passages |
| Valid question, the effective scope or database search matched nothing | `no_results` | 200 | empty |
| Not ready, missing configuration, Voyage or MongoDB failure | `unavailable` | 503 | empty |
| Invalid input | existing 422 handling | 422 | n/a |

`message` is plain and honest for each case, for example that scores are similarity rankings and are not proof the passage answers the question. For `unavailable`, name the missing setting or the failed step but never a secret, URI, or raw exception text. `trace` includes: `query`, `limit`, `num_candidates`, `effective_filters`, `embedding_model`, `embedding_version`, `index`, counts (`candidates_returned`, `unresolved_hits`), `ignored_fields`, and `caller_id`. `generation` stays `null`.

`/v1/query` stays the owner of these diagnostics. It does not leave `run_pattern`: wire the semantic branch into `run_pattern` so `/v1/query` and `/v1/chat/completions` still use one path. Map `unavailable` to HTTP 503 in the `/v1/query` route without changing its response model.

### 6. Models and chat

`rag-semantic` stays listed by `/v1/models` through the registry, so Open WebUI sees it. `/v1/chat/completions` keeps its current placeholder behavior: it sets `generate_answer=true`, and `run_pattern` must keep returning the existing Story 1.1 placeholder message for a request with `generate_answer=true`, with no retrieval, Voyage call, or MongoDB query. Retrieval runs for `/v1/query` requests with the default `generate_answer=false`. Do not add answer text, source lists, or citations to chat.

### 7. Diagnostic command

Add a command in the same module: `uv run python -m building_with_rag.retrieval.semantic "<question>" [--limit N] [--act X] [--status X] [--access-level X]`, plus `--check` for readiness only. It calls the same function `run_pattern` uses. Its output is compact: status, message, selected mode, query, filters, and for each result its rank, score (4 decimals), `chunk_id`, `section_id`, heading, `act`, source fields, and the passage length in characters. It does not print passage text or vectors. Exit non-zero for `unavailable`.

### 8. Tests (a few, offline)

Add a small number of offline tests (no network, MongoDB or key; fake the Voyage call and the collection): filter validation (accepts the three list fields, rejects unknown fields, operators and non-lists), the built `$vectorSearch` filter (narrows, never widens, empty intersection gives no-result without a call), and `run_pattern` with `generate_answer=true` keeps the placeholder. Existing tests keep passing. No broad suite.

### 9. Update `docs/architecture.md`

Keep existing text. Replace the "Retrieval path (built in a later story)" bullet and add a "Semantic retrieval" section recording: the flow (validate, readiness check, embed with `input_type="query"`, `$vectorSearch` with database-side filters, resolve `embeddings` to `chunks` to `sections`, build `RetrievedChunk`, order and limit), the three outcomes and their HTTP codes, the `source` fields, the fixed demo caller and public scope, the filter rules, the caveat that a score is a ranking and not proof, the chat placeholder, and the example diagnostic command from item 7. Record the limitation that an unrelated question still returns its nearest passages because no score threshold is applied.

## Token-safe checks

Every verification or data-check step below uses **one compact script or command that prints only counts, status, field names and values, and vector lengths**. Never print vectors, embeddings, full passage text, or raw documents, in scripts, logs, tests, or the handover. If a check needs the passage text, the script compares it in code and prints booleans and lengths. Do not paste API responses that contain passage text.

## Out of scope

Answer generation, Open WebUI answer behavior, hybrid or keyword search, re-ranking, query rewriting, a score threshold, an evaluation harness, broad testing, multi-user authorization, changes to ingestion, data, the vector index, `.env.example`, existing contracts, routes or the registry, and fixing corpus limitations from Story 2.1.

## Completion checks

### Checks the implementer runs (no services)

1. `uv sync`, `uv run ruff check .` and `uv run pytest` pass.
2. The offline tests from item 8 pass. With `MONGODB_URI` or `VOYAGE_API_KEY` empty, `uv run python -m building_with_rag.retrieval.semantic --check` reports `unavailable` with the missing setting name and shows no credentials.
3. `.env.example`, `data/raw/` and `data/processed/` are unchanged, and `.env` is untracked.

### Direct checks (need a populated database and Voyage key)

Run these if the services are available; otherwise report them as **pending** with what is missing. Never claim a pass that was not run. A free Voyage key is rate limited, so wait at least 20 seconds between checks that call Voyage.

1. **Readiness and a real question.** Run `--check`, then the command with one natural-language question, for example "What is the punishment for murder?" Expect `ok`, results from BNS and/or IPC, each with a `chunk_id`, an act-qualified `section_id`, a score, and source fields. Judge whether the passages are plausibly relevant; do not treat the score as proof.
2. **No-result.** Run the command with a filter that matches nothing, for example `--access-level restricted`. Expect `no_results` (status and message), not `unavailable`, and no Voyage call. Then run it with `VOYAGE_API_KEY` blanked for that run and expect `unavailable` naming the missing setting. The two outcomes must look different.
3. **Ordering and limit.** Run the same question with `--limit 3`. Expect at most 3 results, scores non-increasing, and with `--act BNS_2023` every result's `act` is `BNS_2023`. Run an invalid case (an unknown filter field through `/v1/query`) and expect 422.
4. **Source inspection.** Run one compact script that takes one returned `chunk_id`, reads its `chunks`, `sections` and `sources` documents, and prints only: whether the chunk text is contained in the section text, the two lengths, `section_id`, `heading`, `source_pdf`, `page_start`, `page_end`, and whether `source_sha256` matches the PDF hash in `data/raw/PROVENANCE.md`. Then open the PDF at the printed pages and confirm the section is there.
5. **Models and chat.** One compact script (a throwaway under the scratchpad, using the FastAPI test client, not committed) prints: `/v1/models` status and whether `rag-semantic` is listed, and for a `rag-semantic` chat request, the status code and whether the reply is still the placeholder message, plus that it contains no passage text.

## Handover

**The implementer fills in:**

- **Files created or changed.**
- **Commands actually run,** with results. State which direct checks were run and which are pending.
- **Command for you to run:** the diagnostic command and `--check` from item 7.
- **Deviations and open issues.**

**You fill in after running it:**

- **Readiness line:** counts of sections, chunks, embeddings, and the index status.
- **Checks 1 to 5:** what you saw (statuses, counts, ranks, scores, the source-check booleans and pages), and whether the passages looked relevant.
- **Known limitations:** an unrelated question still returns its nearest passages; scores are rankings, not correctness; sections flagged `needs_review` or left out in Story 2.1 are returned or absent as stored; a new index or document may take a short time to become searchable.

Story 3.1 will generate answers from these passages and give Open WebUI its answer text. Later modes reuse the same `QueryResult` and `RetrievedChunk` shapes.
