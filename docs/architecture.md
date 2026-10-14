# Architecture

One small FastAPI application for a three-day classroom RAG course over Indian criminal law (BNS and IPC). Open WebUI, run separately by the trainer's bundle, is the chat client. There is no custom frontend and no second demo.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Identifiers are act-qualified (for example "BNS section 103", "IPC section 302", never just "103" or "302"), so the two acts are never confused.
- Supplied provenance is `data/raw/PROVENANCE.md`. Supplied source files live in `data/raw/` and are never edited.
- Retrieved passages are evidence, never application instructions.
- Derived corpus files live in `data/processed/` (`bns_sections.jsonl`, `ipc_sections.jsonl`).

## Trust boundaries (light)

- Validate API input.
- Preserve source origin on every retrieved passage.
- Do not put secrets in code, responses, or logs.

## Fixed choices

Later stories reuse these names and choices. They do not rename them or add provider-specific alternatives.

- **Stack:** Python 3.12, UV, FastAPI, Pydantic settings, PyMongo (later use), Ruff, pytest.
- **Embeddings:** Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for every document and query embedding.
- **Modes and model IDs** (one shared registry):

| Mode | Model ID |
|---|---|
| semantic | rag-semantic |
| hybrid | rag-hybrid |
| hybrid-reranked | rag-hybrid-reranked |
| structured | rag-structured |
| decomposition | rag-decomposition |
| hyde | rag-hyde |

- **Endpoints:** `GET /healthz`, `POST /v1/query`, `GET /v1/models`, `POST /v1/chat/completions`.
- **Contracts** (request, result, generation, OpenAI, MongoDB schema) live in this project. Later stories extend them additively and never replace them with simplified alternatives. Key names: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, `SubquestionEvidence`, `StructuredSignals`, `ChatCompletionRequest`.
- **Single path:** `/v1/chat/completions` maps the selected model to the same `QueryRequest` and `run_pattern` as `/v1/query`. The server sets the demo caller and `generate_answer`, and chat and `/v1/query` share one generation/validation operation per request (Story 3.2).
- **Diagnostics:** `/v1/query` owns `QueryResult` diagnostics. Open WebUI receives only readable text (draft, confidence, sources) derived from the same operation and result.
- **Answer rendering:** Story 3.2 shows streamed drafts labelled `DRAFT`, then final confidence, sources, and low-confidence warnings as clearly labelled text, while the full `GenerationResult` stays in `QueryResult.generation`.
- **Startup:** the app starts with no database or model credentials. `GET /healthz` is safe and returns no settings.

## Document ingestion

Story 2.1 turns the two supplied PDFs into section-level JSONL records. One record is one section of one act. BNS and IPC are never mixed, so "section 103" in each act is two records.

- **Parser:** `pypdf` (version recorded in every record as `parser_version`). It is pure Python, needs no system libraries, and is enough for text-based legal PDFs. If it proves unsuitable on the real PDFs, it is swapped and the corpus regenerated.
- **Output** (UTF-8, one JSON object per line, ordered by section number): `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl`.
- **Fields:** `section_id` (`bns:<n>` / `ipc:<n>`), `act` (`BNS_2023` / `IPC_1860`), `act_label`, `status` (`in_force` / `repealed`), `chapter`, `chapter_title`, `section_number` (integer), `heading`, `text`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, `needs_review`. `page_start` and `page_end` are extra. `section_id`, `act`, `status`, `section_number`, and `text` are never null; any other value that cannot be found is `null` with `needs_review` set.
- **Command** (run by the person running the course, from the repository root): `uv run python -m building_with_rag.ingest`. It needs no database, network, or `.env`. It reports the PDF hash against `PROVENANCE.md`, skips an output file that is valid and matches the PDF hash, and otherwise rewrites the whole file through a temporary file, so re-runs never append or duplicate.
- **Known limitations** (found when the script was tried on the supplied PDFs; confirm after your own run):
  - Lettered sections (for example IPC 153A, 498A, 171A-171H) do not fit the integer `section_number`. They end the preceding section and are left out; the command lists them.
  - Repealed ranges printed as one line (IPC "161 to 165A", "478 to 489") are left out and listed. IPC sections 15 and 16 have no start line in the PDF text, so their placeholder text sits inside section 14 (flagged `gap_before_17`).
  - IPC amendment footnotes printed at page bottoms can remain inside a section's `text`. Such records are flagged `contains_footnote_text`.
  - Repealed IPC sections have a bracketed title and "Rep. by ..." instead of a heading; they keep that text and have a null `heading` with `needs_review` true.
  - The BNS PDF prints each heading above its section number. Headings come from the PDF's own index and are removed from the end of the previous section; a few that differ in wording are flagged.
  - No page had unusable text, so there is no OCR; pages that do are reported and their sections flagged.
  - The command's "missing section numbers" check compares against 358 (BNS) and 511 (IPC), a sanity check that is never written to a record.

**Chunking.** Story 2.2 chunks each section's `text`. A chunk is a piece of one section and is never merged across sections or acts. `heading`, `source_pdf`, and `source_sha256` stay on the `sections` record and are read through the chunk's `section_id`; the section record stays the source of truth.

## MongoDB collections and vector index

Story 2.2 loads the section records into one Atlas database (`MONGODB_DB_NAME`; `MONGODB_TEST_DB_NAME` with `--test-db`). Names, IDs, validators, ordinary indexes, and the vector-index definition live in `src/building_with_rag/ingestion/mongodb_schema.py`.

- **Collections** (separate, never merged; each `_id` equals its own ID field):
  - `sources`: one per PDF (`source_id` such as `source:BNS_2023`, act, label, status, PDF, hash, parser data).
  - `sections`: one per Story 2.1 record (act-qualified `section_id`, `source_id`, text, provenance, `access_level`).
  - `chunks`: `chunk_id` (`bns:103#0`), parent `section_id`, `chunk_index`, `chunking_version`, `splitter`, `text`, `embedding_input_text`, `text_sha256`, and denormalized `act`, `chapter`, `status`, `access_level`.
  - `embeddings`: `embedding_id` (`emb:<chunk_id>:voyage-3.5`), `chunk_id`, model, version, dimensions, `vector`, `input_sha256`, and denormalized `act`, `chapter`, `status`, `access_level`. Vectors are stored only here. Section number and source identifiers are not copied here.
- **Chunking choice:** about 512 tokens (2,048 characters) with 256 characters of overlap, splitter `building_with_rag.sentence_window` (paragraph, then sentence, then word boundaries), `chunking_version` `chunk-v1`. A section of 2,048 characters or less is one chunk. The supplied corpus gives 939 chunks. `embedding_input_text` is the act label, section number, and heading followed by the chunk text.
- **Embedding decision:** Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, `input_type` `document` for chunks and `query` for questions. `access_level` is `public` on every classroom record; the field is kept for later compatibility, with no access control built.
- **Vector index:** `embeddings_vector_index`, Atlas type `vectorSearch`, on `embeddings.vector` (1,024 dimensions, `cosine`), with `filter` fields `act`, `chapter`, `status`, `access_level`.
- **Capability check:** the command proves index management works with `list_search_indexes` and prints "Atlas Search index management available; vectorSearch index type used". On a non-Atlas deployment or without the privilege it stops and reports the missing prerequisite.
- **Retrieval path:** see "Semantic retrieval" below. Section number and source filters are not index filters.
- **Command** (run by the person running the course, with an untracked `.env`): `uv run python -m building_with_rag.ingestion.load`. Options: `--accept-changes` replaces changed documents and deletes stale chunks and embeddings; `--test-db` uses the test database. Embeds in batches of 8 with a 20-second delay and rate-limit retries. The first run takes about 40 minutes on a free Voyage key; it is resumable, and a re-run with everything present makes no Voyage calls.
- **Re-runs:** unchanged documents are reused. If section text, chunking, or embedding settings changed, the command prints new/reused/changed/stale counts and stops until `--accept-changes` is given, so old and new data are never mixed silently.
- **Limitations:** an existing index named `embeddings_vector_index` with a different definition is reported and left alone, never deleted. Sections flagged `needs_review` in Story 2.1 are loaded as they are, and sections Story 2.1 left out are absent. A new index or document can take a short time to become searchable. An M0 cluster allows only a few Atlas Search indexes, so the test database index is for a brief trial.

## Semantic retrieval

Story 2.3 builds the `semantic` mode (`rag-semantic`) in `src/building_with_rag/retrieval/semantic.py`. It returns source passages in `QueryResult.results` only. Answer generation and Open WebUI answer text come later.

**Flow** (`run_pattern` calls it for `semantic` requests with `generate_answer=false`, so `/v1/query` is the diagnostic owner):

1. Validate: the question is trimmed and must be 1 to 4,000 characters; `limit` is 1 to 20 (default 5).
2. Readiness check, counts and metadata only: both acts in `sections`, `chunks` and `embeddings` equal in number, stored model, version and 1,024 dimensions match, and `embeddings_vector_index` is `vectorSearch`, matches the schema definition, and is queryable. A successful check is cached for 60 seconds.
3. Embed the question with Voyage `voyage-3.5`, 1,024 dimensions, `input_type="query"` (one bounded attempt, no retry loop).
4. `$vectorSearch` on `embeddings` with a database-side filter built from validated lists, `limit`, and `numCandidates` of `max(limit * 10, 50)` capped at 200 (constants in the module).
5. Resolve each hit's `chunk_id` to `chunks` and then `section_id` to `sections` (and `sources`), build a `RetrievedChunk`, order by score descending (ties by `chunk_id`), and return at most `limit`. Hits that do not resolve are left out and counted in `trace["unresolved_hits"]`.

**Outcomes:** `ok` (HTTP 200, passages), `no_results` (HTTP 200, empty `results`; the effective scope or search matched nothing), `unavailable` (HTTP 503, empty `results`; not ready, a missing setting such as `VOYAGE_API_KEY`, or a Voyage or MongoDB failure; `trace["reason"]` names it and no secret is shown), and invalid input (HTTP 422).

**Result fields:** `chunk_id`, act-qualified `section_id`, `act`, chunk `text`, `heading` (the section heading, `null` if none), `score`, and `source` (`section_label`, `act_label`, `section_number`, `chapter`, `chapter_title`, `status`, `source_id`, `source_pdf`, `source_sha256`, `page_start`, `page_end`, `needs_review`, `chunk_index`; `null` when absent). `trace` has the selected `query`, `limit`, `num_candidates`, `effective_filters`, model, index, counts, `ignored_fields` and `caller_id`.

**Scope and filters:** one fixed local demo caller (`WEBUI_DEMO_CALLER_ID`) and the `public` access level; no multi-user authorization. `SemanticFilters` accepts only `act`, `status` and `access_level`, each a non-empty list of short plain tokens. Unknown fields, operators and MongoDB-shaped values are rejected. Filters only narrow the public scope, and are part of the `$vectorSearch` stage. An empty scope (for example `access_level=["restricted"]`) is `no_results` without a Voyage or MongoDB search call. `chapter` and `required_acts` are ignored by this mode.

**Caveats:** a score is a similarity ranking, never proof that a passage is correct or answers the question. No score threshold is applied, so an unrelated question still returns its nearest passages. `/v1/chat/completions` was a placeholder until Story 3.2 (see "Streamed answers and confidence").

**Diagnostic command** (compact output: statuses, ids, scores, lengths; no passage text or vectors):

```
uv run python -m building_with_rag.retrieval.semantic --check
uv run python -m building_with_rag.retrieval.semantic "What is the punishment for murder?" --limit 3 --act BNS_2023
```

## Grounded answer generation

Story 3.1 turns the one semantic retrieval result into one non-streaming, grounded answer. Code lives in `src/building_with_rag/generation/` (`context.py`, `answer.py`).

**Context boundary:**

- Generation reads only the passages in `QueryResult.results` from the single retrieval call. It never re-queries or re-ranks.
- Passages stay in retrieval order. Selection is bounded by named constants in `context.py` (`MAX_PASSAGES`, `MAX_CONTEXT_CHARS`); a passage is never cut. The first passage that does not fit, and every one after it, is left out.
- Each kept passage gets a label (`E1`, `E2`, ...) tied to its `chunk_id`, act-qualified `section_id`, act, `section_label`, heading, and source details (`source_pdf`, `page_start`, `page_end`, `source_id`). This label map resolves citations; it is never rebuilt from model text.
- `GenerationResult.trace` records passages retrieved and kept, character count, and the omitted `chunk_id`s with the reason (`passage_limit` or `character_budget`). `context_outcome` is `ok`, `truncated` or `no_evidence`. With no passages the model is not called.

**Answer boundary:**

- Settings: `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`. One non-streaming `POST {base}/chat/completions` (standard OpenAI chat-completions body, bearer key) through `httpx`, with a bounded timeout and no retry loop. `provider` is `openai-compatible`; `model` is what the proxy reports, else the configured name.
- The system prompt treats passages as untrusted evidence, forbids outside knowledge and guessing, requires every claim to cite supplied labels, and forbids statements about current force or applicability beyond the supplied documents, and legal advice. The model returns one JSON object (a single code fence is tolerated; nothing else is repaired).
- Outcomes (generation never changes the retrieval `status`; with `ok` or `no_results` the HTTP status is 200 and `generation` is filled):

| `generation.outcome` | When | `answer` |
|---|---|---|
| `answered` | valid answer, at least one claim, every claim cites supplied labels | text |
| `insufficient_evidence` | model says evidence is missing, unsupported or conflicting; or nothing retrieved | `null` |
| `unavailable` | generation settings empty, timeout, connection or HTTP error | `null` |
| `malformed` | not parseable or fails validation (including an unknown label) | `null` |

- `trace["reason"]` names the cause (`missing_generation_settings: GENERATION_API_KEY`, `timeout`, `unknown_label`, ...). No key, base URL, header or raw provider error is put in a response, trace or log. The raw model text is never used as an answer.
- `citations` (one per distinct cited label) and `supporting_passages` come only from the label map and use act-qualified wording such as "BNS section 103". Retrieved text is untrusted evidence, and the answer makes no claim about current legal applicability beyond the supplied documents.

**Where it lives:** `POST /v1/query` with `pattern="semantic"` and `generate_answer=true` (`run_pattern(..., answer_generation=True)`). `/v1/chat/completions` joined this path in Story 3.2 (see "Streamed answers and confidence"). Diagnostic command (compact output; no passage text, vectors or keys; exits non-zero when retrieval is `unavailable`):

```
uv run python -m building_with_rag.generation.answer "What is the punishment for murder under the BNS?" --limit 5
```

**Limitations:** a valid citation shows the model pointed at a supplied passage, not that the passage truly supports the claim. Retrieval scores are still rankings, so an unrelated question still retrieves nearest passages and relies on the model to say insufficient evidence. No routing or follow-up memory; streaming and confidence arrive in Story 3.2.

## Streamed answers and confidence

Story 3.2 connects `/v1/chat/completions` (and so Open WebUI) to the Story 3.1 generation. Code: `generation/answer.py` (`generate_events`, `call_model_stream`), `service.py` (`run_pattern_events`, `run_pattern`) and the chat adapter in `main.py`.

**One operation.** `/v1/query` and chat both go through `run_pattern_events`: semantic retrieval once, context assembly, then one generation/validation operation. It yields text events and ends with the final `QueryResult` (with its nested `GenerationResult`). `run_pattern` returns only that final result; chat renders the same text events. For one request exactly one operation runs, so the streamed text and the returned result describe the same LLM run. `stream: false` chat returns the same text joined into one message. `/v1/query` stays the richer view and uses the non-streaming model call; streamed chat uses the provider's streaming call (`"stream": true`, same prompt, attempt loop, checks and result building).

**What is streamed.** Only answer writing. Retrieval and context assembly finish before the first chunk. The model still returns the Story 3.1 JSON object (`outcome` first, `answer` second); a display-only extractor decodes the `answer` string as it arrives, and the complete text is always parsed and checked afterwards. SSE is standard: a role chunk, `delta.content` chunks, a `finish_reason: "stop"` chunk, then `data: [DONE]`. If the provider fails after text began, a final line `UNAVAILABLE — the text above is an incomplete draft, not a final answer.` is sent and the stream is still closed. A retrieval failure or empty result is sent as one readable message.

**Attempts and confidence.** After each attempt the text is checked; `MAX_ATTEMPTS = 2` (one retry, which lists the failed checks for the model). `GenerationResult` fields:

| Field | Meaning |
|---|---|
| `confidence` | `HIGH_CONFIDENCE` (0.9) when an attempt passed every check; `LOW_CONFIDENCE` (0.2) when answer text was judged and the last attempt failed; `null` when no answer text could be judged. A coarse label, not a probability. |
| `attempts` | attempts made (at most 2) |
| `issues` | every failed check of every attempt, in plain language (`attempt 1: claim 2 cites E9, which was not supplied`) |
| `draft_answer` | the last judged but unvalidated answer text; only when the final check failed |
| `low_confidence_reason` | short reason when the final check failed |

A validated answer is `outcome="answered"`. A draft that still fails after the last attempt is `outcome="malformed"` with `answer=null`, `draft_answer` set, low confidence, and no citations or supporting passages (nothing is presented as validated). Text that cannot be judged (not JSON, no answer string) has no confidence and reports the real failure; provider failures are `unavailable`; the model's own "insufficient evidence" is unchanged. `trace["attempt_log"]` keeps, per attempt, the failed checks and the labels the model cited exactly as given, including invalid ones; nothing is stripped.

**Checks** (deterministic, no second LLM call): every claim cites at least one label and every label was supplied; each claim's key words mostly appear in its cited passages (`SUPPORT_MIN_OVERLAP`); numbers in a claim appear in the cited evidence; every act-qualified section the answer mentions ("BNS section 103") is among the supplied passages. These are weak heuristics. A validated answer means the checks passed, not that the law is correctly stated or applies to anyone.

**Open WebUI display** (text only, built from the same events):

- Each attempt starts with `DRAFT — checking evidence`, because Open WebUI cannot retract text already streamed.
- A failed check with a retry coming shows `Check failed: ... Retrying once (attempt 2 of 2)`, then the next draft.
- A passing attempt closes with `Checks passed. Confidence: high.`, a `Sources:` list (act-qualified section, PDF and pages), and the attempts and failed-check counts.
- A failed final check closes with `DRAFT — low confidence, not the final answer.`, the reason, the failed checks and the draft's cited sources marked unverified.
- Insufficient evidence, unavailable and unusable-output cases show their own distinct messages.

**Authentication.** When `CAPSTONE_API_KEY` is set, `/v1/chat/completions` requires `Authorization: Bearer <CAPSTONE_API_KEY>` and answers 401 (`invalid_api_key`) before any retrieval or model call. When it is empty chat stays open. `/healthz`, `/v1/models` and `/v1/query` are unchanged. The key is never logged or returned. Whether the trainer's Pipe sends this header is confirmed in Open WebUI, not by the offline tests.

**Using Open WebUI.** Start the API (`uv run uvicorn building_with_rag.main:app --port 8000`), open `http://127.0.0.1:8080`, select **Building with RAG**, choose **semantic** in the RAG-options chip, and ask a question. Watch for the draft label, any retry line, and the closing confidence and sources block. Do not change the bundle, admin panel or Functions. Choose **hybrid** for Story 4.1's hybrid mode; the remaining modes still reply that they are not implemented.

**Limitations:** a draft is visible before it is validated and cannot be retracted; the lexical support check can be too strict or too lenient; retrieval scores are still rankings; no agentic behavior, new modes or evaluation harness.

## Hybrid retrieval

Story 4.1 builds the `hybrid` mode (`rag-hybrid`) in `src/building_with_rag/retrieval/hybrid.py`. It adds a keyword route beside the semantic route over the same corpus and database. Everything after retrieval is shared with `semantic`: `QueryResult`/`RetrievedChunk`, context assembly, citations, `/v1/query`, the chat adapter, streaming and confidence. `run_pattern_events` dispatches `semantic` and `hybrid` the same way; the mode is chosen only by the model ID or `rag_options.pattern`, never automatically.

**Keyword capability.** Atlas Search (a `search`-type index queried with `$search`). The setup command prints "Atlas Search available; search index type used for keyword retrieval" after proving `list_search_indexes` works on `chunks`. Without that capability it stops and reports the prerequisite; `hybrid` is then `unavailable` and `semantic` is unchanged. No `$text`, regex or client-side fallback.

**Keyword field and index.** Field: `chunks.text` (the plain chunk text, not `embedding_input_text`). Index: `chunks_text_index`, type `search`, static mapping (`dynamic: false`): `text` as `string` with analyzer `lucene.standard` (no stemming, so exact terms and numbers stay literal), and `act`, `status`, `access_level` as `token` so the scope filter runs inside `$search`. Names and `keyword_index_definition()` are in `ingestion/mongodb_schema.py`.

**Routes.** Both routes use the same validated scope (public base scope narrowed by `act`, `status`, `access_level`), the same candidate count per route (`max(limit, min(max(limit * 4, 20), 50))`), and resolve hits through `chunks`, `sections` and `sources` as semantic does. The semantic route is the Story 2.3 `$vectorSearch`. The keyword route is `$search` with a `compound` (`must`: `text` operator on the trimmed question over `text`; `filter`: `in` clauses from the scope), then `$limit` and `searchScore`. Keyword scores and cosine scores are on different scales and are never combined directly.

**Fusion: Reciprocal Rank Fusion, equal weights.** `fused_score = sum over routes containing the chunk of 1 / (RRF_K + rank_in_route)`, with `RRF_K = 60` and 1-based ranks from each route's own order. A chunk in both routes sums both terms. Order by fused score, ties by `chunk_id`, then apply `limit`; candidates cut by the limit are listed in `omitted_candidates` (`below_limit`). RRF was chosen because it needs no tuning or score normalisation and works with two unlike scores.

**Result fields.** `RetrievedChunk` gains optional `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`. For hybrid, `score` equals `fused_score`; a route that did not return the chunk leaves its score and rank `null`, so the contributing route (both, semantic only, keyword only) is readable from the result. Semantic results keep these fields `null`. `trace` adds the keyword index, field and analyzer, `candidates_per_route`, `semantic_candidates_returned`, `keyword_candidates_returned`, `route_overlap`, `semantic_only`, `keyword_only`, per-route unresolved counts, and `fusion` (`name`, `k`, `weights`).

**Outcomes** match semantic: `ok`, `no_results` (HTTP 200), `unavailable` (HTTP 503), 422. `unavailable` also covers a keyword index that is missing, incompatible or not queryable (`keyword_index_missing`, `keyword_index_incompatible`, `keyword_index_not_ready`). Hybrid never degrades silently to one route.

**Commands:**

```
uv run python -m building_with_rag.ingestion.keyword_index
uv run python -m building_with_rag.retrieval.hybrid --check
uv run python -m building_with_rag.retrieval.hybrid "What is culpable homicide?" --limit 5
```

**Limitations:** fused scores are rankings, not proof of correctness; equal weights and `RRF_K` are judgments, not tuned; a chunk found by one route can still rank high; rank fusion ignores how strong a score was; no score threshold, so an unrelated question still returns its nearest passages. The keyword route searches `chunks.text` only, so a section number and heading (which are only in `embedding_input_text`) are not keyword-searchable; "section 302" matches only chunks whose text contains that number. The `text` operator matches any query term, and `lucene.standard` removes no stop words, so a long natural-language question can pull in weak keyword hits. A new index may take a short time to become queryable, and an M0 cluster allows only a few Atlas Search indexes.

## Out of scope

Multi-user authorization, a security programme, an evaluation harness, GraphRAG, agentic RAG, and broad deployment.
