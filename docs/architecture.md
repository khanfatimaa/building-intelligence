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
- **Single path:** `/v1/chat/completions` maps the selected model to the same `QueryRequest` and `run_pattern` as `/v1/query`. The server sets the demo caller and `generate_answer`.
- **Diagnostics:** `/v1/query` owns `QueryResult` diagnostics. Open WebUI receives only normal answer text derived from the same result.
- **Answer rendering:** later stories show final confidence, sources, and low-confidence warnings as clearly labelled text after the answer, while the full `GenerationResult` stays in `QueryResult.generation`.
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

**Caveats:** a score is a similarity ranking, never proof that a passage is correct or answers the question. No score threshold is applied, so an unrelated question still returns its nearest passages. `/v1/chat/completions` keeps its placeholder reply (it sets `generate_answer=true`, which skips retrieval) until answer generation.

**Diagnostic command** (compact output: statuses, ids, scores, lengths; no passage text or vectors):

```
uv run python -m building_with_rag.retrieval.semantic --check
uv run python -m building_with_rag.retrieval.semantic "What is the punishment for murder?" --limit 3 --act BNS_2023
```

## Out of scope

Multi-user authorization, a security programme, an evaluation harness, GraphRAG, agentic RAG, and broad deployment.
