# Story 2.2: MongoDB Chunks, Embeddings, and Vector Index

## Purpose

Turn the section records from Story 2.1 into searchable evidence in one MongoDB Atlas database: sections are chunked, each chunk is embedded with Voyage, and an Atlas Vector Search index is created over the embeddings.

**The implementer writes the scripts but does not run them against MongoDB or Voyage.** Do not run the load command (except `--help`), connect to Atlas, call Voyage, or create collections, documents, embeddings, or indexes. The person running the course runs the command and does the checks in "Checks to run yourself". The implementer may run only `uv sync`/`uv add`, `uv run ruff check .`, `uv run pytest`, and the no-credentials checks listed below.

This is the only goal. Do not add retrieval endpoints, answer generation, hybrid or keyword search, re-ranking, or a separate vector store. `/v1/query` and the other routes stay as they are (Story 1.1 placeholders). Story 3.1 will add semantic retrieval on top of what this story prepares.

## Prerequisites

- Stories 1.1 and 2.1 are complete: `uv sync` works and `data/processed/bns_sections.jsonl` (358 records) and `ipc_sections.jsonl` (493 records) exist. If either file is missing or invalid, stop and say to run `uv run python -m building_with_rag.ingest` first. Do not regenerate or edit them here.
- No `docs/config.yaml` exists, so use the repository root.
- A MongoDB Atlas free (M0) cluster. Assume Atlas Search index management and `$vectorSearch` are available, but verify (work item 2) and stop with a clear message if they are not.
- An untracked `.env` made from `.env.example`, with these settings and no others: `MONGODB_URI`, `MONGODB_DB_NAME`, `MONGODB_TEST_DB_NAME`, `VOYAGE_API_KEY`. The Atlas IP access list must allow this machine. Load them through the existing `Settings` in `src/building_with_rag/settings.py`. Do not rename, remove, or add settings, and do not change `.env.example`.
- If `MONGODB_URI` is empty or the server cannot be reached, stop with a message such as "MongoDB is unavailable: set MONGODB_URI in .env and allow your IP in Atlas." If `VOYAGE_API_KEY` is empty, setup, schema, chunking, and database loading can still run, but embedding and the sample query are reported as pending. Never print the URI, key, or any credential, including inside error messages and logs.
- Read `docs/architecture.md`, Stories 1.1 and 2.1, and the existing code under `src/building_with_rag/` (especially `settings.py`, `contracts.py`, and `ingest.py`) before writing anything. Preserve existing work, routes, and contracts. `contracts.py` already has a placeholder `ChunkDocument`; extend contracts only additively and never replace them.

## Fixed choices (from `docs/architecture.md`; do not guess or change)

- Embeddings: Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for documents. Document embeddings use `input_type="document"`; the sample query uses `input_type="query"`.
- Similarity: `cosine`.
- Access level: `access_level="public"` on every classroom record. Keep the field so the shape stays compatible with later work. Do not build access control.
- Section identifiers stay act-qualified (`bns:103`, `ipc:302`). Acts are never mixed.

## Work to do

### 1. Dependencies

Add only what is needed, through UV: the official Voyage client (`uv add voyageai`). `pymongo` is already present and must be recent enough for `create_search_index` with a `vectorSearch` type (it is, at `>=4.8`). Add nothing else. Update `pyproject.toml` and `uv.lock` through UV only.

### 2. Identify the deployment and its vector capability

At the start of setup, before writing anything:

1. Connect with a short timeout and run `ping`. On failure, stop with the prerequisite message above.
2. Confirm the deployment is Atlas and that search-index management works for this user, by calling the documented PyMongo search-index API (`list_search_indexes`) on the `embeddings` collection. Do not rely on guessing from the URI alone. A hostname ending in `mongodb.net` is a hint, not proof.
3. If the call fails because the deployment is not Atlas (for example a local `mongod`) or the user lacks the privilege (an authorization error), stop and report which prerequisite is missing. Do not fall back to a generic or legacy index, a brute-force scan, or another vector store.
4. Print one line recording the capability found, for example "Atlas Search index management available; vectorSearch index type used". Record the same exact wording in `docs/architecture.md`.

### 3. Schema module: `mongodb_schema.py`

Add a subpackage `src/building_with_rag/ingestion/` (with `__init__.py`) containing `mongodb_schema.py`. Leave `ingest.py` and its command unchanged. Everything later in this story and in later stories imports from this one module; do not repeat names, field lists, or IDs elsewhere. It holds:

- Collection-name constants: `SOURCES = "sources"`, `SECTIONS = "sections"`, `CHUNKS = "chunks"`, `EMBEDDINGS = "embeddings"`. The four collections are separate and are never merged. Vectors are never stored on `chunks`.
- Project constants: `EMBEDDING_MODEL = "voyage-3.5"`, `EMBEDDING_VERSION = "voyage-3.5"`, `EMBEDDING_DIMENSIONS = 1024`, `CHUNKING_VERSION` (for example `"chunk-v1"`), `VECTOR_INDEX_NAME = "embeddings_vector_index"`, `VECTOR_FIELD = "vector"`, `ACCESS_LEVEL_PUBLIC = "public"`, and the filter field names `act`, `chapter`, `status`, `access_level`.
- Deterministic ID functions, with no random or time-based parts, so a re-run produces the same IDs:
  - `source_id(act)` → for example `source:BNS_2023`
  - `section_id` comes from the 2.1 record (`bns:103`)
  - `chunk_id(section_id, chunk_index)` → for example `bns:103#0`
  - `embedding_id(chunk_id, model_version)` → for example `emb:bns:103#0:voyage-3.5`
  - Each document's `_id` is the same value as its own ID field.
- One `$jsonSchema` validator per collection and one function that creates each collection with its validator, or updates the validator if the collection already exists (`collMod`). Keep validators to required fields and types; do not build broad validation.
- Ordinary (non-search) indexes: unique on each collection's ID field; `sections` on `(act, section_number)`; `chunks` on `section_id` and on `(section_id, chunk_index)`; `embeddings` on `chunk_id`. `create_index` is idempotent when the definition is the same.
- A function that returns the exact vector-index definition used in item 6.

### 4. Document model

All four collections use this shape. Field names are exact.

**`sources`** (2 documents, one per supplied PDF): `source_id`, `act` (`BNS_2023` / `IPC_1860`), `act_label`, `status`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version` (may be null), `ingested_at`.
Copy the values from the 2.1 records. If records of one act disagree on `source_sha256`, stop and say the JSONL is inconsistent.

**`sections`** (851 documents, one per 2.1 record): `section_id`, `source_id` (link to `sources`), `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading` (may be null), `text`, provenance (`source_pdf`, `source_sha256`, `page_start`, `page_end`, `needs_review`), and `access_level` (`"public"`). Keep 2.1's `needs_review` as it is; do not "fix" the corpus here.

**`chunks`**: `chunk_id`, `section_id` (the parent), `chunk_index` (0-based within the section), `chunking_version`, `splitter` (name and settings, see item 5), `text`, `embedding_input_text`, `text_sha256`, denormalized `act`, `chapter`, `status`, and `access_level`.

**`embeddings`**: `embedding_id`, `chunk_id`, `embedding_model`, `embedding_version`, `embedding_dimensions`, `vector` (a list of 1,024 floats), `input_sha256` (hash of the exact embedded text), and the denormalized vector-filter fields `act`, `chapter`, `status`, `access_level`.

Section number and source identifiers are not denormalized onto `embeddings`. They stay on `chunks` and `sections` and are applied after `chunk_id` resolution.

Later semantic retrieval will query `embeddings` with `$vectorSearch`, resolve each returned `chunk_id` to its `chunks` document and then its `sections` document (through `section_id`), and build a `RetrievedChunk` from them. This story builds the data for that and does not build the retrieval.

### 5. Chunking (one command: `uv run python -m building_with_rag.ingestion.load`)

Add `src/building_with_rag/ingestion/load.py` as the command that does items 5 to 7. It reads the two JSONL files and uses the schema module.

- Chunk each section's `text` on its own, so a chunk never contains text from two sections or two acts.
- Target about 512 tokens, which is about 2,048 characters, with about 256 characters of overlap. A section of 2,048 characters or less is one chunk (`chunk_index` 0). Break on paragraph or sentence boundaries where possible and fall back to a plain character split. A small project-local splitter is enough; do not add a text-splitting dependency.
- Record the splitter on each chunk, for example `{"name": "building_with_rag.sentence_window", "chunk_chars": 2048, "overlap_chars": 256}`, together with `chunking_version`.
- `embedding_input_text` is the chunk text prefixed with its act label, section number, and heading (for example `Bharatiya Nyaya Sanhita, 2023, section 103. Murder.` then a blank line, then the text); omit the heading part when it is null. `text` stays the plain chunk text. Nothing is rewritten or summarised.
- Copy `act`, `chapter`, `status` from the section, and set `access_level="public"`.

### 6. Load, re-run, and change handling

Write in this order: `sources`, `sections`, `chunks`, then `embeddings`. Use upserts keyed by the deterministic `_id`, never plain inserts.

- Unchanged documents are left alone and counted as "reused". Re-running with no changes must create no duplicates and call Voyage zero times.
- A changed document (different section text or source hash, chunking version or splitter settings, or embedding model, version, or dimensions) must be visible. Print counts of new, reused, changed, and stale (no longer produced, such as higher chunk indexes after a section shrinks, or embeddings from another model version).
- By default, if anything is changed or stale, update nothing that would mix old and new data: print the counts and stop with the message "Re-run with `--accept-changes` to replace them." With `--accept-changes`, replace changed documents and delete stale chunks and embeddings. Changed chunk text always means its embedding is re-created.
- `--test-db` uses `MONGODB_TEST_DB_NAME` instead of `MONGODB_DB_NAME`. Note: a free M0 cluster allows only a few Atlas Search indexes, so the test database is for a brief trial and not required.

### 7. Embeddings: small batches, resumable

- Embed only chunks with no `embeddings` document whose `input_sha256`, model, version, and dimensions match. That is what makes the run resumable: an interrupted run is simply started again and continues where it stopped.
- Use small batches (for example 8 chunks), a delay between requests (for example 20 seconds), and retry on rate-limit errors with a growing wait and a bounded number of attempts. Make these named constants at the top of the module. After a failed batch, keep what is already stored and stop with a clear message.
- Write each batch to MongoDB as soon as it returns. Check that every returned vector has `EMBEDDING_DIMENSIONS` numbers before storing it.
- Print progress, for example `embedded 120/1013 chunks (batch 15/127)`. Never print vectors, keys, or request headers.
- **Expected duration.** About 1,000 chunks (the command prints the real count) on a free rate-limited Voyage key take about 40 minutes, matching the note in `.env.example`. A run where everything already exists takes under a minute. Tell the person running it before they start.

### 8. Vector index

Create the index after the `embeddings` collection exists (creating the collections is part of loading, so it can be done before or while embeddings are loading). The index goes on `embeddings.vector`.

- Use the documented PyMongo search-index API with `type="vectorSearch"`. The definition is a `fields` array and nothing else:
  - one `vector` entry: `path` is the vector field name from the schema module, `numDimensions` is `1024`, `similarity` is `cosine`;
  - four `filter` entries, with `path` set to `act`, `chapter`, `status`, and `access_level`.
- Do not use the legacy `knnVector` or `mappings` syntax, which belongs to `search`-type indexes. Take every field name from the schema module constants.
- Safe to repeat: if an index named `embeddings_vector_index` exists and its type and definition match, reuse it and say so. If it exists with a different definition or type, report the difference and stop. Do not delete, drop, or recreate it, and do not create a second index under another name.
- After creating it, wait a bounded time (for example up to 5 minutes, polling every 10 seconds) for the index to be queryable. If it is not ready, report "index not ready yet" and treat verification as pending. Do not hang.
- Run one real sample query once the index is ready and embeddings are stored: embed a fixed question (for example `What is the punishment for murder?`) with `input_type="query"`, then call `$vectorSearch` on `embeddings` with the index name, `path`, `queryVector`, a `numCandidates` of about 50, and a small `limit`. Use a filter on `status`/`act`/`access_level` that the filter fields allow (`public` is enough). Resolve the top result's `chunk_id` to `chunks` and then `sections`, and print `chunk_id`, `section_id`, `heading`, and the score. A newly stored document can take a short time to become searchable; retry briefly before reporting failure.
- If services or credentials are not available, report vector verification as **pending** and say what is missing. Do not claim it passed.

### 9. Tests (offline, small)

Add a few tests under `tests/` that need no network, MongoDB, or key: chunker (short section is one chunk; long section splits with overlap; no chunk crosses sections), deterministic IDs, and the vector-index definition (type `vectorSearch`, one `vector` entry with 1,024 dimensions and `cosine`, four filter paths with exact names). Existing tests keep passing.

### 10. Update `docs/architecture.md`

Keep existing text and add a "MongoDB collections and vector index" section recording: the four collections and their roles; the chunking choice (about 2,048 characters, 256 overlap, the splitter, `chunking_version`); the embedding decision (`voyage-3.5`, 1,024 dimensions, `access_level="public"`, cosine, vectors only on `embeddings`); the index name, type, and filter fields; the retrieval resolution path (`embeddings` → `chunks` → `sections`); the commands (the load command with `--accept-changes` and `--test-db`); the capability line from item 2; and the limitations (below). Fix the "Chunking later" paragraph so it matches the document model above: the fields `heading`, `source_pdf`, and `source_sha256` live on `sections` and are read through `section_id`, not copied onto chunks.

## Out of scope

Retrieval endpoints or functions that serve `/v1/query`, answer generation, hybrid or keyword search, re-ranking, a separate vector store, access control, deployment, broad corpus validation, fixing 2.1 corpus limitations, and any change to `.env.example`, existing routes, the registry, or existing contracts.

## Completion checks

### Checks the implementer runs

1. `uv sync` succeeds; `uv run ruff check .` and `uv run pytest` pass.
2. `uv run python -m building_with_rag.ingestion.load --help` works without touching MongoDB or Voyage.
3. With `MONGODB_URI` empty, the command stops with the clear prerequisite message and shows no credentials.
4. `.env` is untracked, `.env.example` and `data/raw/` and `data/processed/` are unchanged.
5. Do not run the live load. State plainly that MongoDB loading, embedding, index creation, and the sample query were **not run** and are pending for you.

### Checks to run yourself (classroom-sized)

1. Run `uv run python -m building_with_rag.ingestion.load`. It connects, prints the capability line, loads data, embeds with progress, and creates or reuses the index.
2. **Section-to-chunk link:** pick one section (for example `bns:103`) in `sections` and confirm its `chunks` have `section_id` equal to it, `chunk_index` starting at 0, and the same `act`.
3. **Embedding dimension:** take one `embeddings` document, confirm its `chunk_id` exists in `chunks`, `embedding_dimensions` is 1024, and `len(vector)` is 1024.
4. **Repeat setup:** run the command again. It reuses all documents and the index, makes no Voyage calls, and counts do not change (`sources` 2, `sections` 851, and `chunks` and `embeddings` equal in number).
5. **Index readiness:** in Atlas (or the command output), `embeddings_vector_index` is type `vectorSearch` and queryable.
6. **Sample result:** the sample query prints the top result's `chunk_id`, `section_id`, heading, and score, and the section is plausibly about murder or its punishment.

## Handover

**The implementer fills in:**

- **Files created or changed:** each file, including `pyproject.toml`, `uv.lock`, and `docs/architecture.md`.
- **Commands actually run:** each command with its result. Do not list commands that were not run. State plainly that the live Atlas and Voyage run was not done.
- **Command for you to run:** `uv run python -m building_with_rag.ingestion.load` (about 40 minutes the first time).
- **Deviations and open issues:** anything that differs from this story.

**You fill in after running it:**

- **Deployment:** the capability line printed, and index status.
- **Counts:** `sources`, `sections`, `chunks`, `embeddings`, and the number of chunks per act.
- **Checks 2 to 6:** what you looked at and what you found, including the sample result (`chunk_id`, `section_id`, heading, score).
- **Known limitations:** for example, sections flagged `needs_review` in 2.1 were loaded as they are, lettered and repealed-range sections left out by 2.1 are absent, and embeddings are only as good as the section text.

Story 3.1 will add semantic retrieval that queries `embeddings`, resolves `chunk_id` through `chunks` and `sections`, and returns `RetrievedChunk` items.

## Handover record (implementer)

- **Files created or changed:** `pyproject.toml` and `uv.lock` (`uv add voyageai`), `src/building_with_rag/ingestion/__init__.py`, `src/building_with_rag/ingestion/mongodb_schema.py`, `src/building_with_rag/ingestion/load.py`, `tests/test_mongodb_load.py`, `docs/architecture.md` (new "MongoDB collections and vector index" section; "Chunking later" paragraph corrected).
- **Commands actually run:** `uv add voyageai` (voyageai 0.5.0); `uv run ruff check .` (passed); `uv run pytest -q` (21 passed); `uv run python -m building_with_rag.ingestion.load --help` (works); a no-URI check inside pytest (stops with "MongoDB is unavailable"); an offline count of the corpus (2 sources, 851 sections, 939 chunks). The live Atlas and Voyage run was **not** done: no collections, embeddings, index, or sample query were created or run. `uv` is not on PATH here; it was run from the user-level Python Scripts folder.
- **Command for you to run:** `uv run python -m building_with_rag.ingestion.load` (about 40 minutes the first time; 939 chunks, 118 batches at 20 seconds apart).
- **Deviations and open issues:** none from the story. Untested against a real cluster: the exact `latestDefinition` shape Atlas returns for the compatibility comparison, and error code 26 handling for a missing collection in `list_search_indexes`.
