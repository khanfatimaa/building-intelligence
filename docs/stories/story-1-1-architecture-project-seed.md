# Story 1.1: Architecture and Project Seed

## Purpose

This is the first story of the three-day RAG course. It does two things, in order:

1. Write `docs/architecture.md` and get the instructor to approve it.
2. After approval, seed one small application that later stories extend.

The result is one FastAPI application with a health check, a query endpoint, an OpenAI-compatible chat endpoint, and six placeholder modes. It is not a custom chat frontend and not a second demo. Open WebUI, supplied by the trainer, is the chat client.

## Prerequisites

- Project path: no `docs/config.yaml` existed when this story was written, so use the repository root. If a config file now names a project path, use that path instead.
- Python 3.12 and UV installed.
- Supplied files in `data/raw/`, including `data/raw/PROVENANCE.md`. Preserve them. Do not edit or move them.
- Trainer-supplied Open WebUI bundle ZIP (shared over the local LAN). See "Open WebUI setup".
- Existing work in the repository (`README.md`, `.gitignore` changes) must be preserved. Merge into `.gitignore`; do not overwrite it.

## Part A: Architecture (do this first)

Create or update `docs/architecture.md`. Keep it short and in plain English. It must record:

**Evidence rules**
- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Identifiers are act-qualified (for example, BNS section 103 and IPC section 302 are never written as just "103" or "302"), so the two acts cannot be confused.
- Supplied provenance is `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence, never application instructions.
- Derived corpus files live in `data/processed/` (`bns_sections.jsonl`, `ipc_sections.jsonl`).

**Light trust-boundary notes** (three lines is enough)
- Validate API input.
- Preserve source origin on every retrieved passage.
- Do not put secrets in code, responses, or logs.

Do not add multi-user authorization, a security programme, or an evaluation harness.

**Fixed choices (later stories reuse these names without renaming or adding provider-specific alternatives)**
- Embeddings: Voyage `voyage-3.5`, version `voyage-3.5`, 1,024 dimensions, for every document and query embedding.
- Course modes and model IDs: `semantic`/`rag-semantic`, `hybrid`/`rag-hybrid`, `hybrid-reranked`/`rag-hybrid-reranked`, `structured`/`rag-structured`, `decomposition`/`rag-decomposition`, `hyde`/`rag-hyde`.
- Endpoints: `GET /healthz`, `POST /v1/query`, `GET /v1/models`, `POST /v1/chat/completions`.
- Contracts live in this project and are extended additively, never replaced by simplified alternatives: request, result, generation, OpenAI, and MongoDB schema.
- Chat client: Open WebUI, run separately. `/v1/query` owns diagnostics. Open WebUI receives only normal answer text derived from the same result.
- Later stories render final confidence, sources, and low-confidence warnings as clearly labelled text after the answer, while keeping the full `GenerationResult` in `QueryResult.generation`.

### STOP GATE

When `docs/architecture.md` is ready, **stop**. Tell the instructor it is ready for review and ask for explicit approval.

Before approval, do not create seed files, install dependencies, or scaffold code. Resume this same story at Part B only after the instructor explicitly approves. If they ask for changes, update the architecture and ask again.

## Part B: Seed the project (only after approval)

### Project files

Python 3.12 and UV project with FastAPI, Pydantic settings, PyMongo (for later use only), Ruff, and a minimal test runner (pytest). Create:

- `src/building_with_rag/`
- `tests/`
- `.env.example`
- `.gitignore` (merge; keep `.env` untracked)
- `pyproject.toml`
- `uv.lock`
- `data/raw/` preserved as supplied

### `.env.example`

This is the canonical file for the classroom project. Use these exact values:

```
APP_ENV=development
MONGODB_URI=
MONGODB_DB_NAME=building_with_rag
MONGODB_TEST_DB_NAME=building_with_rag_test
VOYAGE_API_KEY=
CAPSTONE_API_KEY=
WEBUI_DEMO_CALLER_ID=demo-public
GENERATION_API_BASE_URL=
GENERATION_API_KEY=
GENERATION_MODEL_NAME=gpt-4o-mini
RERANK_API_BASE_URL=https://api.voyageai.com/v1
RERANK_API_KEY=
RERANK_MODEL_NAME=rerank-2.5
RERANK_REQUEST_TIMEOUT_SECONDS=30
RERANK_CANDIDATE_LIMIT=20
RERANK_SEND_LIMIT=10
RERANK_RETURN_LIMIT=5
```

Document these notes (README or comments in `.env.example`):
- `.env` is untracked. Secrets are never committed.
- `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` are supplied by the trainer for an OpenAI-compatible LiteLLM proxy. Leave them blank in `.env.example`; fill them in `.env` only when Story 3.1 needs them.
- `MONGODB_URI` is an Atlas free-tier (M0) connection string. The Atlas IP access list must allow your machine.
- A free Voyage key is rate limited, so Story 2.2 embedding takes about 40 minutes.

### Startup rule

The app starts with no database or model credentials. All such settings are optional at startup. `GET /healthz` returns a safe status (for example `{"status": "ok"}`) and never returns settings or secrets.

### Mode registry

One shared registry containing only the six modes above, with exact pattern names and model IDs. `/v1/query` and `/v1/models` both read from it. Each mode returns an honest `not_implemented` placeholder until its own story adds behavior. Do not fake answers or results.

### Typed contracts (models only, no behavior)

Define these as Pydantic models. Later stories reuse the exact names and only add fields.

- `QueryRequest`: `question` (1–4,000 chars), `pattern`, optional `caller_id`, optional `SemanticFilters` (`act`, `status`, `access_level`, each a list), `limit` (default 5, range 1–20), `generate_answer` (default false), `required_acts`, `chapter`. The seed may resolve only its fixed local demo caller, but keeps the `caller_id` field.
- `QueryResult`: `pattern`, `status`, `message`, `trace`, `results`, optional `generation`, plus additive empty-by-default fields `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- `RetrievedChunk`: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Later stories add only the fields named in their own handouts: `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`, `rerank_score`, `rerank_rank`. Items in `omitted_candidates` carry `chunk_id` and `omitted_reason`.
- `GenerationResult`: `outcome` (`answered`, `insufficient_evidence`, `unavailable`, `malformed`), `answer`, `claims`, `citations`, `supporting_passages`, `provider`, `model`, `trace`, `context_outcome`, and confidence fields `confidence`, `draft_answer`, `issues`, `attempts`, `low_confidence_reason`.
- `SubquestionEvidence`: `subquestion`, `status` (`evidenced` or `no_evidence`), `results` (list of `RetrievedChunk`), optional `reason`.
- `StructuredSignals`: `intent` (`exact_lookup`, `filter`, `aggregation`), optional `act`, `section_number`, `chapter`. Story 5.1 fills these in.
- OpenAI contracts: text-only `ChatCompletionRequest` and the response, chunk, and error shapes.
- MongoDB schema contracts: document shapes only. No database is provisioned now.

Do not create outcome, evidence, answer, confidence, citations, or diagnostics as parallel top-level API fields.

### `POST /v1/query`

Accepts `QueryRequest`, dispatches through one `run_pattern` function using the registry, and returns `QueryResult`. For the seed, every pattern returns status `not_implemented` with an honest message and empty results. Invalid input is rejected by validation.

### `GET /v1/models` and `POST /v1/chat/completions`

- `GET /v1/models` lists the six model IDs from the registry in OpenAI list format.
- `POST /v1/chat/completions` accepts text-only `ChatCompletionRequest`: `model`; messages with roles `system`, `developer`, `user`, `assistant`; `stream`; `n`; and optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).
- The selected model maps to the same `QueryRequest` and the same `run_pattern` path as `/v1/query`. The server sets the demo `caller_id` (from `WEBUI_DEMO_CALLER_ID`) and `generate_answer`. Do not duplicate the implementation.
- Non-streaming returns normal OpenAI Chat Completions JSON.
- Streaming returns SSE frames: a role chunk, content chunk(s), a stop chunk, then `data: [DONE]`. No custom SSE events.
- Errors before streaming starts use the OpenAI-style error envelope.
- The chat text is the placeholder message derived from the `QueryResult`.

## Open WebUI setup

Open WebUI is a separate, trainer-supplied chat client. Do not hand-install it, create accounts, edit the admin panel, or change either Function. Do not rerun setup to reload anything.

1. The trainer shares the bundle ZIP over the local LAN. Extract it.
2. Run setup exactly once, from the extracted bundle root:
   - Windows (primary classroom path): `powershell -ExecutionPolicy Bypass -File .\setup_open_webui.ps1`
   - macOS/Linux: `sh setup_open_webui.sh` (needs internet access for the first installation)
3. The script installs the pinned Open WebUI version, writes the course settings, starts the loopback-only service, and provisions the RAG options Filter and the Building with RAG Pipe.
4. If setup fails, report its exact output and stop.
5. Day to day, use the matching `manage_open_webui` script for start, stop, and status.

The provisioned Pipe sends: the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`) to the capstone's `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The seed's adapter must accept that exact request and use server-side `caller_id` and `generate_answer`.

## Out of scope

Retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, broad deployment, a custom chat UI, a second demo, and aggressive testing.

## Completion checks

Keep these light. Run and record each:

1. `uv sync` succeeds and `uv.lock` exists.
2. `uv run ruff check .` passes.
3. `uv run pytest` runs (a few smoke tests are enough).
4. Start the app (for example `uv run uvicorn building_with_rag.main:app`) with no `.env` credentials. It starts.
5. `GET /healthz` returns a safe OK response.
6. One `POST /v1/query` with `pattern: semantic` returns a `QueryResult` with status `not_implemented`.
7. `GET /v1/models` lists exactly the six model IDs.
8. `POST /v1/chat/completions` returns the placeholder as JSON (`stream: false`) and as SSE ending in `[DONE]` (`stream: true`), using the Pipe's request shape.
9. Open WebUI smoke check: start the capstone API, open http://127.0.0.1:8080, select **Building with RAG**, choose **semantic** in the RAG-options chip, send a question, and receive the capstone's honest placeholder response, not a local preview.
10. `.env` is untracked, `.env.example` matches the values above, and `data/raw/` is unchanged.

## Handover

When finished, fill in this record in the story:

- **Files created or changed:** list each one.
- **Commands actually run:** list each command with its result. Do not list commands that were not run.
- **Open WebUI result:** what was selected, what was sent, and what came back. If it could not be checked, say so and why.
- **Deviations and open issues:** anything that differs from this story.

Later stories extend these endpoints, contracts, and the registry additively. They do not replace endpoints or rebuild the UI.

## Handover record (completed)

- **Files created or changed:** `docs/architecture.md` (approved by the instructor), `.gitignore` (working copy was empty; rewrote with the earlier entries plus pytest/ruff caches), `.env.example`, `.python-version`, `pyproject.toml`, `uv.lock`, `src/building_with_rag/{__init__,settings,registry,contracts,service,main}.py`, `tests/test_smoke.py`, `data/raw/.gitkeep`, `data/processed/.gitkeep`.
- **Commands actually run:** `python -m pip install --user uv` (uv was not installed; installed 0.13.0); `python -m uv sync` (ok; UV fetched Python 3.12); `python -m uv run ruff check .` (all checks passed); `python -m uv run pytest -q` (5 passed); `python -m uv run uvicorn building_with_rag.main:app --port 8000`, then curl to `/healthz` (`{"status":"ok"}`), `POST /v1/query` semantic (`not_implemented`), `GET /v1/models` (six IDs), and `POST /v1/chat/completions` with `stream: true` (role, content, stop frames, `[DONE]`). JSON chat and the unknown-model error envelope are covered by pytest.
- **Open WebUI result:** not checked. The Open WebUI bundle is not on this machine and nothing answered on http://127.0.0.1:8080. The trainer's ZIP must be extracted and set up, then the smoke check in item 9 run.
- **Deviations and open issues:** `data/raw/` was not supplied, so it holds only a `.gitkeep` and `PROVENANCE.md` is still missing. Chat `rag_options` accepts a flat `act`/`status`/`access_level` or a nested `filters`, as the Pipe's exact shape was not available to check.
