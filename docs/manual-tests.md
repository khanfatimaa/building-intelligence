# Manual tests

## Story 1.1 — Architecture and Project Seed

Adds a FastAPI app with a health check, `/v1/query`, an OpenAI-compatible chat endpoint, and six placeholder RAG modes.

Start the API first (no `.env` credentials needed): `uv run uvicorn building_with_rag.main:app --port 8000`

```bash
curl http://127.0.0.1:8000/healthz
```

Expect `{"status":"ok"}`.

```bash
curl http://127.0.0.1:8000/v1/models
```

Expect a list of exactly six IDs: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder?", "pattern": "semantic"}'
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder?", "pattern": "hybrid"}'
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder?", "pattern": "hybrid-reranked"}'
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder?", "pattern": "structured"}'
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder?", "pattern": "decomposition"}'
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder?", "pattern": "hyde"}'
```

Each returns status 200 with `"status": "not_implemented"`, the matching `pattern`, an honest message ("... mode is not implemented yet. No retrieval was run."), and empty `results`.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "", "pattern": "semantic"}'
```

Expect HTTP 422 with a `detail` list naming `question` (empty string rejected).

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "rag-semantic", "stream": false, "messages": [{"role": "user", "content": "What is the punishment for murder?"}]}'
```

Expect a `chat.completion` JSON whose `choices[0].message.content` is the semantic not-implemented placeholder and `finish_reason` is `stop`.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "rag-semantic", "stream": true, "messages": [{"role": "user", "content": "What is the punishment for murder?"}], "rag_options": {"pattern": "semantic", "limit": 5}}'
```

Expect SSE frames: a role chunk, a content chunk with the placeholder, a stop chunk, then `data: [DONE]`.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "rag-unknown", "messages": [{"role": "user", "content": "hi"}]}'
```

Expect HTTP 404 with an OpenAI-style error envelope: `error.code` is `model_not_found`.

## Story 2.3 — Semantic Retrieval

Adds the `semantic` mode: `/v1/query` embeds the question and returns matching source passages in `results` (no generated answer).

Needs the Story 2.2 data and an untracked `.env` with `MONGODB_URI` and `VOYAGE_API_KEY`. Start the API: `uv run uvicorn building_with_rag.main:app --port 8000`. A free Voyage key is rate limited, so wait about 20 seconds between commands that call it.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "limit": 3}'
```

Expect HTTP 200, `"status": "ok"`, `"pattern": "semantic"`, and at most 3 `results` ordered by `score` (highest first), each with `chunk_id`, an act-qualified `section_id`, `act`, `text`, `heading`, `score` and `source`. `trace` shows the query and filters. Scores are rankings, not proof of correctness.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "limit": 3, "filters": {"act": ["BNS_2023"]}}'
```

Expect HTTP 200, `"status": "ok"`, and every result has `"act": "BNS_2023"`.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder?", "pattern": "semantic", "filters": {"access_level": ["restricted"]}}'
```

Expect HTTP 200, `"status": "no_results"` and empty `results` (the filter leaves nothing in scope; this is not a service error).

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is theft?", "pattern": "semantic", "filters": {"$where": ["x"]}}'
```

Expect HTTP 422 with a `detail` list (unknown filter fields are rejected). If the database or `VOYAGE_API_KEY` is not configured, a valid request returns HTTP 503 with `"status": "unavailable"` and `trace.reason`.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "rag-semantic", "stream": false, "messages": [{"role": "user", "content": "What is the punishment for murder?"}]}'
```

Expect the same placeholder reply as Story 1.1: chat does no retrieval until answer generation exists.

## Story 3.1 — Grounded Answer Generation

Adds `generate_answer` to `/v1/query` for the `semantic` mode: the retrieved passages are sent once to the model, which returns one cited answer or an insufficient-evidence outcome in `generation`.

Needs the Story 2.3 setup plus `GENERATION_API_BASE_URL`, `GENERATION_API_KEY` and (optionally) `GENERATION_MODEL_NAME` in the untracked `.env`. Start the API: `uv run uvicorn building_with_rag.main:app --port 8000`. Wait about 20 seconds between commands that call Voyage.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder under the BNS?", "pattern": "semantic", "limit": 5, "generate_answer": true}'
```

Expect HTTP 200, `"status": "ok"`, `generation.outcome` `"answered"`, a non-empty `generation.answer`, `claims` each with `labels`, and `citations` with act-qualified `section_id` (for example `bns:103`), `chunk_id` and `page_start`/`page_end`. `provider` is `openai-compatible`, `context_outcome` is `ok` or `truncated`, and `results` still lists the retrieved passages.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the capital gains tax rate for a salaried employee?", "pattern": "semantic", "limit": 5, "generate_answer": true}'
```

Expect HTTP 200, `generation.outcome` `"insufficient_evidence"`, `generation.answer` `null`, no `citations`, and a message saying no answer was generated. Retrieval `results` still show the nearest passages.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "What is the punishment for murder under the BNS?", "pattern": "semantic", "limit": 5}'
```

Expect the Story 2.3 behavior: `generation` is `null` (no model call without `generate_answer`).

With `GENERATION_API_KEY` blank, repeat the first command: expect HTTP 200, `generation.outcome` `"unavailable"`, `trace.reason` `missing_generation_settings: GENERATION_API_KEY`, `answer` `null`, and the retrieved `results` still present (clearly different from `insufficient_evidence`).

Diagnostic CLI for the same path (prints outcomes, labels, citations and the answer, never passage text or keys):

```bash
uv run python -m building_with_rag.generation.answer "What is the punishment for murder under the BNS?" --limit 5
```

Expect `generation outcome: answered`, the label-to-`chunk_id` list, cited labels and resolved citations. It exits non-zero only when retrieval is `unavailable`.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "rag-semantic", "stream": false, "messages": [{"role": "user", "content": "What is the punishment for murder?"}]}'
```

Expect the Story 3.2 chat reply described below: the same generation as `/v1/query`, rendered as text.

## Story 3.2 — Streamed Answers with Confidence

Connects `/v1/chat/completions` to the same semantic retrieval and generation as `/v1/query`, streaming the answer text with a confidence and sources block.

Needs the Story 3.1 setup. If `CAPSTONE_API_KEY` is set in `.env`, add `-H "Authorization: Bearer <your CAPSTONE_API_KEY>"` to the chat commands. Start the API: `uv run uvicorn building_with_rag.main:app --port 8000`.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "rag-semantic", "stream": true, "messages": [{"role": "user", "content": "What is the punishment for murder under the BNS?"}]}'
```

Expect HTTP 200 SSE chunks: a role chunk, content chunks starting `DRAFT — checking evidence`, a closing `Checks passed. Confidence: high.` block with BNS sources, a `finish_reason` `stop` chunk, and a final `data: [DONE]`.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "rag-semantic", "stream": false, "messages": [{"role": "user", "content": "What is the capital gains tax rate for a salaried employee?"}]}'
```

Expect normal Chat Completions JSON whose content says the retrieved passages do not give enough evidence (no draft, no sources, no confidence).

## Story 4.1 — Hybrid Search

Adds the `hybrid` mode: a semantic and a keyword route over the chunk text, fused by reciprocal rank fusion, through the same `/v1/query` and chat path as `semantic`.

Needs the Story 3.2 setup plus the keyword index: run `uv run python -m building_with_rag.ingestion.keyword_index` once. Start the API: `uv run uvicorn building_with_rag.main:app --port 8000`. Wait about 20 seconds between commands that call Voyage. If `CAPSTONE_API_KEY` is set, add `-H "Authorization: Bearer <your CAPSTONE_API_KEY>"` to the chat command.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "culpable homicide not amounting to murder", "pattern": "hybrid", "limit": 5}'
```

Expect HTTP 200, `"status": "ok"`, `"pattern": "hybrid"`, and `results` ordered by `fused_rank`, each with `score` equal to `fused_score` and `semantic_*` / `keyword_*` ranks and scores (`null` for a route that did not return the passage). `trace` shows `route_overlap`, `semantic_only`, `keyword_only` and the `fusion` rule.

```bash
curl -X POST http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question": "culpable homicide", "pattern": "hybrid", "filters": {"access_level": ["restricted"]}}'
```

Expect HTTP 200, `"status": "no_results"` and empty `results`. If the keyword index is missing or not queryable, a valid request returns HTTP 503 with `"status": "unavailable"` and a `trace.reason` such as `keyword_index_missing`.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "rag-hybrid", "stream": true, "messages": [{"role": "user", "content": "What is culpable homicide not amounting to murder?"}]}'
```

Expect HTTP 200 SSE chunks like Story 3.2 (draft label, confidence and sources block, a `stop` chunk), ending with `data: [DONE]`.
