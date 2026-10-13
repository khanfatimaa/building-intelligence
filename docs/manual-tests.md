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
