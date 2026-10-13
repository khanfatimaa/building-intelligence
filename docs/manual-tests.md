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
