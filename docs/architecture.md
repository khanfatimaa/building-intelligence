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

## Out of scope

Multi-user authorization, a security programme, an evaluation harness, GraphRAG, agentic RAG, and broad deployment.
