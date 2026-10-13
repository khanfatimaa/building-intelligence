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

**Chunking later.** Story 2.2 chunks each section's `text`. Every chunk carries its parent `section_id`, `act`, `status`, `chapter`, `heading`, `source_pdf`, and `source_sha256`, copied from the section record. A chunk is a piece of one section and is never merged across sections or acts. The section record stays the source of truth.

## Out of scope

Multi-user authorization, a security programme, an evaluation harness, GraphRAG, agentic RAG, and broad deployment.
