# Story 5.1: Structured Exact Retrieval

## Purpose

Make the `structured` mode (`rag-structured`) real. Semantic and hybrid retrieval rank passages by similarity or shared words, so a question such as "What does BNS section 103 say?" can return neighbours instead of the section itself. This story adds a **read-only exact lookup** over the existing `sections` collection: a small rule-based classifier (no LLM) turns the question into `StructuredSignals`, and only those validated signals build the MongoDB query.

Everything else is shared with the other modes: `QueryRequest`/`QueryResult`/`RetrievedChunk`, the registry, `/v1/query`, the chat adapter, context assembly, grounded generation, citations, confidence and Open WebUI streaming. No new endpoint, UI, setting or application. `semantic`, `hybrid` and `hybrid-reranked` must behave exactly as they do today.

The mode is chosen only by the `structured` pattern, the `rag-structured` model ID or `rag_options.pattern`. It is not an LLM extraction step, not a replacement for semantic search and not an automatic router. A found section is source text, not a statement that the law currently applies.

Not in this story: LLM extraction or classification, automatic mode choice or fallback to another mode, arbitrary or user-supplied database queries, write operations, executing `filter` or `aggregation` intents, ranges or multi-section lookups, lettered sections, permissions, new settings or indexes, new UI, decomposition, HyDE, an evaluation harness.

## Prerequisites

- Stories 1.1 to 4.2 are complete. First run `uv run pytest` and `uv run ruff check .`; if they fail, stop and report. Do not fix other stories here.
- Read first: `docs/architecture.md`, every story in `docs/stories/`, `contracts.py` (especially `StructuredSignals`, `QueryRequest`, `RetrievedChunk`), `registry.py`, `service.py`, `main.py` (the `/v1/query` route and chat adapter), `settings.py`, `ingestion/mongodb_schema.py` (`sections`, `sources`, `chunks` validators and the ordinary `sections (act, section_number)` index), `retrieval/semantic.py` (`_source`, `_database`, `Unavailable`, `retrieve`), `generation/context.py`, `generation/answer.py`, `docs/manual-tests.md`, `docs/master-prompt-update-manual-tests.md`, and `.env.example`.
- The database is already loaded (Story 2.2). A live check needs `MONGODB_URI` in the untracked `.env`; the answer path also needs the generation settings. This mode needs **no** Voyage, re-rank, vector or search index.
- Use the existing `.env.example` settings through `Settings` only (`MONGODB_URI`, `MONGODB_DB_NAME`, `WEBUI_DEMO_CALLER_ID`, and the generation settings for answers). Do not rename, add, remove or re-default any setting or edit `.env.example`.
- Facts about the stored data (confirm against `data/processed/*.jsonl` and `mongodb_schema.py` before coding): `sections.act` is `BNS_2023` or `IPC_1860`; `section_number` is an integer; `chapter` is a string such as `"I"`; `status` is `in_force` or `repealed`; `access_level` is `public`. Lettered sections (for example IPC 498A) and some repealed ranges are not stored.

## Contracts to preserve

Do not rename, replace or remove: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `SemanticFilters`, `StructuredSignals`, `GenerationResult`, `/v1/query` and its response model, the registry (six modes, `rag-<pattern>` IDs), `/v1/models`, the chat adapter with its streaming and `CAPSTONE_API_KEY` behavior, context assembly and citation rules, MongoDB collection names and validators, or `.env.example`. Changes are additive only.

- `StructuredSignals` keeps exactly `intent`, `act`, `section_number`, `chapter` with the Story 1.1 types (`section_number` is a string there; the integer conversion happens only in the validated query).
- `tests/test_semantic.py`, `tests/test_hybrid.py` and `tests/test_rerank.py` pass unmodified.
- `decomposition` and `hyde` stay `not_implemented`.
- `unavailable` stays HTTP 503 with empty `results` and no generation; `generation` outcomes never change the retrieval `status`.

## Work to do

### 1. The classifier (rule-based, no LLM)

Add `src/building_with_rag/retrieval/signals.py` with one pure function, for example `classify(question: str, chapter: str | None = None) -> Classification`. It uses fixed regular expressions or keyword rules only: no model call, no network, no database.

- **Output.** A `StructuredSignals` with `intent` limited to `exact_lookup`, `filter` or `aggregation`, or no signals at all with a `reason`. `act` is only `"BNS"` or `"IPC"` (case-insensitive; "Bharatiya Nyaya Sanhita" and "Indian Penal Code" may map to them). `section_number` is the digits as a string.
- **Exact lookup** only when the question names **exactly one** act and **exactly one** plain integer section, in phrasing such as "BNS section 103", "section 302 of the IPC", "IPC s. 302" or "what does BNS 103 say". Both pieces must be present.
- **Never guess.** These return no exact signals and a reason: no act ("section 103"); both acts ("BNS and IPC section 302"); more than one section or a range ("sections 100 to 105"); a lettered section ("IPC 498A": the letter is never dropped to match 498); a section number outside 1 to 999; no section at all.
- **Filter and aggregation.** Recognise only a small fixed set of phrases (for example "list the sections in chapter VI of the BNS" as `filter`, "how many sections ..." as `aggregation`) so they are reported as signals, but they are **not executed** in this story.
- **Chapter.** `QueryRequest.chapter` is the only chapter input; it is never parsed from the question for the query. Validate it as a short plain token (the existing filter-token pattern, or Roman numerals, whichever matches the stored values); an invalid value is a 422-style rejection through the existing validation or a `clarification_needed` result, never part of a query. It is passed through unchanged otherwise.
- Question text is trimmed (already done by `QueryRequest`) and is used by the classifier only. It never reaches MongoDB.

### 2. The exact lookup (read-only, signals only)

Add `src/building_with_rag/retrieval/structured.py` with `retrieve(request, *, settings=None, db=None)` (same injectable shape as semantic, so tests pass a fake `db`).

1. Run the classifier. If the signals are not a supported `exact_lookup`, return the outcome in item 3 **without any MongoDB call** (the database is not even opened).
2. Build the predicate from **validated values only**, in a fixed shape: `{"act": <BNS_2023|IPC_1860 mapped from the signal>, "section_number": <int>, "access_level": "public"}`, plus `status` only when the request's `filters.status` is given (validated tokens, `$in` built by the code), plus `chapter` only when the validated request chapter is given. Request `filters.act`, if given, may only narrow: if it excludes the classified act the scope is empty and the result is `not_found` with no database call. `filters.access_level` can only narrow the public base scope; an empty scope never queries. `required_acts` is ignored and listed in `trace["ignored_fields"]`.
3. One `find` on `sections` (the ordinary `(act, section_number)` index covers it). Nothing else is ever sent to MongoDB: no aggregation pipeline, no `$where`, no regex, no raw text, no update, insert or delete. The module contains no write call.
4. Resolve the section's `chunks` by `section_id` (ordered by `chunk_index`) and its `sources` record, and return one `RetrievedChunk` per chunk, built with the same shape and `source` fields as semantic (`section_label` such as "BNS section 103", `act_label`, `chapter`, `status`, `source_id`, `source_pdf`, `source_sha256`, pages, `needs_review`, `chunk_index`), plus `source_status_version` when present. `score` and all hybrid and re-rank fields stay `null`: an exact match has no similarity score. At most `request.limit` chunks are returned; `trace` records the chunk total and count returned. If a section has no chunk records, report it as `unavailable` with reason `section_without_chunks`; do not invent a chunk from the section text.
5. Status and version are carried as metadata exactly as stored (`in_force` or `repealed`, `source_status_version`). No message, field or prompt may claim the section is currently applicable law.

### 3. Outcomes (clear and safe; no guessing)

Use `QueryResult.status` strings with the existing HTTP rules (200 except `unavailable`).

| Situation | `status` | HTTP (`/v1/query`) | `trace["reason"]` |
|---|---|---|---|
| Exact record found | `ok` | 200 | none |
| Valid exact request, no matching record (or empty scope) | `not_found` | 200 | `no_matching_section` or `scope_excludes_request` |
| Act or section missing, both acts, several sections, a range, a lettered section, out-of-range number, invalid chapter | `clarification_needed` | 200 | `missing_act`, `missing_section`, `ambiguous_act`, `multiple_sections`, `lettered_section_unsupported`, `section_out_of_range`, `invalid_chapter` |
| `filter` or `aggregation` signals, or a question that is not a structured request at all | `recommendation` | 200 | `intent_not_executed` or `not_structured_request` |
| `MONGODB_URI` empty, or a MongoDB failure | `unavailable` | 503 | `missing_setting` (names `MONGODB_URI` only) or `database_error` |

- `clarification_needed` and `recommendation` return empty `results`, no MongoDB call and a plain message: what was understood, what is missing, an example such as "Try: 'BNS section 103'", and, for `recommendation`, that `semantic` or `hybrid` is the mode for a meaning-based question. The mode is only **suggested** in text; the application never switches to it.
- `not_found` says the section is not in the loaded corpus (some sections, for example lettered ones, are not loaded) and does not say it does not exist in law.
- No URI, secret or raw exception text in any message, trace or log.

### 4. Result shape and diagnostics (additive)

- `QueryResult.pattern="structured"`. No new top-level field and no new `RetrievedChunk` field.
- `trace` holds: `caller_id`, `ignored_fields`, `classifier` (`name`, the `StructuredSignals` dump or `null`, the `reason` and matched rule name), `predicate` (the four validated fields actually used, or `null` when no query ran), `section_id`, `chunks_total`, `chunks_returned`, `database_called` (boolean), and `reason` when not `ok`. No passage text.
- The `/v1/query` response shows this without generation, so direct inspection works with `generate_answer=false`.

### 5. Make the mode selectable; answer boundary

- `structured` / `rag-structured` already exist in the registry and `/v1/models`; do not add or rename modes. Change `run_pattern_events` in `service.py` so `structured` runs the new `retrieve` through the same single path as the other retrieval modes. The existing generation condition (`generate_answer`, `answer_generation`) stays as it is.
- **Direct inspection:** `generate_answer=false` (the `/v1/query` default) returns the `QueryResult` with no generation call.
- **Explanatory answer:** only when `generate_answer=true` **and** the status is `ok`, the same result goes to the existing `generate_events` operation. Context assembly, `E1`..`En` labels, citation validation, confidence, DRAFT/sources rendering and Open WebUI streaming are reused unchanged, so the answer cites the retrieved section. For `not_found`, `clarification_needed`, `recommendation` and `unavailable`, no model call is made; the result's `message` is the visible text (Open WebUI shows it as one readable message). Check `generation/answer.py` and context assembly for anything that assumes `semantic` or `hybrid` and change it only additively, without altering their output.
- Open WebUI always sends `generate_answer=true`, so chat shows the grounded answer for a found section and the message otherwise. Raw structured diagnostics stay in `/v1/query`.
- There is no new endpoint, UI, automatic routing, fallback from one mode to another, or permissions system.

### 6. Diagnostic command

`uv run python -m building_with_rag.retrieval.structured "<question>" [--chapter X] [--limit N]` calls the same `retrieve` function `run_pattern` uses and prints compact output: status, message, classifier signals and reason, the predicate, `database_called`, and per chunk `chunk_id`, `section_id`, `act`, status, heading, pages and text length. No passage text, keys or URIs. Exit non-zero for `unavailable`.

### 7. Tests (a few, offline)

No network, MongoDB, Voyage, model provider or server. Use a fake `db`. Cover only:

- the classifier: "BNS section 103" and "section 302 of the IPC" give `exact_lookup` with act and section; "section 103" (no act), "BNS and IPC section 302", "IPC 498A", "sections 100 to 105" and a plain meaning question give no exact signals and the right `reason`; one `filter` and one `aggregation` phrase are recognised;
- no-guess and read-only: for each non-exact case the fake `db` records zero calls; for an exact case it records exactly one `sections` find with only the validated predicate keys and values, and no write method is ever called; a question containing MongoDB-looking text (for example `{"$gt": ""}` beside "BNS section 103") never appears in the predicate;
- result shape: `ok` returns `RetrievedChunk` records with act-qualified `section_id`, source metadata including `status`, `score` null; a missing record is `not_found` (200 through `/v1/query`); no `MONGODB_URI` is `unavailable` (503);
- `run_pattern` with `pattern="structured"`: `generate_answer=false` makes no generation call; `generate_answer=true` with `ok` makes one generation operation over exactly the returned chunks; with `not_found` or `clarification_needed` it makes none.

Existing tests keep passing unmodified except those that assert `structured` is a `not_implemented` placeholder (`tests/test_smoke.py`, `tests/test_generation.py`); move those assertions to `decomposition`.

### 8. Documentation

- `docs/architecture.md`: keep existing text. Add a "Structured exact retrieval" section covering: the **exact-input contract** (the classifier is the only reader of the question; it emits `StructuredSignals` for exactly one BNS or IPC act and one plain integer section; the query is built only from the validated act, section number, optional validated chapter, optional status filter and the public access level; raw question text never becomes a predicate; one read-only `find` on `sections`, then `chunks` and `sources` by id; no writes, no arbitrary queries); the outcomes table; the **answer boundary** (the mode never generates by itself; an explanatory answer goes through the existing context/grounded-answer path only when requested and only for `ok`; direct inspection stays in `/v1/query` diagnostics; status and version are source metadata, never a claim of current applicability; citations and Open WebUI streaming are unchanged); the diagnostic command; and **Limitations** (rules cover only the phrasings listed, so other wording gets a recommendation; no lettered sections, ranges or multi-section lookups; `filter` and `aggregation` are classified but not executed; sections the corpus lacks are `not_found`; no automatic routing, so the user chooses the mode). Update only the lines that call `structured` a placeholder, including the "remaining modes" line under Using Open WebUI.
- `docs/manual-tests.md`: add a `## Story 5.1 — Structured Exact Retrieval` section following `docs/master-prompt-update-manual-tests.md`: a `/v1/query` curl for "BNS section 103", an ambiguous-section curl (no act) and a missing-record curl, and a `rag-structured` chat curl with `"stream": true` ending in `data: [DONE]`. Update the earlier line that lists `structured` as `not_implemented`. Do not run them as part of this item.

## Token-safe checks

Every check prints only statuses, counts, ids, headings, lengths and booleans. Never print keys, URIs or full passage text in scripts, logs, tests or the handover.

## Completion checks

### Checks the implementer runs (no services)

1. `uv sync`, `uv run ruff check .` and `uv run pytest` pass, including the item 7 tests and the unmodified semantic, hybrid and re-rank tests.
2. With `MONGODB_URI` empty, an exact question gives `unavailable` naming only `MONGODB_URI` with empty `results`, while an ambiguous question still gives `clarification_needed` (it never needs the database).
3. Reading the code shows one dispatch path through `run_pattern_events`, a single read-only query on `sections`, no write calls, and `generate_events` consuming only `QueryResult.results`.
4. `.env.example`, `data/raw/`, `data/processed/`, the indexes, the registry and `Settings` are unchanged.

### Direct checks (need the populated database; the answer check also needs the generation proxy)

Run if services are available; otherwise report them **pending** with what is missing. Never claim a pass that was not run.

1. **Valid exact lookup.** `uv run python -m building_with_rag.retrieval.structured "What does BNS section 103 say?"`, then the same through `/v1/query` with `pattern: "structured"`. Expect `ok`, `database_called` true, a predicate of `BNS_2023` and 103, results whose `section_id` is `bns:103`, and `status` shown as source metadata. Repeat once for an IPC section (for example "IPC section 302") and confirm it returns `ipc:302`, not the BNS section.
2. **Ambiguous section number.** "What does section 103 say?" Expect `clarification_needed`, reason `missing_act`, empty `results`, `database_called` false and a message asking for BNS or IPC. Confirm "BNS and IPC section 302" gives `ambiguous_act`.
3. **Missing record.** A valid request for a section the corpus lacks, such as "IPC section 600" or a lettered section only after confirming it is unloaded (a lettered request itself is `clarification_needed`). Expect `not_found`, HTTP 200, empty `results`, and wording that says "not in the loaded corpus".
4. **Answer boundary.** `/v1/query` with `generate_answer: true` for "BNS section 103" returns a grounded answer citing `bns:103`; the same question in Open WebUI (select **Building with RAG**, choose **structured**) streams the draft, confidence and sources as for the other modes; the missing-record and ambiguous questions in Open WebUI show their message with no model call.

## Handover

**The implementer fills in:**

- **Files created or changed.**
- **Classifier rules as implemented:** the supported phrasings, the reasons it can return, and the chapter validation chosen.
- **Predicate as implemented:** the exact fields sent to MongoDB and where `status`, `chapter` and `filters` narrow it.
- **Commands actually run,** with results. State which direct checks ran and which are pending and why.
- **Commands for you to run:** the diagnostic command, the `/v1/query` curls and the chat curl.
- **Deviations and open issues.**

**You fill in after running it:**

- **Checks 1 to 4:** what you saw (statuses, predicate, section ids, reasons, the Open WebUI display).
- **Known limitations:** the rules match only fixed phrasings; no ranges, lettered sections or multi-section lookups; `filter` and `aggregation` are not executed; found text is source text, not a claim of current law; the user must choose the mode.

Later stories add the remaining modes behind the same `run_pattern`, `QueryResult` and `RetrievedChunk`, additively.

## Handover record (implementer)

- **Files created or changed:** `src/building_with_rag/retrieval/signals.py`, `src/building_with_rag/retrieval/structured.py`, `src/building_with_rag/service.py` (structured dispatch; generation only for `ok`), `tests/test_structured.py`, `tests/test_smoke.py` and `tests/test_generation.py` (placeholder assertions moved to `decomposition`), `docs/architecture.md`, `docs/manual-tests.md`, this record.
- **Classifier rules as implemented:** `structured-rules-v1`. Exact lookup needs one act (BNS/IPC or the full names) and one integer section after "section(s)", "sec.", "s." or directly after the act ("BNS 103"). Reasons: `missing_act`, `missing_section`, `ambiguous_act`, `multiple_sections`, `lettered_section_unsupported`, `section_out_of_range`, `invalid_chapter`, `intent_not_executed`, `not_structured_request`. Chapter: Roman numerals (as stored, I to XXIII), upper-cased; anything else is `invalid_chapter`.
- **Predicate as implemented:** `{act, section_number (int), access_level: "public"}` plus `status: {$in: filters.status}` and `chapter` only when supplied. `filters.act` / `filters.access_level` can only empty the scope (`not_found`, no database call). Reads: one `find` on `sections`, then `chunks` by `section_id` and `sources` by id.
- **Commands actually run:** `uv run pytest` (88 passed, 8 new), `uv run ruff check .` (clean). Against the real database: the diagnostic command for "What does BNS section 103 say?" (`ok`, `bns:103`), "IPC section 302" (`ok`, `ipc:302`, stored status `repealed`), "IPC section 600" (`not_found`), "What does section 103 say?" (`clarification_needed`, `missing_act`, no database call), "BNS and IPC section 302" (`ambiguous_act`). `/v1/query` with `generate_answer: true` for BNS section 103 returned an `answered` generation, confidence 0.9; chat `rag-structured` with "section 103" returned the clarification message. Pending: Open WebUI display, and the empty `MONGODB_URI` outcome through the live app (covered offline by tests).
- **Commands for you to run:** the curls in `docs/manual-tests.md` under Story 5.1 and `uv run python -m building_with_rag.retrieval.structured "What does BNS section 103 say?"`.
- **Deviations and open issues:** none. `ruff format` was not applied (the repository was not formatter-clean beforehand). The Story 1.1 manual-test curl for `structured` now returns a `recommendation` (the text there notes it).
