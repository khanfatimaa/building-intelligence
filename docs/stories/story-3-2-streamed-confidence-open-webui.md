# Story 3.2: Streamed Answers with Confidence in Open WebUI

## Purpose

Connect the real RAG generation from Story 3.1 to the Open WebUI streaming path that Story 1.1 already established. After this story a question typed into Open WebUI runs the selected pattern (`rag-semantic` today), assembles context, writes an answer with the configured LLM, validates it, and shows the answer as it is written, followed by a readable confidence, sources and checks block.

`POST /v1/query` and `POST /v1/chat/completions` are two views of one operation, not two implementations. Both call the same `run_pattern` path, the same retrieval, context assembly, answer validation, citation resolution and final outcome, and both produce the same `QueryResult` with its nested `GenerationResult`. `/v1/query` stays the richer, learning-oriented view; Open WebUI receives readable text derived from that same result. There is no second chat path.

Only answer writing is streamed. Retrieval and context assembly finish before any text is sent.

Because Open WebUI cannot retract text already streamed, an answer is shown as a visibly labelled draft until its checks pass. A draft is never presented as a validated answer.

Not in this story: agentic behavior, new retrieval modes, UI plugins or changes to the Open WebUI bundle and its Functions, a broad test suite, an evaluation harness, multi-user authorization.

## Prerequisites

- Stories 1.1, 2.1, 2.2, 2.3 and 3.1 are complete. Story 3.1's code is in the working tree (`generation/context.py`, `generation/answer.py`, `tests/test_generation.py`); first run `uv run pytest` and `uv run ruff check .` and confirm they pass. If they do not, stop and report; do not fix other stories here.
- Read first: `docs/architecture.md`, Story 1.1 (especially "Open WebUI setup" and the `/v1/chat/completions` contract), Stories 2.1 to 3.1, `contracts.py`, `registry.py`, `service.py`, `main.py` (the chat adapter and `_sse`), `settings.py`, `generation/context.py`, `generation/answer.py`, `tests/test_generation.py`, `tests/test_smoke.py`, and `.env.example`, plus the supplied Open WebUI course bundle notes. There is no `docs/config.yaml`; use the repository root. The bundle notes are not in the repository; if you cannot see them, say so in the handover and rely on Story 1.1's description of the provisioned Pipe. Never edit, rerun setup for, or reconfigure the bundle, its Filter or its Pipe.
- Settings already exist in `.env.example` and `Settings`: `CAPSTONE_API_KEY`, `WEBUI_DEMO_CALLER_ID`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`. Use these exact names. Do not rename, replace, add or remove settings, change defaults, or edit `.env.example`.

## Contracts to preserve

- Endpoints, the shared registry (six modes, `rag-<pattern>` model IDs), `GET /v1/models`, model discovery, mode selection from the model ID and `rag_options`, `QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult` and its existing field names. Changes are additive only.
- Story 3.1 behavior of `/v1/query` stays: `generate_answer=false` gives `generation=null`; retrieval `unavailable` is HTTP 503 with no generation; other modes keep their `not_implemented` placeholders; generation outcomes never change the retrieval `status`.
- Story 3.1 outcomes and rules stay: `answer` is set only when `outcome="answered"`; citations resolve only from the context label map; retrieved text is untrusted evidence; no keys, base URLs, headers, raw provider errors or passage text in responses, traces, logs or the handover.
- The Pipe's request shape (`model`, `stream: true`, latest user message, `rag_options`) is accepted exactly as today. `stream: false` keeps returning normal Chat Completions JSON.

## Work to do

### 1. One operation, shared by both endpoints

- `run_pattern` stays the only dispatcher. Retrieval runs once, then context assembly, then one generation/validation operation. Both endpoints reach it; chat passes `answer_generation=True` like `/v1/query` (the Story 3.1 placeholder reason for the separate switch ends here; keep the argument if other code or tests rely on it).
- Restructure the generation core in `generation/answer.py` as one event-producing operation: it yields text events (answer text pieces, check/retry notes) and ends with the final `QueryResult`. The non-streamed paths (`/v1/query`, chat with `stream: false`, the diagnostic command) consume the same operation and use only its final result. `generate(result, question, settings=, call=)` keeps its name and signature, so Story 3.1 tests and the diagnostic command keep working.
- For one request exactly one operation runs. The streamed text and the returned final result describe the same operation, never two independent LLM calls. The chat text for `stream: false` and the concatenated streamed text for the same operation must be identical (one shared rendering function).
- `/v1/query` keeps using the existing non-streaming model call; streamed chat uses a streaming call. They share the same prompt, attempt loop, validation and result building.

### 2. Streaming the configured LLM

- Add a streaming model call beside `call_model`: `POST {GENERATION_API_BASE_URL}/chat/completions`, standard OpenAI body with `"stream": true`, bearer `GENERATION_API_KEY`, through `httpx`. Parse the provider's SSE `data:` lines and yield `choices[0].delta.content` pieces until `[DONE]`. Bounded timeouts as named constants. No provider SDK or replacement parameters. Same missing-setting, timeout, connection and HTTP-error mapping to `unavailable` as Story 3.1, never leaking secrets.
- The model still returns Story 3.1's JSON object, now with `outcome` first and `answer` second (update the system prompt order only; every other prompt rule stays). A small incremental extractor decodes only the `answer` string value as it arrives and emits it as answer text events. It is display-only: the complete model text is always parsed and validated afterwards. If an attempt parses as an answer but the extractor streamed nothing, emit the parsed answer once so stream and result still agree. When the outcome is `insufficient_evidence` no answer text is streamed.
- Emit the answer text as OpenAI-compatible chunks: a role chunk, `delta.content` chunks, a chunk with `finish_reason: "stop"`, then `data: [DONE]`. Reuse the existing `_sse` helper and chunk shape; no custom SSE events. Retrieval and context assembly finish before the first chunk, so a retrieval `unavailable` or empty result is sent as one readable text message (HTTP 200, as chat does today); errors before streaming starts still use the OpenAI error envelope.
- If the provider fails after answer text began, emit a final clearly labelled line, for example `**UNAVAILABLE — the text above is an incomplete draft, not a final answer.** <plain detail>`, then the stop chunk and `[DONE]`. The stream is always closed. The result is `outcome="unavailable"`; partial text is never validated or shown as an answer.

### 3. Bounded validation attempts and the confidence extension

Validate after each attempt. Use the existing `GenerationResult` fields with these meanings (document them in the architecture):

| Field | Meaning |
|---|---|
| `confidence` | `0.9` (named constant `HIGH_CONFIDENCE`) when an attempt passed every check; `0.2` (`LOW_CONFIDENCE`) when answer text was judged and the last attempt failed; `null` when no answer text could be judged. A coarse label, not a calibrated probability. |
| `attempts` | number of attempts made (named constant `MAX_ATTEMPTS = 2`: the first attempt and one retry; no loop beyond that) |
| `issues` | every failed check from every attempt, each a plain-language string such as `attempt 1: claim 2 cites E9, which was not supplied` |
| `draft_answer` | the last judged but unvalidated answer text; set only when the final check failed, otherwise `null` |
| `low_confidence_reason` | short plain-language reason when the final check failed, otherwise `null` |

Checks, deterministic and offline (no second LLM call), each recording a failure with plain detail:

- **Citations:** every claim has at least one label and every label was supplied.
- **Support (simple, documented as a weak heuristic):** the significant words of each claim mostly appear in its cited passage text (named threshold constant); numbers (years, counts) in a claim appear in the cited text; every act-qualified section reference in the answer (for example "BNS section 103") matches a supplied passage's `section_label`. Name thresholds as constants in `answer.py` so they can be tuned.

Flow:

- Attempt text that is not a JSON object, has no usable `answer` string, or has an invalid outcome cannot be judged. If no attempt could be judged, `confidence` is `null`, `outcome` is `malformed` (real reason in `trace["reason"]`) with no draft; a provider failure is `unavailable`; the model saying the evidence is insufficient is `insufficient_evidence`, as in Story 3.1. None of these retries.
- Judged and failed: retry once, appending a user message that lists the plain-language failed checks and asks for the same JSON format again. A passing attempt gives `outcome="answered"`, `confidence=HIGH_CONFIDENCE`, and the Story 3.1 fields (`answer`, `claims`, `citations`, `supporting_passages`). Earlier failed checks stay in `issues`.
- Still failing after the last attempt: `outcome="malformed"`, `answer=null`, `confidence=LOW_CONFIDENCE`, `draft_answer` set, `low_confidence_reason` set, `citations` and `supporting_passages` empty (nothing is presented as validated), and `QueryResult.message` says plainly that the draft failed its checks.
- Keep diagnostics: `trace["attempt_log"]` holds, per attempt, the number, whether it was judged, the failed checks, and the labels the model cited exactly as given, including invalid ones. Never strip an invalid citation or drop a failed check, and never reduce a failed answer to silence. `/v1/query` returns all of this; retrieval `status` and `results` are untouched.

### 4. What Open WebUI shows

Text only, in readable Markdown, produced by one rendering function from the same operation:

- Each attempt's streamed text starts with `**DRAFT — checking evidence**` on its own line, because validation happens after the text.
- On a failed check with a retry coming: a short line such as `**Check failed:** <plain detail>. Retrying once (attempt 2 of 2).`, then the next attempt, again prefixed as a draft.
- When an attempt passes: a closing block, separated from the answer, with `Checks passed. Confidence: high.`, then `Sources:` one line per citation with the act-qualified section (for example "BNS section 103") and pages when known, and a count of checks run and retries used. It states that the draft above is now the validated answer.
- When the final check fails: `**DRAFT — low confidence, not the final answer.**` plus the short reason, the failed checks in a few lines, and the sources the draft cited marked as unverified.
- `insufficient_evidence`, `unavailable` and `malformed` without judged text each show their existing Story 3.1 message, clearly different from one another.
- Keep it compact: no raw JSON, no passage text, no keys or provider errors. `GET /v1/models` and the six `rag-<pattern>` mode selection are unchanged; modes other than `rag-semantic` still return their honest `not_implemented` text.

### 5. Authentication with `CAPSTONE_API_KEY`

Use the existing `capstone_api_key` setting. When it is non-empty, `POST /v1/chat/completions` requires `Authorization: Bearer <CAPSTONE_API_KEY>`; a missing or wrong key gets an OpenAI-style 401 error envelope (`invalid_api_key`) before any retrieval or model call, with constant-time comparison and no key echoed. When it is empty the endpoint stays open, so the app still starts and works with no credentials. `/healthz`, `/v1/models` and `/v1/query` are unchanged. The key is never logged or returned. If the bundle notes or Story 1.1 do not show that the Pipe sends this header, do not guess; say so in the handover so the participant's Open WebUI check (below) settles it.

### 6. Tests (a few, offline)

No network, MongoDB, Voyage, model provider or running server. Fake retrieval and the model call (a scripted list of text pieces per attempt) and use the FastAPI test client in-process. Cover only:

- a validated first attempt: one retrieval and one model operation, a `DRAFT — checking evidence` prefix, then `confidence` high, `attempts=1`, citations resolved, valid SSE (role, content, stop, `[DONE]`), and streamed text equal to the `stream: false` content and consistent with the final result;
- a first attempt with an unknown label then a passing retry: `attempts=2`, the failed check kept in `issues` and `trace["attempt_log"]`, retry line shown, high confidence;
- both attempts failing: `outcome="malformed"`, `answer=null`, `draft_answer` set, low confidence, the closing `DRAFT — low confidence, not the final answer` line, invalid citations still visible in the trace, nothing marked validated;
- a provider failure after text began: the labelled unavailable line is emitted and the stream still ends with a stop chunk and `[DONE]`; unparseable output gives `confidence` null and a real failure outcome;
- auth: with `CAPSTONE_API_KEY` set, missing and wrong keys get 401 and no retrieval or model call; the right key passes; with it empty the request passes;
- `/v1/query` with `generate_answer=true` and chat call the same path with the same result shape, `generate_answer=false` still gives `generation=null`.

Update the two existing tests that asserted the chat placeholder (in `tests/test_generation.py` and `tests/test_smoke.py`) to the new behavior; existing tests otherwise keep passing. No broad suite or evaluation harness.

### 7. Documentation

- `docs/architecture.md`: keep existing text. Add a "Streamed answers and confidence" section covering the single shared operation, what is and is not streamed, the attempt loop and constants, the field meanings table above, the checks and their limits (a validated answer means the deterministic checks passed, not that the law is correctly stated), the draft labelling, the SSE shape and failure behavior, the `CAPSTONE_API_KEY` rule, and the Open WebUI display. Update the fixed-choice "Single path", "Diagnostics" and "Answer rendering" lines, and the 2.3 and 3.1 sentences that say chat is a placeholder, only as far as needed to stay consistent.
- Open WebUI instructions: in the architecture, add a short "Using Open WebUI" part (start the API, open `http://127.0.0.1:8080`, select **Building with RAG**, choose **semantic** in the RAG-options chip, what the draft, retry, confidence, sources and unavailable lines mean). Do not tell anyone to change the bundle, admin panel or Functions.
- `docs/manual-tests.md`: replace the placeholder sentence at the end of the Story 3.1 section and add a `## Story 3.2 — ...` section following `docs/master-prompt-update-manual-tests.md` (curl for `stream: true` and `stream: false`, stating the stream ends with `data: [DONE]`). These are for the participant; do not run them.

## Do not run live checks

The coding assistant (Claude, Codex or any other) must **not** use curl, Postman, `uvicorn`, `httpx` against a running server, or any other live HTTP or server call to verify this story, and must not call Voyage, MongoDB or the model provider. It verifies only with the offline tests above and `ruff`. The live smoke checks below belong to the participant and are reported as "pending participant". Never claim they passed.

## Out of scope

Agentic behavior or tool use, new retrieval modes or changes to retrieval, re-ranking, a second chat path, UI plugins or any change to the Open WebUI bundle, Filter or Pipe, a second LLM call for validation, calibrated confidence, follow-up memory, a broad test suite or evaluation harness, multi-user authorization, changes to ingestion, data, the vector index, `.env.example`, or the registry.

## Completion checks

### Checks the implementer runs (offline only)

1. `uv sync`, `uv run ruff check .` and `uv run pytest` pass, including the Story 3.2 tests above.
2. `.env.example`, `data/raw/`, `data/processed/` and the registry are unchanged; `.env` is untracked.
3. Reading the code shows one dispatch path: chat and `/v1/query` both call `run_pattern`, and one operation per request feeds both the stream and the final result.

### Participant checks (live, pending participant)

Run by the participant directly in the Open WebUI interface at `http://127.0.0.1:8080`, not in Postman (that only repeats Story 3.1's checks). Needs the populated database, `VOYAGE_API_KEY`, the trainer's generation proxy settings, `CAPSTONE_API_KEY` set in `.env` (the Pipe must be able to send it), and the API running: `uv run uvicorn building_with_rag.main:app --port 8000`. Select **Building with RAG**, **semantic**. Wait at least 20 seconds between questions (Voyage).

| # | Do this in Open WebUI | Expected |
|---|---|---|
| 1 | Ask: "What is the punishment for murder under the BNS?" | Text appears progressively, beginning `DRAFT — checking evidence`; ends with `Checks passed. Confidence: high.`, 1 or more BNS sources with act-qualified sections, no error. Outcome `answered`, confidence high, 0 or 1 retries. |
| 2 | Ask: "What is the capital gains tax rate for a salaried employee?" | No draft; an insufficient-evidence message, no sources, no confidence. Outcome `insufficient_evidence`. |
| 3 | Ask a question likely to produce a poor answer, for example a vague one such as "Is this punishable?", or one mixing acts, until a retry or low-confidence result appears. | Either a `Check failed … Retrying once` line then a validated answer, or `DRAFT — low confidence, not the final answer` with a short reason and the draft's sources marked unverified; no answer is presented as validated. If every try is answered cleanly, say so; if the check never fails, the deliberately unsupported case is covered by #2 and the offline tests. |
| 4 | Blank `GENERATION_API_KEY` in `.env`, restart the API, ask the question from #1. | One clear unavailable message, different from #2, with no key or URL shown. Restore the key afterwards. |
| 5 | Stream shape: watch the answer arrive. | Text arrives in pieces, never as one block after a long pause, and the reply finishes (no spinner left running). |
| 6 | Auth: with `CAPSTONE_API_KEY` set, ask #1 again. | The reply works, proving the Pipe's key is accepted. The 401 behavior for missing or wrong keys is covered by the offline tests. |
| 7 | Switch to another mode (for example `rag-hybrid`) and ask a question. | Its honest `not_implemented` text; model discovery still lists all six modes. |

## Handover

**The implementer fills in:**

- **Files created or changed.**
- **Commands actually run,** with results (offline tests and ruff only).
- **Participant checks 1 to 7:** report each as **pending participant**; the assistant ran none of them and made no curl or live HTTP/server calls.
- **Tuning notes:** the constants chosen (`MAX_ATTEMPTS`, `HIGH_CONFIDENCE`, `LOW_CONFIDENCE`, support thresholds, timeouts) and anything likely to need adjusting after live use.
- **Deviations and open issues,** including whether the Open WebUI bundle notes were available and whether the Pipe is known to send `CAPSTONE_API_KEY`.

**The participant fills in after running the checks in Open WebUI:**

- Checks 1 to 7: status (pass/fail), outcome, confidence, source count, retries used, and any wrong or unreadable display.
- **Known limitations:** a validated answer means the deterministic citation and support checks passed, not that every statement is legally correct; the lexical support check can be too strict or too lenient; a draft is visible before it is validated and cannot be retracted; unrelated questions still retrieve nearest passages; the answer says nothing about current legal applicability.

Later stories add retrieval modes behind the same `run_pattern`, `QueryResult` and `GenerationResult`, and reuse this stream and display.

## Handover record (implementer)

- **Files created or changed:** `src/building_with_rag/generation/answer.py` (event-producing operation, streaming call, extractor, attempts, checks, confidence), `service.py` (`run_pattern_events`, `run_pattern`), `main.py` (chat adapter, `CAPSTONE_API_KEY` check, SSE from events), `tests/test_streaming.py`, `tests/conftest.py` (chat auth off unless a test enables it), `tests/test_generation.py` and `tests/test_smoke.py` (chat placeholder tests updated; the passage fixture text is now realistic so the support check passes), `docs/architecture.md`, `docs/manual-tests.md`. `uv` rewrote `uv.lock` to match Story 3.1's `pyproject.toml` change (httpx as a runtime dependency).
- **Commands actually run:** `uv run ruff check .` (all checks passed); `uv run pytest -q` (54 passed). No curl, uvicorn, Postman or other live call was made on purpose. Note: one early `pytest` run, before the smoke test was changed to fake retrieval, let the old chat smoke test call the real retrieval path with the local `.env` once; the tests now fake retrieval and never read `.env`.
- **Participant checks 1 to 7:** **pending participant** (not run by the assistant).
- **Tuning notes:** `MAX_ATTEMPTS=2`, `HIGH_CONFIDENCE=0.9`, `LOW_CONFIDENCE=0.2`, `SUPPORT_MIN_OVERLAP=0.5` (5-letter word stems, words of 4+ letters, small stop list), request timeout 30 s, stream read timeout 30 s, stream total 90 s. The lexical support check is the likeliest thing to need loosening if good answers get retried.
- **Deviations and open issues:** the Open WebUI bundle notes were not available, so it is not known whether the Pipe sends `CAPSTONE_API_KEY`; with the key set in `.env` and no header sent, chat returns 401 until participant check 6 settles it (blank the key to run open). Retrieval and context assembly run when the stream's first event is requested, before the response opens. Only `rag-semantic` runs real generation.
