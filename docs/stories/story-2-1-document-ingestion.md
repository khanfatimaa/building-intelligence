# Story 2.1: Document Ingestion

## Purpose

Turn the supplied BNS and IPC PDFs into a small corpus of section-level records that a person can open and read. One record is one section of one act, written as one JSON object per line.

**The implementer writes the extraction script but does not run it.** Do not run the extraction command, and do not create or change `data/processed/*.jsonl`. The person running the course runs the command and does the checks in "Checks to run yourself". The implementer may run only `uv sync`/`uv add`, `uv run ruff check .`, and `uv run pytest`.

This is the only goal. MongoDB, embeddings, vector indexes, retrieval, chunking, and answer generation belong to later stories. Do not start any of them here.

This story follows Story 1.1. It reuses the existing FastAPI application, the existing `data/` folders, and the existing `.env.example`. Do not add or rename settings, and do not change the endpoints, registry, or contracts.

## Prerequisites

- Story 1.1 is complete: `uv sync` works and the project in the repository root runs. No `docs/config.yaml` exists, so use the repository root.
- Supplied inputs, which must be preserved exactly as they are:
  - `data/raw/BNS_2023_bare_act.pdf`
  - `data/raw/IPC_1860_bare_act.pdf`
  - `data/raw/PROVENANCE.md`
- If any of these is missing or unreadable, stop and report which one. Do not download a replacement and do not guess at its content.
- Read `docs/architecture.md`, Story 1.1, `PROVENANCE.md`, and the existing code under `src/building_with_rag/` before writing anything. Preserve existing work. Do not edit or move the PDFs or `PROVENANCE.md`.

## Work to do

### 1. Choose a parser

- Pick a PDF text-extraction library that works on these PDFs (for example `pypdf`, `pdfplumber`, or `pymupdf`). Choose on suitability for text-based legal PDFs; do not run extraction on the PDFs to compare. If the chosen parser later proves unsuitable, the runner reports it and the parser is swapped.
- Add it with UV (`uv add <package>`), so `pyproject.toml` and `uv.lock` are updated. Add nothing else unless it is needed.
- Record the parser name and its installed version. They go into every record as `parser` and `parser_version`.

### 2. Write the extraction

Add a small module and command inside the existing package, for example `src/building_with_rag/ingest.py`, run with `uv run python -m building_with_rag.ingest`. Keep it plain and readable. It reads the two PDFs and writes the two output files. It does not need the database, the network, or any credentials, and it must not read or require `.env`.

Notes about the inputs (from `PROVENANCE.md`): the BNS PDF also contains an index and a BNS-to-IPC correspondence table. These are not section text. Do not turn them into section records.

**Extraction rules**

- Extract each section's text as it appears. Do not summarise, rephrase, translate, correct, or reorder it.
- Remove repeated page headers, footers, and page numbers, and normalise spacing and line breaks. Make no other change to the wording. Join a section that runs across pages into one record.
- Take chapter number, chapter title, section number, and heading from the PDF's own structure. Do not fill a value from memory.
- Handle duplicate extraction (the same text appearing twice because of the PDF layout) conservatively: remove it only when it is clearly an exact repeat. Otherwise keep it and set `needs_review` to `true`.
- If a page has no usable text (scanned or image-only), or a section cannot be identified with confidence, do not repair it silently. Set `needs_review` to `true` on the affected record, or report the unparsed pages. Do not add OCR in this story.
- Section numbers with a letter suffix (for example IPC 153A or 498A) do not fit the agreed integer `section_number`. Do not force them into the integer and do not merge their text into the preceding section. End the preceding section at the lettered heading, leave the lettered section out, and list the omitted ones in the handover as a known limitation for the instructor to decide on.
- Keep BNS and IPC records fully separate. A matching section number in both acts is two different records.

### 3. Record format

Write UTF-8 JSONL, one JSON object per line, ordered by section number, to:

- `data/processed/bns_sections.jsonl`
- `data/processed/ipc_sections.jsonl`

Every record has exactly these fields so the later classroom importer can read it:

| Field | Value |
|---|---|
| `section_id` | `bns:<number>` or `ipc:<number>` |
| `act` | `BNS_2023` or `IPC_1860` |
| `act_label` | a readable name for the act, taken from the PDF or `PROVENANCE.md` |
| `status` | `in_force` for BNS, `repealed` for IPC |
| `chapter` | string, for example `I` |
| `chapter_title` | the chapter title from the PDF |
| `section_number` | integer |
| `heading` | the section heading from the PDF |
| `text` | the section text, never empty |
| `source_pdf` | the PDF file name, for example `BNS_2023_bare_act.pdf` |
| `source_sha256` | SHA-256 of that PDF, computed when the command runs |
| `parser` | the parser library name |
| `parser_version` | the installed parser version |
| `source_status_version` | the retrieval date and source as stated in `PROVENANCE.md`; `null` if it does not state one. Do not invent it. |
| `needs_review` | `true` or `false` |

Page references (for example first and last page of the section) may be added as extra fields. They do not replace any field above. If a value cannot be found in the PDF or `PROVENANCE.md`, use `null` for it and set `needs_review` to `true`. Do not make one up. `section_id`, `act`, `status`, `section_number`, and `text` must never be null.

### 4. Safe re-runs (behaviour the script must have)

- Compare each PDF's SHA-256 with the one in `PROVENANCE.md`. If they differ, report it, because the corpus would then come from a file other than the one on record.
- If an output file already exists, is valid, and every record's `source_sha256` matches the current PDF, leave it untouched. Do not rewrite it.
- If the hashes differ, or the file is missing or invalid, regenerate the whole file as one replacement. Write to a temporary file in the same folder, then replace the old file in one step. Never append, and never leave records from two PDF versions in one file.
- Running the command twice must give the same file with no duplicate records.

### 5. Update `docs/architecture.md`

Add a short "Document ingestion" section. Keep the existing text. Record:

- the parser chosen, with its version, and why;
- the output format and file locations, including the field list above;
- the command to run (written for the person who will run it);
- the known limitations the script is designed around (scanned or uncertain pages, lettered sections left out). Mark them "to be confirmed after the first run", because the implementer has not run the script.

Also explain how later chunking will keep the parent section and source information: every chunk will carry its parent `section_id`, `act`, `status`, `chapter`, `heading`, `source_pdf`, and `source_sha256`, copied from the section record. A chunk is a piece of one section's `text` and is never merged across sections or acts. The section record stays the source of truth.

## Out of scope

MongoDB, embeddings, vector or text indexes, chunking itself, retrieval, answer generation, OCR, an evaluation or corpus-validation framework, reference-difference reports, broad security tests, and synthetic permission or access-level records. Do not change `.env.example`.

## Completion checks

### Checks the implementer runs

1. `uv sync` succeeds, and `uv run ruff check .` and `uv run pytest` still pass.
2. The script imports and `uv run python -m building_with_rag.ingest --help` (or equivalent) works without touching the PDFs or `data/processed/`. If that is not possible, skip it and say so.
3. `data/raw/` is unchanged and `data/processed/` contains only `.gitkeep`.

### Checks to run yourself (after the implementer finishes)

Keep these small.

1. Run `uv run python -m building_with_rag.ingest`. It writes both files and each line is valid JSON.
2. Confirm every record has all the required fields with the exact values and types above, and no `text` is empty. A short script is enough.
3. Inspect representative records against the PDFs: for each act, at least three sections, including the first section, one in the middle, and one near the end. Include one section that spans pages, if there is one. Confirm the number, heading, chapter, and text match the PDF.
4. Run the command a second time. The files are unchanged, with no duplicate records and the same line counts.
5. Check that the PDFs and `PROVENANCE.md` are unchanged.

## Handover

**The implementer fills in:**

- **Files created or changed:** list each one, including `pyproject.toml`, `uv.lock`, and `docs/architecture.md`.
- **Commands actually run:** each command with its result. Do not list commands that were not run. State plainly that the extraction was not run.
- **Command for you to run:** the exact extraction command and where the output will appear.
- **Deviations and open issues:** anything that differs from this story.

**You fill in after running the script:**

- **Inputs:** the files read, with the SHA-256 found and whether it matched `PROVENANCE.md`.
- **Output:** the two files written, with record counts and how many have `needs_review` true.
- **Records inspected:** which sections were compared with the PDF and what was found.
- **Known limitations:** unparsed pages, omitted lettered sections, flagged records, and anything uncertain. Update `docs/architecture.md` to match.

Story 2.2 will chunk these section records, embed them, and load them into MongoDB.
