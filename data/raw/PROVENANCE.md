# Source data provenance

Raw legal text used by `distillation/bns_corpus.py` to build
`distillation/data/bns_sections.jsonl`. Both files were fetched from
official government sources and verified live before download (per
`distillation/instructions.md` Phase B) — no third-party aggregators or
blogs were used.

| File | Source | Source URL | Retrieved (UTC) | SHA-256 |
|---|---|---|---|---|
| `BNS_2023_bare_act.pdf` | National Crime Records Bureau (NCRB), an attached office of the Ministry of Home Affairs — the ministry that administers the BNS 2023 criminal-law overhaul | https://www.ncrb.gov.in/uploads/SankalanPortal/DownloadPDF/BNS2023.pdf | 2026-09-21 | `a83f12a93e32c9e0b39a85f75850a448dc4682b1b44b2d6bd680165bb9931549` |
| `IPC_1860_bare_act.pdf` | India Code portal, maintained by the Legislative Department, Ministry of Law and Justice | https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf | 2026-09-21 | `ef8945c5d1b02904da67959e245b87bd5751ed5563d03ab0079758909f145309` |

## Notes

- `BNS_2023_bare_act.pdf` is the Bharatiya Nyaya Sanhita, 2023 (Act No. 45
  of 2023), assented to 25 December 2023, commenced 1 July 2024. It is
  **in force** and repeals the Indian Penal Code, 1860. The PDF includes,
  in addition to the bare act text, an index and a section-correspondence
  table mapping each BNS section to its repealed IPC counterpart — useful
  context for the parser but not itself training data.
- `IPC_1860_bare_act.pdf` is the Indian Penal Code, 1860 (Act No. 45 of
  1860). It is **repealed** as of 1 July 2024 and is kept here only for
  the "savings clause" historical-comparison framing described in
  `instructions.md` — records parsed from it are tagged
  `status: "repealed"` and `act: "IPC_1860"` and must never be presented
  as current law without that tag.
- `www.indiacode.nic.in` (the primary official source named in
  `instructions.md`) intermittently returned `504 Gateway Time-out` for
  the BNS bare-act bitstream during retrieval; NCRB (an MHA body) was used
  instead as an equally authoritative government source for that one
  file. The IPC 1860 download succeeded directly from `indiacode.nic.in`
  on retry.
- Both PDFs were spot-checked after download (page count, first pages'
  extracted text) to confirm they are the expected acts before being
  treated as authoritative — see `bns_corpus.py` for the parser that
  turns them into `distillation/data/bns_sections.jsonl`.