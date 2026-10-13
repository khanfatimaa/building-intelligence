import json

from building_with_rag import ingest

HASH = "a" * 64


def _pages() -> list[str]:
    header = "THE TEST CODE, 1860"
    index = f"{header}\nCHAPTER I\nPRELIMINARY\n1. Short title.\n2. Definitions.\n3. Other.\n9"
    body1 = (
        f"{header}\nCHAPTER I\nPRELIMINARY\n"
        "1. Short title.—This Act may be called the Test Code,\n"
        "and it extends to all.\n"
        "2. Definitions.—In this Code the word\n10"
    )
    body2 = (
        f"{header}\nmeans a thing.\n"
        "2A. Special rule.—Lettered text that must be left out.\n"
        "3. Other.—Whoever acts shall be punished.\n"
        "CHAPTER II\nOF THINGS\n"
        "4. Fourth.—Last section here.\n"
        "3. Subs. by Act 1 of 1870, s. 2, for the former words.\n11"
    )
    return [index, body1, body2]


def _scan(monkeypatch):
    monkeypatch.setattr(ingest, "NEW_RUN_MIN_LAST", 3)
    lines, empty = ingest.clean_pages(_pages())
    runs, omitted = ingest.scan_runs(lines)
    return lines, empty, runs, omitted


def test_clean_pages_removes_headers_and_page_numbers(monkeypatch):
    lines, empty, _, _ = _scan(monkeypatch)
    texts = [ln.text for ln in lines]
    assert not any("THE TEST CODE" in t for t in texts)
    assert "9" not in texts and "10" not in texts and "11" not in texts
    assert empty == []


def test_runs_pick_body_over_index_and_join_pages(monkeypatch):
    _, empty, runs, omitted = _scan(monkeypatch)
    assert len(runs) == 2
    records = ingest.build_records("ipc", runs, empty, HASH, "1.0", "Retrieved x", True)
    assert [r["section_number"] for r in records] == [1, 2, 3, 4]
    two = records[1]
    assert two["heading"] == "Definitions"
    assert two["text"] == "In this Code the word means a thing."
    assert (two["page_start"], two["page_end"]) == (2, 3)
    assert two["section_id"] == "ipc:2" and two["status"] == "repealed"
    assert records[0]["chapter"] == "I" and records[0]["chapter_title"] == "PRELIMINARY"
    assert records[3]["chapter"] == "II" and records[3]["chapter_title"] == "OF THINGS"
    assert records[2]["text"] == "Whoever acts shall be punished."
    assert omitted == ["2A"]
    # section 4 is followed by a footnote line, which is kept and flagged
    assert [r["needs_review"] for r in records] == [False, False, False, True]


def test_footnote_is_not_a_section(monkeypatch):
    _, _, runs, _ = _scan(monkeypatch)
    body = max(runs, key=len)
    assert len(body) == 4


def test_missing_provenance_or_hash_flags_review(monkeypatch):
    _, empty, runs, _ = _scan(monkeypatch)
    records = ingest.build_records("bns", runs, empty, HASH, "1.0", None, True)
    assert all(r["needs_review"] for r in records)
    records = ingest.build_records("bns", runs, empty, HASH, "1.0", "x", False)
    assert all(r["needs_review"] for r in records)


def test_exact_doubled_text_is_removed_only_when_exact(monkeypatch):
    assert ingest._dedupe("The same sentence repeated here. The same sentence repeated here.") == (
        "The same sentence repeated here."
    )
    other = "The same sentence repeated here. The same sentence repeated there."
    assert ingest._dedupe(other) == other


def test_read_provenance_parses_table(monkeypatch, tmp_path):
    path = tmp_path / "PROVENANCE.md"
    path.write_text(
        "| File | Source | Source URL | Retrieved (UTC) | SHA-256 |\n|---|---|---|---|---|\n"
        f"| `X.pdf` | Some body | https://example.org/x.pdf | 2026-09-21 | `{HASH}` |\n",
        encoding="utf-8",
    )
    entry = ingest.read_provenance(path)["X.pdf"]
    assert entry["sha256"] == HASH
    assert (
        ingest.status_version(entry) == "Retrieved 2026-09-21 (UTC) from https://example.org/x.pdf"
    )
    assert ingest.status_version(None) is None


def test_safe_rerun_behaviour(monkeypatch, tmp_path):
    _, empty, runs, _ = _scan(monkeypatch)
    records = ingest.build_records("ipc", runs, empty, HASH, "1.0", "x", True)
    out = tmp_path / "ipc_sections.jsonl"
    assert not ingest.is_output_current(out, HASH)
    ingest.write_atomic(out, records)
    first = out.read_bytes()
    assert ingest.is_output_current(out, HASH)
    assert not ingest.is_output_current(out, "b" * 64)
    saved = [json.loads(line) for line in first.decode("utf-8").splitlines()]
    assert all(set(ingest.FIELDS) <= set(r) for r in saved)
    assert not any(k.startswith("_") for r in saved for k in r)
    # a replacement is whole, never appended, and leaves no temp files
    ingest.write_atomic(out, records)
    assert out.read_bytes() == first
    assert [p.name for p in tmp_path.iterdir()] == ["ipc_sections.jsonl"]
    out.write_text("not json\n", encoding="utf-8")
    assert not ingest.is_output_current(out, HASH)


def test_help_does_not_touch_data(capsys):
    try:
        ingest.main(["--help"])
    except SystemExit as exit_:
        assert exit_.code == 0
    assert "Extract section records" in capsys.readouterr().out
