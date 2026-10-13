"""Section layouts seen in the supplied PDFs, reproduced with short synthetic text."""

from building_with_rag import ingest

HASH = "a" * 64


def _join(*lines):
    return "\n".join(lines)


def _records(key, pages, monkeypatch, min_last=50):
    monkeypatch.setattr(ingest, "NEW_RUN_MIN_LAST", min_last)
    lines, empty = ingest.clean_pages(pages)
    runs, omitted = ingest.scan_runs(lines)
    return ingest.build_records(key, runs, empty, HASH, "1.0", "x", True), omitted


def test_ipc_layout_preblock_heading_and_marker_prefix(monkeypatch):
    page = _join(
        "CHAPTER I",
        "INTRODUCTION",
        "It is enacted as follows:--",
        "1.",
        "Title of the Code.",
        "1. Title of the Code.--This Act shall be",
        "called the Code.",
        "2.",
        "Punishment of offences within India.",
        "2. Punishment of offences within India.--Every person",
        "shall be liable.",
        "3.",
        "Punishment beyond, but which by law may",
        "be tried within, India.",
        "3. Punishment beyond, but which by law may",
        "be tried within, India.--Any person liable.",
        "4.",
        "Extension of Code.",
        "8*[4. Extension of Code.--The provisions apply.]",
        "5.",
        "Certain laws.",
        "7*[5. Certain laws.--Nothing shall affect.]",
        "5A.",
        "Lettered one.",
        "9*[5A. Lettered one.--Left out.]",
        "6.",
        "Definitions.",
        "6. Definitions.--Throughout this Code.",
    )
    records, omitted = _records("ipc", [page], monkeypatch)
    assert [r["section_number"] for r in records] == [1, 2, 3, 4, 5, 6]
    assert records[0]["heading"] == "Title of the Code"
    assert records[0]["text"] == "This Act shall be called the Code."
    assert records[2]["heading"] == "Punishment beyond, but which by law may be tried within, India"
    assert records[2]["text"] == "Any person liable."
    assert records[3]["text"] == "The provisions apply.]"
    assert records[4]["text"] == "Nothing shall affect.]"
    assert records[5]["text"] == "Throughout this Code."
    assert omitted == ["5A"]
    assert records[0]["chapter"] == "I" and records[0]["chapter_title"] == "INTRODUCTION"
    assert not any(r["needs_review"] for r in records)


def test_bns_layout_headings_from_index_and_no_space_after_number(monkeypatch):
    index = _join("CHAPTER I", "PRELIMINARY", "1. Short title.", "2. Definitions.", "3. Hurt.")
    body = _join(
        "CHAPTER I",
        "PRELIMINARY",
        "Short title.",
        "1. (1) This Act applies.",
        "Of hurt",
        "Definitions.",
        "2.(1) In this Sanhita,",
        "unless context requires.",
        "Hurt.",
        "3. Whoever causes pain.",
    )
    records, _ = _records("bns", [index, body], monkeypatch, min_last=3)
    assert [r["heading"] for r in records] == ["Short title", "Definitions", "Hurt"]
    assert records[0]["text"] == "(1) This Act applies."
    assert records[1]["text"] == "(1) In this Sanhita, unless context requires."
    assert records[2]["text"] == "Whoever causes pain."
