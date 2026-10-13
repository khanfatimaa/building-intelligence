import itertools

from building_with_rag.ingestion import load
from building_with_rag.ingestion import mongodb_schema as schema


def test_short_section_is_one_chunk():
    assert load.chunk_text("Short text.") == ["Short text."]


def test_long_section_splits_with_overlap():
    text = " ".join(f"Sentence number {i}." for i in range(400))
    chunks = load.chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= load.CHUNK_CHARS for c in chunks)
    # each chunk starts inside the end of the previous one
    for before, after in itertools.pairwise(chunks):
        assert after[:40] in before
    assert chunks[0].startswith("Sentence number 0.")
    assert chunks[-1].endswith("Sentence number 399.")


def test_chunks_never_cross_sections():
    def record(n: int, text: str) -> dict:
        return {
            "section_id": f"bns:{n}", "act": "BNS_2023", "act_label": "Test Act",
            "status": "in_force", "chapter": "I", "chapter_title": "T", "section_number": n,
            "heading": f"Heading {n}", "text": text, "source_pdf": "x.pdf",
            "source_sha256": "a" * 64,
        }

    docs = load.build_documents([record(1, "alpha " * 800), record(2, "beta")])
    for chunk in docs[schema.CHUNKS]:
        word = "alpha" if chunk["section_id"] == "bns:1" else "beta"
        assert word in chunk["text"]
        assert ("beta" in chunk["text"]) == (chunk["section_id"] == "bns:2")
        assert chunk["access_level"] == "public"
    assert docs[schema.CHUNKS][-1]["embedding_input_text"].startswith(
        "Test Act, section 2. Heading 2.\n\nbeta"
    )


def test_ids_are_deterministic():
    assert schema.source_id("BNS_2023") == "source:BNS_2023"
    assert schema.chunk_id("bns:103", 0) == "bns:103#0"
    assert schema.embedding_id("bns:103#0") == "emb:bns:103#0:voyage-3.5"
    assert schema.embedding_id("bns:103#0") == schema.embedding_id("bns:103#0")


def test_vector_index_definition():
    fields = schema.vector_index_definition()["fields"]
    assert list(schema.vector_index_definition()) == ["fields"]
    vectors = [f for f in fields if f["type"] == "vector"]
    assert vectors == [
        {"type": "vector", "path": "vector", "numDimensions": 1024, "similarity": "cosine"}
    ]
    filters = [f["path"] for f in fields if f["type"] == "filter"]
    assert filters == ["act", "chapter", "status", "access_level"]


def test_stops_without_mongodb_uri(monkeypatch, capsys):
    monkeypatch.setenv("MONGODB_URI", "")
    assert load.main([]) == 1
    assert "MongoDB is unavailable" in capsys.readouterr().err
