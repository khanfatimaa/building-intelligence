"""Project-local MongoDB names, IDs, validators, ordinary indexes, and the vector index definition.

Everything that touches the four collections imports its names from here.
"""

from typing import Any

from pymongo.database import Database

SOURCES = "sources"
SECTIONS = "sections"
CHUNKS = "chunks"
EMBEDDINGS = "embeddings"
COLLECTIONS = (SOURCES, SECTIONS, CHUNKS, EMBEDDINGS)

EMBEDDING_MODEL = "voyage-3.5"
EMBEDDING_VERSION = "voyage-3.5"
EMBEDDING_DIMENSIONS = 1024
CHUNKING_VERSION = "chunk-v1"
VECTOR_INDEX_NAME = "embeddings_vector_index"
VECTOR_FIELD = "vector"
SIMILARITY = "cosine"
ACCESS_LEVEL_PUBLIC = "public"

FILTER_ACT = "act"
FILTER_CHAPTER = "chapter"
FILTER_STATUS = "status"
FILTER_ACCESS_LEVEL = "access_level"
VECTOR_FILTER_FIELDS = (FILTER_ACT, FILTER_CHAPTER, FILTER_STATUS, FILTER_ACCESS_LEVEL)

# Keyword retrieval (Story 4.1): an Atlas Search `search` index on the plain chunk text.
KEYWORD_INDEX_NAME = "chunks_text_index"
KEYWORD_FIELD = "text"
KEYWORD_ANALYZER = "lucene.standard"
KEYWORD_FILTER_FIELDS = (FILTER_ACT, FILTER_STATUS, FILTER_ACCESS_LEVEL)


def source_id(act: str) -> str:
    return f"source:{act}"


def chunk_id(section_id: str, chunk_index: int) -> str:
    return f"{section_id}#{chunk_index}"


def embedding_id(chunk_id_: str, model_version: str = EMBEDDING_VERSION) -> str:
    return f"emb:{chunk_id_}:{model_version}"


def _validator(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {"$jsonSchema": {"bsonType": "object", "required": required, "properties": properties}}


_STR = {"bsonType": "string"}
_NULL_STR = {"bsonType": ["string", "null"]}
_INT = {"bsonType": ["int", "long"]}

VALIDATORS: dict[str, dict[str, Any]] = {
    SOURCES: _validator(
        ["_id", "source_id", "act", "act_label", "status", "source_pdf", "source_sha256"],
        {
            "_id": _STR,
            "source_id": _STR,
            "act": _STR,
            "act_label": _STR,
            "status": _STR,
            "source_pdf": _STR,
            "source_sha256": _STR,
            "parser": _NULL_STR,
            "parser_version": _NULL_STR,
            "source_status_version": _NULL_STR,
        },
    ),
    SECTIONS: _validator(
        ["_id", "section_id", "source_id", "act", "status", "chapter", "section_number",
         "text", "access_level"],
        {
            "_id": _STR,
            "section_id": _STR,
            "source_id": _STR,
            "act": _STR,
            "status": _STR,
            "chapter": _STR,
            "section_number": _INT,
            "heading": _NULL_STR,
            "text": _STR,
            "access_level": _STR,
        },
    ),
    CHUNKS: _validator(
        ["_id", "chunk_id", "section_id", "chunk_index", "chunking_version", "text",
         "embedding_input_text", "act", "chapter", "status", "access_level"],
        {
            "_id": _STR,
            "chunk_id": _STR,
            "section_id": _STR,
            "chunk_index": _INT,
            "chunking_version": _STR,
            "text": _STR,
            "embedding_input_text": _STR,
            "act": _STR,
            "chapter": _STR,
            "status": _STR,
            "access_level": _STR,
        },
    ),
    EMBEDDINGS: _validator(
        ["_id", "embedding_id", "chunk_id", "embedding_model", "embedding_version",
         "embedding_dimensions", VECTOR_FIELD, "act", "chapter", "status", "access_level"],
        {
            "_id": _STR,
            "embedding_id": _STR,
            "chunk_id": _STR,
            "embedding_model": _STR,
            "embedding_version": _STR,
            "embedding_dimensions": _INT,
            VECTOR_FIELD: {"bsonType": "array", "items": {"bsonType": ["double", "int"]}},
            "act": _STR,
            "chapter": _STR,
            "status": _STR,
            "access_level": _STR,
        },
    ),
}

# (collection, keys, unique)
ORDINARY_INDEXES: list[tuple[str, list[tuple[str, int]], bool]] = [
    (SOURCES, [("source_id", 1)], True),
    (SECTIONS, [("section_id", 1)], True),
    (SECTIONS, [("act", 1), ("section_number", 1)], False),
    (CHUNKS, [("chunk_id", 1)], True),
    (CHUNKS, [("section_id", 1)], False),
    (CHUNKS, [("section_id", 1), ("chunk_index", 1)], False),
    (EMBEDDINGS, [("embedding_id", 1)], True),
    (EMBEDDINGS, [("chunk_id", 1)], False),
]


def ensure_collections(db: Database) -> None:
    """Create each collection with its validator, or update the validator if it exists."""
    existing = set(db.list_collection_names())
    for name in COLLECTIONS:
        if name in existing:
            db.command("collMod", name, validator=VALIDATORS[name])
        else:
            db.create_collection(name, validator=VALIDATORS[name])


def ensure_indexes(db: Database) -> None:
    for name, keys, unique in ORDINARY_INDEXES:
        db[name].create_index(keys, unique=unique)


def vector_index_definition() -> dict[str, Any]:
    """The Atlas `vectorSearch` index definition: a `fields` array, no legacy mappings."""
    return {
        "fields": [
            {
                "type": "vector",
                "path": VECTOR_FIELD,
                "numDimensions": EMBEDDING_DIMENSIONS,
                "similarity": SIMILARITY,
            },
            *({"type": "filter", "path": path} for path in VECTOR_FILTER_FIELDS),
        ]
    }


def keyword_index_definition() -> dict[str, Any]:
    """The Atlas `search` index definition on `chunks`: static mapping, text plus token filters."""
    fields: dict[str, Any] = {
        KEYWORD_FIELD: {"type": "string", "analyzer": KEYWORD_ANALYZER},
    }
    for path in KEYWORD_FILTER_FIELDS:
        fields[path] = {"type": "token"}
    return {"mappings": {"dynamic": False, "fields": fields}}


def keyword_index_matches(index: dict[str, Any]) -> bool:
    """True when a listed index is the search index defined above.

    Atlas may add defaults when it stores a definition, so each wanted field is checked for its
    type (and the text analyzer) instead of comparing the whole document.
    """
    have = index.get("latestDefinition") or index.get("definition") or {}
    mappings = have.get("mappings") or {}
    if index.get("type") != "search" or mappings.get("dynamic") not in (False, None):
        return False
    have_fields = mappings.get("fields") or {}
    for path, wanted in keyword_index_definition()["mappings"]["fields"].items():
        found = have_fields.get(path)
        if isinstance(found, list):
            found = found[0] if found else None
        if not isinstance(found, dict) or found.get("type") != wanted["type"]:
            return False
        if "analyzer" in wanted and found.get("analyzer", KEYWORD_ANALYZER) != wanted["analyzer"]:
            return False
    return mappings.get("dynamic") is False
