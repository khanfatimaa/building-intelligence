"""Load Story 2.1 sections into MongoDB, chunk them, embed the chunks, and create the vector index.

Run with: uv run python -m building_with_rag.ingestion.load

Needs an untracked .env with MONGODB_URI, MONGODB_DB_NAME, MONGODB_TEST_DB_NAME and VOYAGE_API_KEY.
The first full run takes about 40 minutes on a free (rate limited) Voyage key; re-runs are quick.
Credentials are never printed.
"""

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pymongo import MongoClient, ReplaceOne
from pymongo.database import Database
from pymongo.errors import OperationFailure, PyMongoError
from pymongo.operations import SearchIndexModel

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
SECTION_FILES = ("bns_sections.jsonl", "ipc_sections.jsonl")

CHUNK_CHARS = 2048  # about 512 tokens
OVERLAP_CHARS = 256
SPLITTER = {
    "name": "building_with_rag.sentence_window",
    "chunk_chars": CHUNK_CHARS,
    "overlap_chars": OVERLAP_CHARS,
}

EMBED_BATCH_SIZE = 8
EMBED_DELAY_SECONDS = 20
EMBED_MAX_ATTEMPTS = 5
EMBED_RETRY_BASE_SECONDS = 30  # doubles after each rate-limit error

INDEX_WAIT_SECONDS = 300
INDEX_POLL_SECONDS = 10
SAMPLE_RETRY_SECONDS = 60
SAMPLE_QUESTION = "What is the punishment for murder?"
SAMPLE_NUM_CANDIDATES = 50
SAMPLE_LIMIT = 3

CAPABILITY_LINE = "Atlas Search index management available; vectorSearch index type used"


class Stop(Exception):
    """A prerequisite is missing or the data needs a decision; the message is safe to print."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


# --- chunking ---------------------------------------------------------------------------------


def _break_point(text: str, start: int, end: int) -> int:
    """Pick a paragraph, sentence, or word boundary in the back half of text[start:end]."""
    floor = start + (end - start) // 2
    for marker in ("\n\n", ". ", "; ", " "):
        pos = text.rfind(marker, floor, end)
        if pos != -1:
            return pos + len(marker)
    return end


def chunk_text(
    text: str, chunk_chars: int = CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[str]:
    """Split one section's text. Short text is one chunk; longer text overlaps by ~overlap_chars."""
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while True:
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            end = _break_point(text, start, end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            return chunks
        start = max(end - overlap_chars, start + 1)


def embedding_input(record: dict[str, Any], chunk: str) -> str:
    header = f"{record['act_label']}, section {record['section_number']}."
    heading = (record.get("heading") or "").strip().rstrip(".")
    if heading:
        header += f" {heading}."
    return f"{header}\n\n{chunk}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- building the wanted documents ------------------------------------------------------------


def read_sections() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in SECTION_FILES:
        path = PROCESSED / name
        if not path.is_file():
            raise Stop(f"{name} is missing. Run `uv run python -m building_with_rag.ingest` first.")
        try:
            with path.open(encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
        except ValueError:
            raise Stop(
                f"{name} is not valid JSONL. Run `uv run python -m building_with_rag.ingest`."
            ) from None
    return records


def build_documents(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sources: dict[str, dict[str, Any]] = {}
    sections: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec.get("chapter"), str):
            raise Stop(f"{rec['section_id']} has no chapter; the vector filter needs one.")
        sid = schema.source_id(rec["act"])
        source = {
            "_id": sid,
            "source_id": sid,
            "act": rec["act"],
            "act_label": rec["act_label"],
            "status": rec["status"],
            "source_pdf": rec["source_pdf"],
            "source_sha256": rec["source_sha256"],
            "parser": rec.get("parser"),
            "parser_version": rec.get("parser_version"),
            "source_status_version": rec.get("source_status_version"),
        }
        if sid in sources and sources[sid]["source_sha256"] != source["source_sha256"]:
            raise Stop(f"The {rec['act']} records disagree on source_sha256; the JSONL is mixed.")
        sources[sid] = source
        sections.append(
            {
                "_id": rec["section_id"],
                "section_id": rec["section_id"],
                "source_id": sid,
                "act": rec["act"],
                "act_label": rec["act_label"],
                "status": rec["status"],
                "chapter": rec["chapter"],
                "chapter_title": rec.get("chapter_title"),
                "section_number": rec["section_number"],
                "heading": rec.get("heading"),
                "text": rec["text"],
                "source_pdf": rec["source_pdf"],
                "source_sha256": rec["source_sha256"],
                "page_start": rec.get("page_start"),
                "page_end": rec.get("page_end"),
                "needs_review": rec.get("needs_review"),
                "access_level": schema.ACCESS_LEVEL_PUBLIC,
            }
        )
        for index, piece in enumerate(chunk_text(rec["text"])):
            cid = schema.chunk_id(rec["section_id"], index)
            chunks.append(
                {
                    "_id": cid,
                    "chunk_id": cid,
                    "section_id": rec["section_id"],
                    "chunk_index": index,
                    "chunking_version": schema.CHUNKING_VERSION,
                    "splitter": SPLITTER,
                    "text": piece,
                    "embedding_input_text": embedding_input(rec, piece),
                    "text_sha256": sha256_text(piece),
                    "act": rec["act"],
                    "chapter": rec["chapter"],
                    "status": rec["status"],
                    "access_level": schema.ACCESS_LEVEL_PUBLIC,
                }
            )
    return {
        schema.SOURCES: list(sources.values()),
        schema.SECTIONS: sections,
        schema.CHUNKS: chunks,
    }


def wanted_embedding(chunk: dict[str, Any]) -> dict[str, Any]:
    """The embedding document minus its vector, which is how a stored one is compared."""
    eid = schema.embedding_id(chunk["chunk_id"])
    return {
        "_id": eid,
        "embedding_id": eid,
        "chunk_id": chunk["chunk_id"],
        "embedding_model": schema.EMBEDDING_MODEL,
        "embedding_version": schema.EMBEDDING_VERSION,
        "embedding_dimensions": schema.EMBEDDING_DIMENSIONS,
        "input_sha256": sha256_text(chunk["embedding_input_text"]),
        schema.FILTER_ACT: chunk["act"],
        schema.FILTER_CHAPTER: chunk["chapter"],
        schema.FILTER_STATUS: chunk["status"],
        schema.FILTER_ACCESS_LEVEL: chunk["access_level"],
    }


# --- comparing with what is stored ------------------------------------------------------------


class Plan:
    """new / reused / changed / stale ids for one collection."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.new: list[str] = []
        self.reused: list[str] = []
        self.changed: list[str] = []
        self.stale: list[str] = []

    def summary(self) -> str:
        return (
            f"{self.name}: new {len(self.new)}, reused {len(self.reused)}, "
            f"changed {len(self.changed)}, stale {len(self.stale)}"
        )

    @property
    def needs_decision(self) -> bool:
        return bool(self.changed or self.stale)


def make_plan(
    name: str, wanted: list[dict[str, Any]], stored: dict[str, dict[str, Any]], ignore: set[str]
) -> Plan:
    plan = Plan(name)
    wanted_ids = set()
    for doc in wanted:
        wanted_ids.add(doc["_id"])
        old = stored.get(doc["_id"])
        if old is None:
            plan.new.append(doc["_id"])
        elif {k: v for k, v in old.items() if k not in ignore} == doc:
            plan.reused.append(doc["_id"])
        else:
            plan.changed.append(doc["_id"])
    plan.stale = sorted(set(stored) - wanted_ids)
    return plan


def fetch(db: Database, name: str, projection: dict[str, int] | None = None) -> dict[str, Any]:
    return {doc["_id"]: doc for doc in db[name].find({}, projection)}


def apply_plan(
    db: Database, plan: Plan, wanted: list[dict[str, Any]], extra: dict[str, Any] | None = None
) -> None:
    todo = set(plan.new) | set(plan.changed)
    ops = [
        ReplaceOne({"_id": doc["_id"]}, {**doc, **(extra or {})}, upsert=True)
        for doc in wanted
        if doc["_id"] in todo
    ]
    if ops:
        db[plan.name].bulk_write(ops)
    if plan.stale:
        db[plan.name].delete_many({"_id": {"$in": plan.stale}})


# --- deployment checks ------------------------------------------------------------------------


def connect(uri: str, db_name: str) -> tuple[MongoClient, Database]:
    if not uri:
        raise Stop("MongoDB is unavailable: set MONGODB_URI in .env and allow your IP in Atlas.")
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=8000)
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        client.close()
        raise Stop(
            "MongoDB is unavailable: check MONGODB_URI in .env and that your IP is allowed in "
            f"Atlas ({type(exc).__name__})."
        ) from None
    return client, client[db_name]


def check_capability(db: Database) -> None:
    """Prove Atlas search-index management works for this user, using the documented API."""
    try:
        list(db[schema.EMBEDDINGS].list_search_indexes())
    except OperationFailure as exc:
        if exc.code == 26:  # collection does not exist yet; the command itself is supported
            print(CAPABILITY_LINE)
            return
        if exc.code == 13:
            raise Stop(
                "Prerequisite missing: this database user may not manage Atlas Search indexes. "
                "Give it a role that allows it (for example Atlas admin or dbAdmin)."
            ) from None
        raise Stop(
            "Prerequisite missing: this deployment does not support Atlas Search index "
            f"management (not Atlas, or not a search-enabled cluster; error code {exc.code}). "
            "Use an Atlas free (M0) cluster."
        ) from None
    print(CAPABILITY_LINE)


# --- vector index -----------------------------------------------------------------------------


def _fields(definition: dict[str, Any]) -> str:
    return json.dumps(definition.get("fields"), sort_keys=True)


def ensure_vector_index(db: Database) -> None:
    """Create the vectorSearch index, reuse a matching one, or stop on an incompatible one."""
    coll = db[schema.EMBEDDINGS]
    wanted = schema.vector_index_definition()
    found = list(coll.list_search_indexes(schema.VECTOR_INDEX_NAME))
    if found:
        index = found[0]
        have = index.get("latestDefinition") or index.get("definition") or {}
        if index.get("type") != "vectorSearch" or _fields(have) != _fields(wanted):
            raise Stop(
                f"Index {schema.VECTOR_INDEX_NAME} exists but is not compatible "
                f"(type {index.get('type')}, fields {_fields(have)}; expected vectorSearch, "
                f"fields {_fields(wanted)}). It was not changed. Remove or fix it in Atlas "
                "yourself, then re-run."
            )
        print(f"Vector index {schema.VECTOR_INDEX_NAME}: reused (compatible).")
        return
    coll.create_search_index(
        SearchIndexModel(definition=wanted, name=schema.VECTOR_INDEX_NAME, type="vectorSearch")
    )
    print(f"Vector index {schema.VECTOR_INDEX_NAME}: created.")


def wait_for_index(db: Database) -> bool:
    deadline = time.monotonic() + INDEX_WAIT_SECONDS
    while True:
        found = list(db[schema.EMBEDDINGS].list_search_indexes(schema.VECTOR_INDEX_NAME))
        if found and found[0].get("queryable"):
            print("Vector index is queryable.")
            return True
        if time.monotonic() >= deadline:
            print("Vector index not ready yet; sample query verification is pending.")
            return False
        print("Waiting for the vector index to become queryable...")
        time.sleep(INDEX_POLL_SECONDS)


# --- embeddings -------------------------------------------------------------------------------


def make_voyage_client(api_key: str) -> Any:
    import voyageai

    return voyageai.Client(api_key=api_key, max_retries=0, timeout=60)


def embed_texts(client: Any, texts: list[str], input_type: str) -> list[list[float]]:
    """One Voyage request, retried with a growing wait on rate-limit errors."""
    from voyageai.error import RateLimitError, VoyageError

    wait = EMBED_RETRY_BASE_SECONDS
    for attempt in range(1, EMBED_MAX_ATTEMPTS + 1):
        try:
            result = client.embed(
                texts,
                model=schema.EMBEDDING_MODEL,
                input_type=input_type,
                output_dimension=schema.EMBEDDING_DIMENSIONS,
            )
            break
        except RateLimitError:
            if attempt == EMBED_MAX_ATTEMPTS:
                raise Stop(
                    "Voyage rate limit still hit after several retries. Stored embeddings are "
                    "kept; re-run the command to continue."
                ) from None
            print(f"  rate limited; waiting {wait}s (attempt {attempt}/{EMBED_MAX_ATTEMPTS})")
            time.sleep(wait)
            wait *= 2
        except VoyageError as exc:
            raise Stop(
                f"Voyage request failed ({type(exc).__name__}). Stored embeddings are kept; "
                "re-run the command to continue."
            ) from None
    vectors = result.embeddings
    if len(vectors) != len(texts) or any(len(v) != schema.EMBEDDING_DIMENSIONS for v in vectors):
        raise Stop(f"Voyage returned vectors that are not {schema.EMBEDDING_DIMENSIONS} numbers.")
    return vectors


def batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_missing(
    db: Database, client: Any, chunks: dict[str, dict[str, Any]], ids: list[str]
) -> None:
    groups = list(batches(ids, EMBED_BATCH_SIZE))
    done = 0
    for number, group in enumerate(groups, start=1):
        vectors = embed_texts(client, [chunks[i]["embedding_input_text"] for i in group], "document")
        ops = []
        for chunk_key, vector in zip(group, vectors, strict=True):
            doc = wanted_embedding(chunks[chunk_key])
            doc[schema.VECTOR_FIELD] = vector
            ops.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
        db[schema.EMBEDDINGS].bulk_write(ops)
        done += len(group)
        print(f"embedded {done}/{len(ids)} chunks (batch {number}/{len(groups)})")
        if number < len(groups):
            time.sleep(EMBED_DELAY_SECONDS)


# --- sample query -----------------------------------------------------------------------------


def sample_query(db: Database, client: Any) -> None:
    vector = embed_texts(client, [SAMPLE_QUESTION], "query")[0]
    pipeline = [
        {
            "$vectorSearch": {
                "index": schema.VECTOR_INDEX_NAME,
                "path": schema.VECTOR_FIELD,
                "queryVector": vector,
                "numCandidates": SAMPLE_NUM_CANDIDATES,
                "limit": SAMPLE_LIMIT,
                "filter": {schema.FILTER_ACCESS_LEVEL: schema.ACCESS_LEVEL_PUBLIC},
            }
        },
        {"$project": {"chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    deadline = time.monotonic() + SAMPLE_RETRY_SECONDS
    while True:
        try:
            hits = list(db[schema.EMBEDDINGS].aggregate(pipeline))
        except PyMongoError as exc:
            print(f"Sample query failed ({type(exc).__name__}); verification is pending.")
            return
        if hits or time.monotonic() >= deadline:
            break
        time.sleep(INDEX_POLL_SECONDS)
    if not hits:
        print("Sample query returned nothing yet; verification is pending. Try again shortly.")
        return
    top = hits[0]
    chunk = db[schema.CHUNKS].find_one({"_id": top["chunk_id"]})
    section = db[schema.SECTIONS].find_one({"_id": chunk["section_id"]}) if chunk else None
    if not chunk or not section:
        print(f"Sample query hit {top['chunk_id']} but it does not resolve to chunks/sections.")
        return
    print(f'Sample query: "{SAMPLE_QUESTION}"')
    print(f"  chunk_id:   {chunk['chunk_id']}")
    print(f"  section_id: {section['section_id']}")
    print(f"  heading:    {section.get('heading')}")
    print(f"  score:      {top['score']:.4f}")


# --- main -------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    db_name = settings.mongodb_test_db_name if args.test_db else settings.mongodb_db_name
    client, db = connect(settings.mongodb_uri, db_name)
    try:
        check_capability(db)
        wanted = build_documents(read_sections())
        wanted_chunks = {c["_id"]: c for c in wanted[schema.CHUNKS]}
        wanted_embeddings = [wanted_embedding(c) for c in wanted[schema.CHUNKS]]

        plans = {
            schema.SOURCES: make_plan(
                schema.SOURCES, wanted[schema.SOURCES], fetch(db, schema.SOURCES), {"ingested_at"}
            ),
            schema.SECTIONS: make_plan(
                schema.SECTIONS, wanted[schema.SECTIONS], fetch(db, schema.SECTIONS), set()
            ),
            schema.CHUNKS: make_plan(
                schema.CHUNKS, wanted[schema.CHUNKS], fetch(db, schema.CHUNKS), set()
            ),
            schema.EMBEDDINGS: make_plan(
                schema.EMBEDDINGS,
                wanted_embeddings,
                fetch(db, schema.EMBEDDINGS, {schema.VECTOR_FIELD: 0}),
                set(),
            ),
        }
        print(f"Database: {db.name}")
        for plan in plans.values():
            print(plan.summary())
        if any(p.needs_decision for p in plans.values()) and not args.accept_changes:
            raise Stop(
                "Stored data differs from what this run would write. Nothing was changed. "
                "Re-run with `--accept-changes` to replace them.",
                code=2,
            )

        schema.ensure_collections(db)
        schema.ensure_indexes(db)
        apply_plan(
            db, plans[schema.SOURCES], wanted[schema.SOURCES], {"ingested_at": datetime.now(UTC)}
        )
        apply_plan(db, plans[schema.SECTIONS], wanted[schema.SECTIONS])
        apply_plan(db, plans[schema.CHUNKS], wanted[schema.CHUNKS])
        # Stale or changed embeddings are removed here; changed ones are re-created below.
        emb = plans[schema.EMBEDDINGS]
        if emb.stale or emb.changed:
            db[schema.EMBEDDINGS].delete_many({"_id": {"$in": emb.stale + emb.changed}})
        print("Sources, sections and chunks are up to date.")

        ensure_vector_index(db)

        missing = set(emb.new) | set(emb.changed)
        to_embed = [e["chunk_id"] for e in wanted_embeddings if e["_id"] in missing]
        voyage = make_voyage_client(settings.voyage_api_key) if settings.voyage_api_key else None
        if to_embed and voyage is None:
            print(
                f"{len(to_embed)} chunks still need embeddings: set VOYAGE_API_KEY in .env and "
                "re-run. Embedding and the sample query are pending."
            )
            return
        if to_embed:
            minutes = round(len(to_embed) / EMBED_BATCH_SIZE * EMBED_DELAY_SECONDS / 60)
            print(f"Embedding {len(to_embed)} chunks; at least about {minutes} minutes.")
            embed_missing(db, voyage, wanted_chunks, to_embed)
        else:
            print("All embeddings already exist; no Voyage calls.")

        if voyage is None:
            print("VOYAGE_API_KEY is not set; the sample query is pending.")
            return
        if wait_for_index(db):
            sample_query(db, voyage)
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load sections, chunks and embeddings into MongoDB and create the vector index."
    )
    parser.add_argument(
        "--accept-changes",
        action="store_true",
        help="replace changed documents and delete stale chunks and embeddings",
    )
    parser.add_argument(
        "--test-db",
        action="store_true",
        help="use MONGODB_TEST_DB_NAME instead of MONGODB_DB_NAME",
    )
    args = parser.parse_args(argv)
    try:
        run(args)
    except Stop as stop:
        print(str(stop), file=sys.stderr)
        return stop.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
