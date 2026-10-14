"""Create (or reuse) the Atlas Search keyword index on chunks.text. Story 4.1.

Run from the repository root with an untracked .env:
    uv run python -m building_with_rag.ingestion.keyword_index

It never changes documents or the vector index, and never deletes, drops or recreates an index.
Output is compact: it never prints the URI, keys or passage text.
"""

import argparse
import sys
import time

from pymongo.database import Database
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.ingestion.load import Stop, connect
from building_with_rag.settings import get_settings

CAPABILITY_LINE = "Atlas Search available; search index type used for keyword retrieval"
INDEX_WAIT_SECONDS = 300
INDEX_POLL_SECONDS = 10
SMOKE_QUERY = "culpable homicide"
SMOKE_ATTEMPTS = 6
SMOKE_RETRY_SECONDS = 10


def check_capability(db: Database) -> None:
    """Prove Atlas Search index management works on chunks, using the documented PyMongo API."""
    try:
        list(db[schema.CHUNKS].list_search_indexes())
    except OperationFailure as exc:
        if exc.code == 26:
            raise Stop(
                "The chunks collection does not exist. Run "
                "`uv run python -m building_with_rag.ingestion.load` first."
            ) from None
        if exc.code == 13:
            raise Stop(
                "Prerequisite missing: this database user may not manage Atlas Search indexes. "
                "Give it a role that allows it (for example Atlas admin or dbAdmin). "
                "Hybrid retrieval stays unavailable; semantic is unchanged."
            ) from None
        raise Stop(
            "Prerequisite missing: this deployment does not support Atlas Search "
            f"(not Atlas, or not a search-enabled cluster; error code {exc.code}). "
            "Hybrid retrieval stays unavailable; semantic is unchanged."
        ) from None
    print(CAPABILITY_LINE)


def ensure_keyword_index(db: Database) -> None:
    coll = db[schema.CHUNKS]
    found = list(coll.list_search_indexes(schema.KEYWORD_INDEX_NAME))
    if found:
        if not schema.keyword_index_matches(found[0]):
            raise Stop(
                f"Index {schema.KEYWORD_INDEX_NAME} exists but is not compatible "
                f"(type {found[0].get('type')}). It was not changed. Remove or fix it in Atlas "
                "yourself, then re-run."
            )
        print(f"Keyword index {schema.KEYWORD_INDEX_NAME}: reused (compatible).")
        return
    try:
        coll.create_search_index(
            SearchIndexModel(
                definition=schema.keyword_index_definition(),
                name=schema.KEYWORD_INDEX_NAME,
                type="search",
            )
        )
    except OperationFailure as exc:
        raise Stop(
            f"Could not create {schema.KEYWORD_INDEX_NAME} (error code {exc.code}). An Atlas M0 "
            "cluster allows only a few Atlas Search indexes in total; remove an unused one in "
            "Atlas if you are at the limit."
        ) from None
    print(f"Keyword index {schema.KEYWORD_INDEX_NAME}: created.")


def wait_for_index(db: Database) -> bool:
    deadline = time.monotonic() + INDEX_WAIT_SECONDS
    while True:
        found = list(db[schema.CHUNKS].list_search_indexes(schema.KEYWORD_INDEX_NAME))
        if found and found[0].get("queryable"):
            print("Keyword index is queryable.")
            return True
        if time.monotonic() >= deadline:
            print("Keyword index not ready yet; smoke query verification is pending.")
            return False
        print("Waiting for the keyword index to become queryable...")
        time.sleep(INDEX_POLL_SECONDS)


def smoke_query(db: Database) -> None:
    from building_with_rag.retrieval.hybrid import keyword_hits

    scope = {schema.FILTER_ACCESS_LEVEL: [schema.ACCESS_LEVEL_PUBLIC]}
    for attempt in range(SMOKE_ATTEMPTS):
        hits = keyword_hits(db, SMOKE_QUERY, scope, 3)
        if hits or attempt == SMOKE_ATTEMPTS - 1:
            break
        time.sleep(SMOKE_RETRY_SECONDS)  # a new index can take a moment to return documents
    top = f"top chunk_id={hits[0]['chunk_id']} score={hits[0]['score']:.4f}" if hits else "no hits"
    print(f"Smoke query {SMOKE_QUERY!r}: {len(hits)} hit(s), {top}.")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        description="Create or reuse the Atlas Search keyword index on chunks.text."
    ).parse_args(argv)
    settings = get_settings()
    try:
        client, db = connect(settings.mongodb_uri, settings.mongodb_db_name)
        try:
            check_capability(db)
            ensure_keyword_index(db)
            if wait_for_index(db):
                smoke_query(db)
        finally:
            client.close()
    except Stop as exc:
        print(exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - never print raw provider or driver text
        print(f"Keyword index setup failed ({type(exc).__name__}).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
