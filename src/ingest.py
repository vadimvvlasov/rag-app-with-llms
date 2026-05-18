"""
Ingest module: DataLoader and SearchIndex implementations for the RAG pipeline.

Classes:
    FaqHttpLoader   — fetches FAQ JSON from a URL and parses it into flat dicts
    MinsearchIndex  — in-memory TF-IDF search via minsearch
    SqliteIndex     — persistent FTS5 search via sqlitesearch
    ElasticsearchIndex — search adapter backed by an Elasticsearch cluster
"""

import os
from pathlib import Path

import minsearch
import requests
import sqlitesearch
from elasticsearch import Elasticsearch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FAQ_URL = "https://datatalks.club/faq/json/courses.json"

_TEXT_FIELDS = ["question", "text", "section"]
_KEYWORD_FIELDS = ["course"]


# ---------------------------------------------------------------------------
# FaqHttpLoader
# ---------------------------------------------------------------------------


class FaqHttpLoader:
    """Fetches FAQ data from an HTTP endpoint and parses it into flat dicts.

    Implements the ``DataLoader`` protocol.

    Args:
        url: URL to fetch FAQ JSON from.  If *None*, the value of the
             ``FAQ_DATA_URL`` environment variable is used; if that is also
             unset, a hard-coded default pointing to the LLM Zoomcamp dataset
             is used (Req 5.2).
    """

    def __init__(self, url: str | None = None) -> None:
        self.url: str = url or os.environ.get("FAQ_DATA_URL", _DEFAULT_FAQ_URL)

    def load(self) -> list[dict]:
        """Fetch FAQ JSON and return a flat list of FAQ_Document dicts.

        Expects the DataTalks.Club courses-index format::

            # courses index: [{"course": "...", "path": "/json/....json"}, ...]
            # per-course:    [{"course": "...", "question": ..., "answer": ...,
            #                  "section": ...}, ...]

        The ``answer`` field is normalised to ``text`` for pipeline consistency
        (Req 1.1, 1.2).

        Returns:
            List of dicts with keys ``question``, ``text``, ``section``, ``course``.

        Raises:
            RuntimeError: If any HTTP request returns a non-2xx status code (Req 1.5).
        """
        response = requests.get(self.url, timeout=60)
        if not response.ok:
            raise RuntimeError(f"Failed to fetch FAQ data: {response.status_code}")

        url_prefix = self.url.rsplit("/json/", 1)[0]
        docs: list[dict] = []
        for course_entry in response.json():
            course_resp = requests.get(
                f"{url_prefix}{course_entry['path']}", timeout=60
            )
            if not course_resp.ok:
                raise RuntimeError(
                    f"Failed to fetch course FAQ data: {course_resp.status_code}"
                )
            for doc in course_resp.json():
                docs.append(
                    {
                        "question": doc.get("question", ""),
                        "text": doc.get("answer", ""),
                        "section": doc.get("section", ""),
                        "course": doc.get("course", course_entry.get("course", "")),
                    }
                )
        return docs


# ---------------------------------------------------------------------------
# MinsearchIndex
# ---------------------------------------------------------------------------


class MinsearchIndex:
    """In-memory search index backed by minsearch (TF-IDF + cosine similarity).

    Implements the ``SearchIndex`` protocol.

    Args:
        docs: Flat list of FAQ_Document dicts to index immediately.
    """

    def __init__(self, docs: list[dict]) -> None:
        self._index = minsearch.Index(
            text_fields=_TEXT_FIELDS,
            keyword_fields=_KEYWORD_FIELDS,
        )
        self._index.fit(docs)

    def search(
        self,
        query: str,
        num_results: int,
        boost_dict: dict,
        filter_dict: dict,
    ) -> list[dict]:
        """Search the in-memory index (Req 2.1, 2.2, 2.3).

        Args:
            query:       Free-text query string.
            num_results: Maximum number of results to return.
            boost_dict:  Field-level boost weights, e.g. ``{"question": 3}``.
            filter_dict: Exact-match filters, e.g. ``{"course": "llm-zoomcamp"}``.

        Returns:
            List of matching FAQ_Document dicts, ordered by relevance.
        """
        return self._index.search(
            query,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            num_results=num_results,
        )


# ---------------------------------------------------------------------------
# SqliteIndex
# ---------------------------------------------------------------------------


class SqliteIndex:
    """Persistent FTS5 search index backed by sqlitesearch.

    Implements the ``SearchIndex`` protocol.

    If *db_path* already exists on disk the index is loaded from that file
    without re-ingesting any documents (Req 4.2).  Otherwise the index is
    built from *docs* and persisted to *db_path* (Req 4.1).

    Args:
        docs:    Flat list of FAQ_Document dicts.  Only used when the index
                 file does not yet exist.
        db_path: File-system path for the SQLite database.
    """

    def __init__(self, docs: list[dict], db_path: str) -> None:
        self._db_path = db_path
        self._index = sqlitesearch.TextSearchIndex(
            text_fields=_TEXT_FIELDS,
            keyword_fields=_KEYWORD_FIELDS,
            db_path=db_path,
        )
        # If the DB file already exists and is non-empty, skip ingestion.
        if not Path(db_path).exists() or self._index._is_empty():
            self._index.fit(docs)

    def search(
        self,
        query: str,
        num_results: int,
        boost_dict: dict,
        filter_dict: dict,
    ) -> list[dict]:
        """Search the persistent SQLite index (Req 2.1, 2.2, 2.3).

        Args:
            query:       Free-text query string.
            num_results: Maximum number of results to return.
            boost_dict:  Field-level boost weights.
            filter_dict: Exact-match filters.

        Returns:
            List of matching FAQ_Document dicts, ordered by relevance.
        """
        return self._index.search(
            query,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            num_results=num_results,
        )

    def add_docs(self, docs: list[dict]) -> None:
        """Add new documents to an existing persistent index.

        Unlike the constructor (which skips ingestion when the DB already
        exists), this method always appends the given documents to the index,
        regardless of its current state.

        Args:
            docs: List of FAQ_Document dicts to add.
        """
        self._index._add_docs(docs)


# ---------------------------------------------------------------------------
# ElasticsearchIndex
# ---------------------------------------------------------------------------


class ElasticsearchIndex:
    """Search adapter backed by an Elasticsearch cluster.

    Implements the ``SearchIndex`` protocol.

    Use ``index_docs()`` to create the index and populate it with documents.
    After that, ``search()`` can be used for retrieval.

    Args:
        host:       Elasticsearch host URL, e.g. ``"http://localhost:9200"``.
        index_name: Name of the Elasticsearch index to query.
    """

    def __init__(self, host: str, index_name: str) -> None:
        self._client = Elasticsearch(host)
        self._index_name = index_name

    def index_docs(self, docs: list[dict]) -> None:
        """Create the ES index (if needed) and bulk-index documents.

        Drops and recreates the index if it already exists, so this is
        a full rebuild — not an incremental update.

        Args:
            docs: List of FAQ_Document dicts to index.
        """
        from elasticsearch.helpers import bulk

        if self._client.indices.exists(index=self._index_name):
            self._client.indices.delete(index=self._index_name)

        self._client.indices.create(
            index=self._index_name,
            body={
                "mappings": {
                    "properties": {
                        "question": {"type": "text"},
                        "text": {"type": "text"},
                        "section": {"type": "text"},
                        "course": {"type": "keyword"},
                    }
                }
            },
        )

        actions = [{"_index": self._index_name, "_source": doc} for doc in docs]
        bulk(self._client, actions)
        self._client.indices.refresh(index=self._index_name)

    def search(
        self,
        query: str,
        num_results: int,
        boost_dict: dict,
        filter_dict: dict,
    ) -> list[dict]:
        """Search the Elasticsearch index (Req 2.1, 2.2, 2.3).

        Builds a ``multi_match`` query over the configured text fields with
        optional per-field boosts, and wraps it in a ``bool`` filter for any
        entries in *filter_dict*.

        Args:
            query:       Free-text query string.
            num_results: Maximum number of results to return (``size``).
            boost_dict:  Field-level boost weights, e.g. ``{"question": 3}``.
                         Applied as ``field^boost`` in the multi_match query.
            filter_dict: Exact-match filters applied as ``term`` clauses,
                         e.g. ``{"course": "llm-zoomcamp"}``.

        Returns:
            List of matching FAQ_Document dicts extracted from ``_source``.
        """
        # Build boosted field list: ["question^3", "text", "section^2", ...]
        fields = [
            f"{field}^{boost_dict[field]}" if field in boost_dict else field
            for field in _TEXT_FIELDS
        ]

        must_clause: dict = {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": "best_fields",
            }
        }

        filter_clauses = [
            {"term": {field: value}} for field, value in filter_dict.items()
        ]

        es_query: dict
        if filter_clauses:
            es_query = {
                "bool": {
                    "must": must_clause,
                    "filter": filter_clauses,
                }
            }
        else:
            es_query = must_clause

        response = self._client.search(
            index=self._index_name,
            query=es_query,
            size=num_results,
        )

        return [hit["_source"] for hit in response["hits"]["hits"]]
