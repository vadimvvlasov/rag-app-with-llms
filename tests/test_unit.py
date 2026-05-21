"""
Unit tests for the rag-intro package.

Covers:
- FaqHttpLoader: HTTP success and HTTP error
- MinsearchIndex: build from docs, search returns results
- SqliteIndex: new file (builds index), existing file (loads without re-fitting)
- RAGBase.ask: mock LLMClient success + error propagation
- Import from src/: RAGBase, FaqHttpLoader, MinsearchIndex, SqliteIndex, OpenAIClient
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {
        "question": "How to enroll?",
        "answer": "Go to the website.",
        "section": "General",
        "course": "llm-zoomcamp",
    },
    {
        "question": "What is RAG?",
        "answer": "Retrieval Augmented Generation.",
        "section": "Module 1",
        "course": "llm-zoomcamp",
    },
    {
        "question": "How to install?",
        "answer": "Use pip install.",
        "section": "Setup",
        "course": "mlops-zoomcamp",
    },
]

# Raw FAQ JSON shape returned by the upstream endpoint:
# courses index → per-course JSON with "answer" field
SAMPLE_COURSES_INDEX = [
    {"course": "test-course", "path": "/json/test-course.json"},
]
SAMPLE_COURSE_DOCS = [
    {"question": "Q1", "answer": "A1", "section": "S1", "course": "test-course"},
]


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


def test_imports_from_src():
    """Verify that all public classes are importable from the src package."""
    from src import (  # noqa: F401
        FaqHttpLoader,
        MinsearchIndex,
        OpenAIClient,
        RAGBase,
        SqliteIndex,
    )


# ---------------------------------------------------------------------------
# FaqHttpLoader tests
# ---------------------------------------------------------------------------


class TestFaqHttpLoader:
    def test_load_success_returns_flat_list(self):
        """HTTP 200: load() fetches courses index then per-course JSON, returns flat list."""
        from src import FaqHttpLoader

        index_response = MagicMock()
        index_response.ok = True
        index_response.json.return_value = SAMPLE_COURSES_INDEX

        course_response = MagicMock()
        course_response.ok = True
        course_response.json.return_value = SAMPLE_COURSE_DOCS

        with patch(
            "requests.get", side_effect=[index_response, course_response]
        ) as mock_get:
            loader = FaqHttpLoader(url="http://fake-url/json/courses.json")
            docs = loader.load()

        assert mock_get.call_count == 2
        assert mock_get.call_args_list[0][0][0] == "http://fake-url/json/courses.json"
        assert (
            mock_get.call_args_list[1][0][0] == "http://fake-url/json/test-course.json"
        )

        assert isinstance(docs, list)
        assert len(docs) == 1
        doc = docs[0]
        assert doc["question"] == "Q1"
        assert doc["answer"] == "A1"
        assert doc["section"] == "S1"
        assert doc["course"] == "test-course"
        assert isinstance(doc["id"], str) and len(doc["id"]) == 8

    def test_load_http_error_raises_runtime_error(self):
        """HTTP non-2xx: load() raises RuntimeError."""
        from src import FaqHttpLoader

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 404

        with patch("requests.get", return_value=mock_response):
            loader = FaqHttpLoader(url="http://fake-url/docs.json")
            with pytest.raises(RuntimeError):
                loader.load()

    def test_load_multiple_courses_flattened(self):
        """Multiple course entries are fetched and flattened into a single list."""
        from src import FaqHttpLoader

        courses_index = [
            {"course": "course-a", "path": "/json/course-a.json"},
            {"course": "course-b", "path": "/json/course-b.json"},
        ]
        course_a_docs = [
            {"question": "Q1", "answer": "A1", "section": "S1", "course": "course-a"},
        ]
        course_b_docs = [
            {"question": "Q2", "answer": "A2", "section": "S2", "course": "course-b"},
            {"question": "Q3", "answer": "A3", "section": "S3", "course": "course-b"},
        ]

        index_resp = MagicMock(ok=True)
        index_resp.json.return_value = courses_index
        resp_a = MagicMock(ok=True)
        resp_a.json.return_value = course_a_docs
        resp_b = MagicMock(ok=True)
        resp_b.json.return_value = course_b_docs

        with patch("requests.get", side_effect=[index_resp, resp_a, resp_b]):
            loader = FaqHttpLoader(url="http://fake-url/json/courses.json")
            docs = loader.load()

        assert len(docs) == 3
        assert {d["course"] for d in docs} == {"course-a", "course-b"}


# ---------------------------------------------------------------------------
# MinsearchIndex tests
# ---------------------------------------------------------------------------


class TestMinsearchIndex:
    def test_build_and_search_returns_results(self):
        """MinsearchIndex builds from docs and search() returns a list."""
        from src import MinsearchIndex

        index = MinsearchIndex(SAMPLE_DOCS)
        results = index.search(
            query="enroll",
            num_results=5,
            boost_dict={"question": 3, "answer": 1},
            filter_dict={},
        )
        assert isinstance(results, list)

    def test_search_respects_num_results(self):
        """search() returns at most num_results documents."""
        from src import MinsearchIndex

        index = MinsearchIndex(SAMPLE_DOCS)
        for n in (1, 2, 3):
            results = index.search(
                query="how",
                num_results=n,
                boost_dict={},
                filter_dict={},
            )
            assert len(results) <= n

    def test_search_with_course_filter(self):
        """search() with filter_dict restricts results to matching course."""
        from src import MinsearchIndex

        index = MinsearchIndex(SAMPLE_DOCS)
        results = index.search(
            query="install",
            num_results=5,
            boost_dict={},
            filter_dict={"course": "mlops-zoomcamp"},
        )
        for doc in results:
            assert doc["course"] == "mlops-zoomcamp"


# ---------------------------------------------------------------------------
# SqliteIndex tests
# ---------------------------------------------------------------------------


class TestSqliteIndex:
    def test_new_file_builds_index_and_search_works(self, tmp_path):
        """SqliteIndex creates a new DB file and search() returns results."""
        from src import SqliteIndex

        db_path = str(tmp_path / "test_index.db")
        index = SqliteIndex(docs=SAMPLE_DOCS, db_path=db_path)
        results = index.search(
            query="enroll",
            num_results=5,
            boost_dict={"question": 3, "answer": 1},
            filter_dict={},
        )
        assert isinstance(results, list)

    def test_existing_file_loads_without_refitting(self, tmp_path):
        """SqliteIndex with an existing DB loads it and search still works."""
        from src import SqliteIndex

        db_path = str(tmp_path / "test_index.db")

        # Build the index once
        index1 = SqliteIndex(docs=SAMPLE_DOCS, db_path=db_path)
        results1 = index1.search(
            query="RAG",
            num_results=5,
            boost_dict={},
            filter_dict={},
        )

        # Create a second instance pointing at the same file — should load, not re-fit
        index2 = SqliteIndex(docs=[], db_path=db_path)
        results2 = index2.search(
            query="RAG",
            num_results=5,
            boost_dict={},
            filter_dict={},
        )

        # Both instances should return the same results
        assert len(results2) == len(results1)

    def test_search_num_results_respected(self, tmp_path):
        """SqliteIndex search() returns at most num_results documents."""
        from src import SqliteIndex

        db_path = str(tmp_path / "test_index2.db")
        index = SqliteIndex(docs=SAMPLE_DOCS, db_path=db_path)
        results = index.search(
            query="how",
            num_results=1,
            boost_dict={},
            filter_dict={},
        )
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# RAGBase.ask tests
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Minimal LLMClient implementation for testing."""

    def __init__(self, response: str = "mocked answer"):
        self._response = response

    def complete(self, prompt: str, instructions: str, model: str) -> str:
        return self._response


class ErrorLLMClient:
    """LLMClient that always raises an exception."""

    def complete(self, prompt: str, instructions: str, model: str) -> str:
        raise ValueError("LLM API error")


class TestRAGBaseAsk:
    def _make_index(self):
        from src import MinsearchIndex

        return MinsearchIndex(SAMPLE_DOCS)

    def test_ask_returns_llm_response(self):
        """RAGBase.ask() returns the LLM client's response."""
        from src import RAGBase

        llm = MockLLMClient(response="This is the answer.")
        rag = RAGBase(index=self._make_index(), llm_client=llm)
        result = rag.ask("What is RAG?")
        assert result == "This is the answer."

    def test_ask_propagates_llm_error(self):
        """RAGBase.ask() propagates exceptions raised by the LLM client."""
        from src import RAGBase

        llm = ErrorLLMClient()
        rag = RAGBase(index=self._make_index(), llm_client=llm)
        with pytest.raises(ValueError, match="LLM API error"):
            rag.ask("What is RAG?")

    def test_ask_passes_prompt_to_llm(self):
        """RAGBase.ask() passes the prompt string to llm.complete()."""
        from src import RAGBase

        received_prompts = []

        class CaptureLLM:
            def complete(self, prompt, instructions, model):
                received_prompts.append(prompt)
                return "ok"

        rag = RAGBase(index=self._make_index(), llm_client=CaptureLLM())
        rag.ask("my custom prompt")
        assert received_prompts == ["my custom prompt"]
