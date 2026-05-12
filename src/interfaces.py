"""
Protocol interfaces for the RAG pipeline components.

These protocols define the contracts between the three core components:
- SearchIndex: any search backend (minsearch, sqlitesearch, Elasticsearch, …)
- DataLoader:  any data source (HTTP, local file, database, …)
- LLMClient:   any language model backend (OpenAI, Ollama, OpenRouter, …)

Swapping one implementation for another requires no changes to the rest of
the codebase as long as the new class satisfies the corresponding Protocol.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SearchIndex(Protocol):
    """Contract for search-index backends.

    Any class that implements ``search`` with the signature below is a valid
    ``SearchIndex`` and can be passed to ``RAGBase``.
    """

    def search(
        self,
        query: str,
        num_results: int,
        boost_dict: dict,
        filter_dict: dict,
    ) -> list[dict]:
        """Search the index and return the top matching documents.

        Args:
            query:       Free-text query string.
            num_results: Maximum number of documents to return.
            boost_dict:  Field-level boost weights, e.g. ``{"question": 3}``.
            filter_dict: Exact-match filters, e.g. ``{"course": "llm-zoomcamp"}``.

        Returns:
            A list of FAQ_Document dicts, ordered by relevance (most relevant
            first), with at most ``num_results`` entries.
        """
        ...


@runtime_checkable
class DataLoader(Protocol):
    """Contract for data-source backends.

    Any class that implements ``load`` with the signature below is a valid
    ``DataLoader`` and can be used to feed documents into a ``SearchIndex``.
    """

    def load(self) -> list[dict]:
        """Load and return all documents from the data source.

        Returns:
            A list of FAQ_Document dicts, each containing at minimum the
            fields ``question``, ``text``, ``section``, and ``course``.

        Raises:
            RuntimeError: If the data source is unavailable or returns an
                error response.
        """
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Contract for language-model backends.

    Any class that implements ``complete`` with the signature below is a valid
    ``LLMClient`` and can be passed to ``RAGBase``.
    """

    def complete(self, prompt: str, instructions: str, model: str) -> str:
        """Send a prompt to the language model and return its text response.

        Args:
            prompt:       The user-facing prompt (question + context).
            instructions: System-level instructions that guide the model's
                          behaviour (role, tone, constraints, …).
            model:        Model identifier, e.g. ``"gpt-4o-mini"``.

        Returns:
            The model's text response as a plain string.

        Raises:
            Exception: If the underlying API returns an error; the original
                error details are preserved and re-raised.
        """
        ...
