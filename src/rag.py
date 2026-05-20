"""
RAG pipeline implementation.

``RAGBase`` orchestrates the full retrieval-augmented generation flow:
    search → build_context → build_prompt → ask

It depends only on the ``SearchIndex`` and ``LLMClient`` protocols defined in
``interfaces.py``, so any compatible backend can be swapped in without
changing this module (Req 3.5).
"""

from __future__ import annotations

from dotenv import load_dotenv

from .interfaces import LLMClient, SearchIndex

# ---------------------------------------------------------------------------
# Default instructions and prompt template
# ---------------------------------------------------------------------------

INSTRUCTIONS = """\
Your task is to answer questions from the course participants \
based on the provided context.
Use the context to find relevant information and provide accurate \
answers. If the answer is not found in the context, \
respond with "I don't know."\
"""

USER_PROMPT_TEMPLATE = """\
Question:
{question}

Context:
{context}\
"""


# ---------------------------------------------------------------------------
# RAGBase
# ---------------------------------------------------------------------------


class RAGBase:
    """Retrieval-Augmented Generation pipeline.

    Combines a search index and an LLM client to answer questions grounded in
    FAQ content.

    Args:
        index:            A ``SearchIndex``-compatible object used to retrieve
                          relevant documents.
        llm:              An ``LLMClient``-compatible object used to generate
                          answers.
        model:            LLM model identifier passed to ``llm.complete``.
                          Defaults to ``"gpt-4o-mini"``.
        instructions:     System-level instructions passed to the LLM.
                          Defaults to ``INSTRUCTIONS``.
        prompt_template:  Template string with ``{question}`` and ``{context}``
                          placeholders for the user-facing prompt.
                          Defaults to ``USER_PROMPT_TEMPLATE``.
        num_results:      Number of search results to retrieve per query.
                          Defaults to ``5``.
        course_filter:    If set, restricts search results to documents whose
                          ``course`` field matches this value.  Defaults to
                          ``None`` (no filter).
    """

    def __init__(
        self,
        index: SearchIndex,
        llm_client: LLMClient,
        llm_model: str = "gpt-4o-mini",
        instructions: str = INSTRUCTIONS,
        prompt_template: str = USER_PROMPT_TEMPLATE,
        num_results: int = 5,
        course_filter: str | None = None,
    ) -> None:
        # Load .env file if present (Req 5.3)
        load_dotenv()

        self._index = index
        self._llm = llm_client
        self._model = llm_model
        self._instructions = instructions
        self._prompt_template = prompt_template
        self._num_results = num_results
        self._course_filter = course_filter

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict]:
        """Search the index for documents relevant to *query*.

        Args:
            query: Free-text question or search string.

        Returns:
            List of FAQ_Document dicts ordered by relevance (at most
            ``num_results`` entries).
        """
        filter_dict = {"course": self._course_filter} if self._course_filter else {}
        return self._index.search(
            query,
            num_results=self._num_results,
            boost_dict={"question": 3, "text": 1, "section": 0.5},
            filter_dict=filter_dict,
        )

    def build_context(self, results: list[dict]) -> str:
        """Build a context string from a list of search results.

        Each result is formatted as::

            Q: <question>
            A: <text>

        Results are joined with ``\\n---\\n`` separators (Req 3.1).

        Args:
            results: List of FAQ_Document dicts (typically from ``search``).

        Returns:
            A single formatted string containing all results.
        """
        entries = [
            f"Q: {doc.get('question', '')}\nA: {doc.get('text', '')}" for doc in results
        ]
        return "\n---\n".join(entries)

    def build_prompt(self, question: str, context: str) -> str:
        """Format the user prompt template with *question* and *context* (Req 3.2).

        Args:
            question: The user's question.
            context:  The context string built from search results.

        Returns:
            The fully-formatted user prompt string ready to send to the LLM.
        """
        return self._prompt_template.format(question=question, context=context)

    def ask(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the text response (Req 3.3).

        Args:
            prompt: The fully-formatted user prompt string.

        Returns:
            The LLM's text response.

        Raises:
            Exception: If the LLM backend returns an error (Req 3.4).
        """
        return self._llm.complete(
            prompt,
            instructions=self._instructions,
            model=self._model,
        )

    def rag(self, question: str) -> str:
        """Run the full RAG pipeline for *question*.

        Orchestrates: search → build_context → build_prompt → ask.

        Args:
            question: The user's natural-language question.

        Returns:
            The LLM's answer grounded in the retrieved FAQ context.
        """
        results = self.search(question)
        context = self.build_context(results)
        prompt = self.build_prompt(question, context)
        return self.ask(prompt)
