"""
Monitoring utilities for the RAG pipeline.

Provides:
- evaluate_relevance: LLM-as-a-judge relevance scoring via Ollama structured output
- calculate_cost:     Cost estimation (always 0 for local Ollama)
- get_answer:         Full pipeline call that returns answer + monitoring metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .llm import OllamaClient
    from .rag import RAGBase

# ---------------------------------------------------------------------------
# Ollama client for structured output (uses OpenAI-compatible /v1 endpoint)
# ---------------------------------------------------------------------------

_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _get_ollama_openai_client() -> OpenAI:
    """Return an OpenAI client pointed at the local Ollama /v1 endpoint."""
    return OpenAI(api_key="ollama", base_url=_OLLAMA_BASE_URL)


# ---------------------------------------------------------------------------
# Relevance evaluation
# ---------------------------------------------------------------------------


class _RelevanceEvaluation(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"] = Field(
        description="Relevance classification of the generated answer."
    )
    explanation: str = Field(
        description="Brief explanation of the relevance classification."
    )


_RELEVANCE_INSTRUCTIONS = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.
Classify it as one of: NON_RELEVANT, PARTLY_RELEVANT, or RELEVANT.

- RELEVANT: the answer directly and correctly addresses the question.
- PARTLY_RELEVANT: the answer is related but incomplete or partially correct.
- NON_RELEVANT: the answer does not address the question at all.
""".strip()

_RELEVANCE_PROMPT = """
Question:
{question}

Generated Answer:
{answer}
""".strip()


def evaluate_relevance(
    question: str,
    answer: str,
    model: str = "granite4.1:3b",
) -> tuple[str, str]:
    """Evaluate the relevance of *answer* to *question* using an LLM judge.

    Uses Ollama's OpenAI-compatible endpoint with structured (Pydantic) output
    so the result is always well-formed — no JSON parsing errors.

    Args:
        question: The user's original question.
        answer:   The RAG-generated answer to evaluate.
        model:    Ollama model identifier. Defaults to ``"granite4.1:3b"``.

    Returns:
        A 2-tuple ``(relevance, explanation)`` where *relevance* is one of
        ``"RELEVANT"``, ``"PARTLY_RELEVANT"``, or ``"NON_RELEVANT"``, and
        *explanation* is a brief justification string.
    """
    client = _get_ollama_openai_client()
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _RELEVANCE_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        response_format=_RelevanceEvaluation,
        max_tokens=512,
    )
    result = response.choices[0].message.parsed
    return result.relevance, result.explanation


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


def calculate_cost(model: str, tokens: dict) -> float:
    """Estimate the monetary cost of an LLM call.

    For local Ollama models there is no real cost, but we simulate it using
    GPT-4o-mini pricing so the monitoring dashboard shows realistic numbers
    and the metric stays meaningful if the backend is swapped to OpenAI.

    Simulated pricing (GPT-4o-mini rates, per 1M tokens):
    - Input:  $0.15
    - Output: $0.60

    Args:
        model:  Model identifier (currently unused — same rate for all).
        tokens: Token-usage dict with ``prompt_tokens`` and
                ``completion_tokens`` keys.

    Returns:
        Simulated cost in USD.
    """
    prompt_cost = tokens.get("prompt_tokens", 0) * 0.15 / 1_000_000
    completion_cost = tokens.get("completion_tokens", 0) * 0.60 / 1_000_000
    return prompt_cost + completion_cost


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def get_answer(
    rag: "RAGBase",
    llm_client: "OllamaClient",
    question: str,
    course: str,
    model: str = "granite4.1:3b",
) -> dict:
    """Run the full RAG pipeline and return the answer with monitoring metrics.

    Orchestrates:
    1. ``rag.search`` — retrieve relevant FAQ documents filtered by *course*
    2. ``rag.build_context`` + ``rag.build_prompt`` — format the prompt
    3. ``llm_client.complete_with_metrics`` — call the LLM, capture tokens & time
    4. ``evaluate_relevance`` — LLM-as-a-judge relevance score
    5. ``calculate_cost`` — cost estimate (0 for Ollama)

    Args:
        rag:        Initialised ``RAGBase`` instance.
        llm_client: ``OllamaClient`` instance (must expose
                    ``complete_with_metrics``).
        question:   The user's natural-language question.
        course:     Course slug used as a search filter,
                    e.g. ``"data-engineering-zoomcamp"``.
        model:      Ollama model identifier passed to both the RAG call and
                    the relevance judge. Defaults to ``"granite4.1:3b"``.

    Returns:
        A dict with keys:
        ``answer``, ``response_time``, ``relevance``, ``relevance_explanation``,
        ``model_used``, ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``, ``ollama_cost``.
    """
    results = rag.search(question, filter_dict={"course": course})
    context = rag.build_context(results)
    prompt = rag.build_prompt(question, context)

    answer, tokens, response_time = llm_client.complete_with_metrics(
        prompt, rag._instructions, model
    )

    relevance, explanation = evaluate_relevance(question, answer, model=model)
    cost = calculate_cost(model, tokens)

    return {
        "answer": answer,
        "response_time": response_time,
        "relevance": relevance,
        "relevance_explanation": explanation,
        "model_used": model,
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "ollama_cost": cost,
    }
