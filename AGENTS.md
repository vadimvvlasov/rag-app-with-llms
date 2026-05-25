# AGENTS.md

Guidelines for AI agents (Kiro, Copilot, Claude, etc.) working in this repository.

## Project overview

Minimal RAG pipeline in Python with a Streamlit UI and monitoring layer. Source code lives in `src/`, the UI in `app.py`, notebooks in `notebooks/`, tests in `tests/`. The project uses `uv` for dependency management.

## Commands

Always prefix Python/tool invocations with `uv run`:

```bash
uv run streamlit run app.py            # start the Streamlit UI
uv run pytest                          # run all tests
uv run pytest tests/test_unit.py -v   # unit tests only
uv run pytest tests/test_properties.py -v  # property tests only
uv run jupyter notebook                # start notebook server
uv run python -c "..."                 # one-off Python snippets
```

Do not use bare `python`, `pytest`, `streamlit`, or `jupyter` — the project virtualenv is managed by uv.

## Architecture

Four independently swappable components connected through `Protocol` interfaces in `src/interfaces.py`:

- **DataLoader** (`src/ingest.py`) — `FaqHttpLoader`
- **SearchIndex** (`src/ingest.py`) — `MinsearchIndex` (in-memory), `SqliteIndex` (SQLite, local persistence), `ElasticsearchIndex` (requires a running ES cluster; call `index_docs()` once, then reuse)
- **LLMClient** (`src/llm.py`) — `OpenAIClient`, `OllamaClient`, `OpenRouterClient`
- **Pipeline** (`src/rag.py`) — `RAGBase` depends only on the two protocols above
- **Monitoring** (`src/monitoring.py`) — wraps `RAGBase` with metrics collection:
  - `get_answer` — orchestrates the full pipeline and returns answer + metrics dict
  - `evaluate_relevance` — LLM-as-a-judge via Ollama structured output (Pydantic), classifies answers as `RELEVANT`, `PARTLY_RELEVANT`, or `NON_RELEVANT`
  - `calculate_cost` — simulates cost using GPT-4o-mini pricing
- **UI** (`app.py`) — Streamlit app; calls `monitoring.get_answer`, stores conversation history and feedback in `st.session_state`

When adding a new backend, implement the relevant Protocol — no changes to `RAGBase`, `monitoring.py`, or `app.py` are needed.

## Code conventions

- Python 3.11+, type hints on all public methods
- Match the existing docstring style (Google-style, with Args/Returns/Raises sections)
- `src/__init__.py` re-exports all public classes — update it when adding new ones
- No inline logic in notebooks; notebooks import from `src/` only
- Default Ollama model is `granite4.1:3b` — set in `app.py` as `OLLAMA_MODEL` constant

## Data model

Every FAQ document is a plain `dict` with exactly these keys:

```python
{"question": str, "answer": str, "section": str, "course": str}
```

The field name matches the upstream API — `FaqHttpLoader` keeps it as `answer`. Downstream code always uses `answer`.

`get_answer` returns a dict with these keys:

```python
{
    "answer": str,
    "response_time": float,        # seconds
    "relevance": str,              # "RELEVANT" | "PARTLY_RELEVANT" | "NON_RELEVANT"
    "relevance_explanation": str,
    "model_used": str,
    "prompt_tokens": int,
    "completion_tokens": int,
    "total_tokens": int,
    "ollama_cost": float,          # simulated USD cost
}
```

## Tests

- `tests/test_unit.py` — pytest unit tests, all HTTP calls mocked with `unittest.mock.patch`
- `tests/test_properties.py` — hypothesis property tests, 5 properties covering parsing, search bounds, course filtering, prompt construction, and context building

When changing `FaqHttpLoader`, update the mock data in `test_unit.py` to match the current two-request pattern (courses index → per-course JSON). The mock uses `side_effect=[index_resp, course_resp, ...]`.

Property tests use the `faq_dict_strategy` and `result_dict_strategy` fixtures defined at the top of `test_properties.py`. Reuse them when adding new properties.

After any change, run `uv run pytest` and confirm all tests pass before finishing.

## Environment

Secrets and config live in `.env` (gitignored). The only required variable for OpenAI usage is `OPENAI_API_KEY`. `FAQ_DATA_URL` has a working default and rarely needs to be set.

Never commit `.env`, `*.db` files, or `__pycache__` — all are covered by `.gitignore`.

## Spec

The full requirements, design, and task list are in `.kiro/specs/rag-intro/`. Refer to `design.md` for the correctness properties that property tests must cover.
