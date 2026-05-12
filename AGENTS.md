# AGENTS.md

Guidelines for AI agents (Kiro, Copilot, Claude, etc.) working in this repository.

## Project overview

Minimal RAG pipeline in Python. Source code lives in `src/`, notebooks in `notebooks/`, tests in `tests/`. The project uses `uv` for dependency management.

## Commands

Always prefix Python/tool invocations with `uv run`:

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_unit.py -v   # unit tests only
uv run pytest tests/test_properties.py -v  # property tests only
uv run jupyter notebook                # start notebook server
uv run python -c "..."                 # one-off Python snippets
```

Do not use bare `python`, `pytest`, or `jupyter` — the project virtualenv is managed by uv.

## Architecture

Three independently swappable components connected through `Protocol` interfaces in `src/interfaces.py`:

- **DataLoader** (`src/ingest.py`) — `FaqHttpLoader`
- **SearchIndex** (`src/ingest.py`) — `MinsearchIndex`, `SqliteIndex`, `ElasticsearchIndex`
- **LLMClient** (`src/llm.py`) — `OpenAIClient`, `OllamaClient`, `OpenRouterClient`
- **Pipeline** (`src/rag.py`) — `RAGBase` depends only on the two protocols above

When adding a new backend, implement the relevant Protocol — no changes to `RAGBase` or other components are needed.

## Code conventions

- Python 3.11+, type hints on all public methods
- Match the existing docstring style (Google-style, with Args/Returns/Raises sections)
- `src/__init__.py` re-exports all public classes — update it when adding new ones
- No inline logic in notebooks; notebooks import from `src/` only

## Data model

Every FAQ document is a plain `dict` with exactly these keys:

```python
{"question": str, "text": str, "section": str, "course": str}
```

The upstream API uses `answer` instead of `text` — `FaqHttpLoader` normalises this on load. Downstream code always uses `text`.

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
