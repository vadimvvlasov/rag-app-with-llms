# rag-intro

A minimal Retrieval-Augmented Generation (RAG) project based on [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) module 01-intro. FAQ documents are fetched from the DataTalks.Club API, indexed for full-text search, and used as context for answering questions via an LLM.

## How it works

```mermaid
flowchart LR
    ENV[".env"] -->|OPENAI_API_KEY\nFAQ_DATA_URL| RAGBase

    subgraph Ingest
        Loader["FaqHttpLoader\n(DataLoader)"]
        Loader -->|list[dict]| Index
        Index["MinsearchIndex\nor SqliteIndex\n(SearchIndex)"]
    end

    subgraph LLM
        OAI["OpenAIClient"]
        Ollama["OllamaClient"]
        OR["OpenRouterClient"]
    end

    User["question"] --> RAGBase
    Index --> RAGBase
    RAGBase -->|search → context → prompt| OAI & Ollama & OR
    OAI & Ollama & OR -->|answer| User

    style Ingest fill:#f0f4ff,stroke:#aac
    style LLM fill:#fff4f0,stroke:#caa
```

1. **Load** — `FaqHttpLoader` fetches FAQ documents from the DataTalks.Club courses API
2. **Index** — documents go into an in-memory (`MinsearchIndex`) or persistent SQLite (`SqliteIndex`) index
3. **Search** — `RAGBase.search()` retrieves the top-N relevant documents for a question
4. **Prompt** — retrieved docs are formatted into a context string and injected into a prompt template
5. **Answer** — the prompt is sent to an LLM (`OpenAIClient`, `OllamaClient`, or `OpenRouterClient`)

All three components — index, loader, LLM client — are swappable via Python `Protocol` interfaces defined in `src/interfaces.py`.

## Project structure

```
src/
  interfaces.py   # Protocol definitions: SearchIndex, DataLoader, LLMClient
  ingest.py       # FaqHttpLoader, MinsearchIndex, SqliteIndex, ElasticsearchIndex
  llm.py          # OpenAIClient, OllamaClient, OpenRouterClient
  rag.py          # RAGBase pipeline
notebooks/
  01_intro.ipynb       # In-memory RAG demo (MinsearchIndex + OpenAI)
  02_persistent.ipynb  # Persistent RAG demo (SqliteIndex + OpenAI)
tests/
  test_unit.py        # Unit tests (pytest)
  test_properties.py  # Property-based tests (hypothesis)
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env .env.local   # or edit .env directly
```

Set your OpenAI API key in `.env`:

```
OPENAI_API_KEY=sk-...
```

The `FAQ_DATA_URL` defaults to `https://datatalks.club/faq/json/courses.json` and does not need to be changed.

## Usage

### In a script

```python
from src import FaqHttpLoader, MinsearchIndex, RAGBase, OpenAIClient

docs = FaqHttpLoader().load()
index = MinsearchIndex(docs)
rag = RAGBase(index=index, llm=OpenAIClient(), course_filter="llm-zoomcamp")

print(rag.rag("How do I join the course?"))
```

### With a persistent index

```python
from src import FaqHttpLoader, SqliteIndex, RAGBase, OpenAIClient

docs = FaqHttpLoader().load()
index = SqliteIndex(docs=docs, db_path="faq_index.db")  # loads from disk on subsequent runs
rag = RAGBase(index=index, llm=OpenAIClient())

print(rag.rag("What tools do I need?"))
```

### With Ollama (local, no API key needed)

```python
from src import FaqHttpLoader, MinsearchIndex, RAGBase, OllamaClient

docs = FaqHttpLoader().load()
index = MinsearchIndex(docs)
rag = RAGBase(index=index, llm=OllamaClient(), model="llama3")

print(rag.rag("How do I submit homework?"))
```

### Notebooks

```bash
uv run jupyter notebook
```

Open `notebooks/01_intro.ipynb` for the in-memory demo or `notebooks/02_persistent.ipynb` for the SQLite-backed demo.

## Running tests

```bash
# Unit tests
uv run pytest tests/test_unit.py -v

# Property-based tests
uv run pytest tests/test_properties.py -v

# All tests
uv run pytest
```

## Environment variables

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes (for OpenAI) | — |
| `FAQ_DATA_URL` | no | `https://datatalks.club/faq/json/courses.json` |

## LLM backends

| Class | Backend | Key env var |
|---|---|---|
| `OpenAIClient` | OpenAI API | `OPENAI_API_KEY` |
| `OllamaClient` | Local Ollama | — |
| `OpenRouterClient` | OpenRouter API | `OPENROUTER_API_KEY` |

## Search backends

| Class | Storage | Notes |
|---|---|---|
| `MinsearchIndex` | In-memory | Fast startup, no persistence |
| `SqliteIndex` | SQLite on disk | Persists between runs, skips re-ingestion |
| `ElasticsearchIndex` | Elasticsearch cluster | Expects an existing populated index |
