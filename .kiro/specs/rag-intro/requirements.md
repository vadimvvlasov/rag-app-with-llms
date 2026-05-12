# Requirements Document

## Introduction

Минимальный RAG-проект на основе модуля 01-intro курса LLM Zoomcamp. Система загружает FAQ-данные, строит поисковый индекс (minsearch или sqlitesearch), формирует контекст и отвечает на вопросы через OpenAI API. Код организован модульно в `src/`, ноутбуки в `notebooks/` используют `src/` как библиотеку.

## Glossary

- **RAG_Pipeline**: компонент, объединяющий поиск, построение контекста, формирование промпта и вызов LLM
- **Ingestor**: компонент загрузки FAQ-данных и построения поискового индекса
- **Index**: поисковый индекс (minsearch in-memory или sqlitesearch persistent)
- **FAQ_Document**: единица данных — объект с полями `question`, `text`, `section`, `course`
- **Context**: строка, собранная из результатов поиска, передаваемая в промпт
- **Prompt**: итоговый текст запроса к LLM, включающий контекст и вопрос пользователя

---

## Requirements

### Requirement 1: Загрузка и индексация данных

**User Story:** As a developer, I want to load FAQ data and build a search index, so that the RAG pipeline can retrieve relevant documents.

#### Acceptance Criteria

1. WHEN the Ingestor is invoked, THE Ingestor SHALL fetch FAQ JSON data from a configurable URL via HTTP GET.
2. WHEN FAQ data is fetched successfully, THE Ingestor SHALL parse it into a list of FAQ_Document objects.
3. WHEN the index type is set to `minsearch`, THE Ingestor SHALL build an in-memory minsearch index over the FAQ_Document list.
4. WHEN the index type is set to `sqlitesearch`, THE Ingestor SHALL build a persistent sqlitesearch index stored at a configurable file path.
5. IF the HTTP request fails, THEN THE Ingestor SHALL raise an exception with a descriptive error message.

---

### Requirement 2: Поиск по индексу

**User Story:** As a developer, I want to search the index with a query, so that I can retrieve the most relevant FAQ documents.

#### Acceptance Criteria

1. WHEN a search query is provided, THE Index SHALL return the top-N most relevant FAQ_Document objects, where N is configurable.
2. WHEN search fields are configured, THE Index SHALL restrict matching to the specified FAQ_Document fields (`question`, `text`, `section`).
3. WHEN a filter by `course` is applied, THE Index SHALL return only FAQ_Document objects matching the specified course value.

---

### Requirement 3: RAG-пайплайн

**User Story:** As a user, I want to ask a question and receive an answer grounded in FAQ content, so that I get accurate, context-aware responses.

#### Acceptance Criteria

1. WHEN a question is provided, THE RAG_Pipeline SHALL search the Index and build a Context string from the top results.
2. WHEN a Context is built, THE RAG_Pipeline SHALL construct a Prompt combining the Context and the question using a configurable prompt template.
3. WHEN a Prompt is ready, THE RAG_Pipeline SHALL call the OpenAI API and return the text response.
4. IF the OpenAI API returns an error, THEN THE RAG_Pipeline SHALL raise an exception with the API error details.
5. THE RAG_Pipeline SHALL be instantiable with either a minsearch or sqlitesearch Index without changing the pipeline interface.

---

### Requirement 4: Персистентность индекса

**User Story:** As a developer, I want the sqlitesearch index to persist between runs, so that I avoid re-ingesting data on every startup.

#### Acceptance Criteria

1. WHEN a sqlitesearch index is built and saved, THE Ingestor SHALL write the index to a file at the configured path.
2. WHEN a sqlitesearch index file already exists, THE Ingestor SHALL load the existing index without re-fetching FAQ data.
3. WHEN the index file path is not set, THE Ingestor SHALL default to building an in-memory minsearch index.

---

### Requirement 5: Конфигурация окружения

**User Story:** As a developer, I want all secrets and configurable parameters loaded from environment variables, so that the project is portable and secure.

#### Acceptance Criteria

1. THE RAG_Pipeline SHALL read the OpenAI API key exclusively from the `OPENAI_API_KEY` environment variable.
2. THE Ingestor SHALL read the FAQ data URL from the `FAQ_DATA_URL` environment variable, with a default fallback value.
3. WHERE a `.env` file is present, THE RAG_Pipeline SHALL load it automatically using python-dotenv.

---

### Requirement 6: Модульная структура проекта

**User Story:** As a developer, I want the project code organized in `src/` with notebooks importing from it, so that logic is reusable and not duplicated.

#### Acceptance Criteria

1. THE project SHALL expose `Ingestor` and `RAG_Pipeline` as importable classes from the `src/` package.
2. WHEN a notebook in `notebooks/` demonstrates the RAG pipeline, THE notebook SHALL import `Ingestor` and `RAG_Pipeline` from `src/` rather than redefining them inline.
3. THE project SHALL declare all runtime dependencies in `pyproject.toml` and use `uv` as the package manager.
