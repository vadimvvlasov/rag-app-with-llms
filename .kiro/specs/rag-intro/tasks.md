# Tasks

## Task List

- [x] 1. Настройка проекта
  - [x] 1.1 Создать `pyproject.toml` с зависимостями (minsearch, openai, sqlitesearch, elasticsearch, requests, python-dotenv, jupyter, pytest, hypothesis) и настройкой uv
  - [x] 1.2 Создать `.env` с заглушками `OPENAI_API_KEY` и `FAQ_DATA_URL`
  - [x] 1.3 Создать `src/__init__.py` с реэкспортом всех публичных классов

- [x] 2. Реализовать `src/interfaces.py`
  - [x] 2.1 Определить Protocol `SearchIndex` с методом `search(query, num_results, boost_dict, filter_dict) -> list[dict]`
  - [x] 2.2 Определить Protocol `DataLoader` с методом `load() -> list[dict]`
  - [x] 2.3 Определить Protocol `LLMClient` с методом `complete(prompt, instructions, model) -> str`

- [x] 3. Реализовать `src/ingest.py`
  - [x] 3.1 Реализовать `FaqHttpLoader` (implements `DataLoader`) — HTTP GET по `FAQ_DATA_URL`, парсинг в плоский список FAQ-документов, исключение при ошибке
  - [x] 3.2 Реализовать `MinsearchIndex` (implements `SearchIndex`) — in-memory minsearch по полям `question`, `text`, `section`
  - [x] 3.3 Реализовать `SqliteIndex` (implements `SearchIndex`) — загрузка существующего или построение нового sqlitesearch-индекса
  - [x] 3.4 Реализовать `ElasticsearchIndex` (implements `SearchIndex`) — адаптер к Elasticsearch

- [x] 4. Реализовать `src/llm.py`
  - [x] 4.1 Реализовать `OpenAIClient` (implements `LLMClient`) — вызов OpenAI API, ключ из `OPENAI_API_KEY`
  - [x] 4.2 Реализовать `OllamaClient` (implements `LLMClient`) — вызов локального Ollama REST API
  - [x] 4.3 Реализовать `OpenRouterClient` (implements `LLMClient`) — вызов OpenRouter API

- [x] 5. Реализовать `src/rag.py`
  - [x] 5.1 Реализовать `RAGBase.__init__` с параметрами `index: SearchIndex`, `llm: LLMClient`, `model`, `prompt_template`, `num_results`, `course_filter`; загрузка `.env` через python-dotenv
  - [x] 5.2 Реализовать `search(query)`, `build_context(results)`, `build_prompt(question, context)`, `ask(prompt)`, `rag(question)`

- [x] 6. Тесты
  - [x] 6.1 Unit-тесты: `FaqHttpLoader` (мок HTTP, успех + ошибка), `MinsearchIndex`, `SqliteIndex` (новый + существующий файл), `RAGBase.ask` (мок LLMClient, успех + ошибка), импорт из `src/`
  - [x] 6.2 Property-тесты (hypothesis): Property 1 (парсинг сохраняет поля), Property 2 (search ≤ N), Property 3 (фильтр по курсу), Property 4 (промпт содержит контекст и вопрос), Property 5 (контекст непустой)

- [x] 7. Ноутбуки
  - [x] 7.1 `notebooks/01_intro.ipynb` — in-memory RAG: `FaqHttpLoader` → `MinsearchIndex` → `RAGBase(index, OpenAIClient())` → пример вопроса
  - [x] 7.2 `notebooks/02_persistent.ipynb` — persistent RAG: `FaqHttpLoader` → `SqliteIndex` → `RAGBase(index, OpenAIClient())` → пример вопроса
