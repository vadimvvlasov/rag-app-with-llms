"""
Property-based tests for the rag-intro package using hypothesis.

Each test is annotated with:
    # Feature: rag-intro, Property N: <property_text>
"""

from unittest.mock import MagicMock, patch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Reusable strategies
# ---------------------------------------------------------------------------

faq_dict_strategy = st.fixed_dictionaries(
    {
        "question": st.text(min_size=1),
        "text": st.text(min_size=1),
        "section": st.text(min_size=1),
        "course": st.text(
            min_size=1,
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters="-",
            ),
        ),
    }
)

result_dict_strategy = st.fixed_dictionaries(
    {
        "question": st.text(min_size=1),
        "text": st.text(min_size=1),
        "section": st.text(min_size=1),
        "course": st.text(min_size=1),
    }
)


# ---------------------------------------------------------------------------
# Property 1: Парсинг FAQ сохраняет обязательные поля
# ---------------------------------------------------------------------------

# Feature: rag-intro, Property 1: Парсинг FAQ сохраняет обязательные поля


@given(st.lists(faq_dict_strategy, min_size=1))
@settings(max_examples=100, deadline=None)
def test_property1_faq_parsing_preserves_required_fields(faq_dicts):
    """Validates: Requirements 1.2

    For any list of FAQ dicts with fields question/text/section/course,
    after parsing through FaqHttpLoader.load() (mock HTTP), each document
    must contain all four fields with non-empty string values.
    """
    from src.ingest import FaqHttpLoader

    # Build the upstream two-request format:
    # 1) courses index: [{"course": "...", "path": "/json/....json"}, ...]
    # 2) per-course docs: [{"question": ..., "answer": ..., "section": ..., "course": ...}, ...]
    # Group docs by course so each course gets one index entry + one per-course response.
    courses: dict[str, list[dict]] = {}
    for doc in faq_dicts:
        courses.setdefault(doc["course"], []).append(
            {
                "question": doc["question"],
                "answer": doc[
                    "text"
                ],  # upstream uses "answer", loader normalises to "text"
                "section": doc["section"],
                "course": doc["course"],
            }
        )

    courses_index = [
        {"course": course, "path": f"/json/{course}.json"} for course in courses
    ]

    index_resp = MagicMock(ok=True)
    index_resp.json.return_value = courses_index

    course_resps = [MagicMock(ok=True) for _ in courses]
    for resp, course in zip(course_resps, courses):
        resp.json.return_value = courses[course]

    with patch("src.ingest.requests.get", side_effect=[index_resp, *course_resps]):
        loader = FaqHttpLoader(url="http://fake-url/json/courses.json")
        docs = loader.load()

    assert len(docs) == len(faq_dicts)
    for doc in docs:
        assert (
            "question" in doc and isinstance(doc["question"], str) and doc["question"]
        )
        assert "text" in doc and isinstance(doc["text"], str) and doc["text"]
        assert "section" in doc and isinstance(doc["section"], str) and doc["section"]
        assert "course" in doc and isinstance(doc["course"], str) and doc["course"]


# ---------------------------------------------------------------------------
# Property 2: Поиск возвращает не более N результатов
# ---------------------------------------------------------------------------

# Feature: rag-intro, Property 2: Поиск возвращает не более N результатов


@given(st.lists(faq_dict_strategy, min_size=1), st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_property2_search_returns_at_most_n_results(docs, num_results):
    """Validates: Requirements 2.1

    For any MinsearchIndex with docs and any num_results N >= 1,
    search() returns a list of length <= N.
    """
    from src.ingest import MinsearchIndex

    index = MinsearchIndex(docs)
    results = index.search(
        query="test query",
        num_results=num_results,
        boost_dict={"question": 3, "text": 1, "section": 0.5},
        filter_dict={},
    )
    assert isinstance(results, list)
    assert len(results) <= num_results


# ---------------------------------------------------------------------------
# Property 3: Фильтр по курсу возвращает только совпадающие документы
# ---------------------------------------------------------------------------

# Feature: rag-intro, Property 3: Фильтр по курсу возвращает только совпадающие документы


@st.composite
def docs_with_target_course(draw):
    """Generate a list of docs where at least one doc has the target course."""
    target_course = draw(
        st.text(
            min_size=1,
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters="-",
            ),
        )
    )
    # At least one doc with the target course
    target_doc = draw(faq_dict_strategy)
    target_doc = {**target_doc, "course": target_course}

    # Additional docs (may have any course)
    other_docs = draw(st.lists(faq_dict_strategy, min_size=0, max_size=5))

    all_docs = [target_doc] + other_docs
    return all_docs, target_course


@given(docs_with_target_course())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property3_course_filter_returns_only_matching_docs(docs_and_course):
    """Validates: Requirements 2.3

    For any set of FAQ docs with varying course values and any course_filter,
    all results must have course == course_filter.
    """
    from src.ingest import MinsearchIndex

    docs, course_filter = docs_and_course
    index = MinsearchIndex(docs)
    results = index.search(
        query="test",
        num_results=20,
        boost_dict={"question": 3, "text": 1, "section": 0.5},
        filter_dict={"course": course_filter},
    )
    for doc in results:
        assert doc["course"] == course_filter


# ---------------------------------------------------------------------------
# Property 4: Промпт содержит контекст и вопрос
# ---------------------------------------------------------------------------

# Feature: rag-intro, Property 4: Промпт содержит контекст и вопрос


class _DummyIndex:
    """Minimal SearchIndex stub — never called in build_prompt tests."""

    def search(self, query, num_results, boost_dict, filter_dict):
        return []


class _DummyLLM:
    """Minimal LLMClient stub — never called in build_prompt tests."""

    def complete(self, prompt, instructions, model):
        return ""


@given(st.text(), st.text())
@settings(max_examples=100)
def test_property4_prompt_contains_context_and_question(context, question):
    """Validates: Requirements 3.2

    For any context string and question string,
    build_prompt(question, context) must contain both as substrings.
    """
    from src.rag import RAGBase

    rag = RAGBase(index=_DummyIndex(), llm_client=_DummyLLM())
    prompt = rag.build_prompt(question=question, context=context)
    assert context in prompt
    assert question in prompt


# ---------------------------------------------------------------------------
# Property 5: Контекст строится из результатов поиска
# ---------------------------------------------------------------------------

# Feature: rag-intro, Property 5: Контекст строится из результатов поиска


@given(st.lists(result_dict_strategy, min_size=1))
@settings(max_examples=100)
def test_property5_context_built_from_search_results(results):
    """Validates: Requirements 3.1

    For any non-empty list of result dicts, build_context(results) must
    return a non-empty string containing text from at least one result.
    """
    from src.rag import RAGBase

    rag = RAGBase(index=_DummyIndex(), llm_client=_DummyLLM())
    context = rag.build_context(results)

    assert isinstance(context, str)
    assert len(context) > 0

    # Context must contain text from at least one result
    assert any(doc["text"] in context for doc in results)
