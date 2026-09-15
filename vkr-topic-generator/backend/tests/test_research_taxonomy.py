from app.models import Teacher
from app.services.generator import TopicGenerator
from app.services.research_taxonomy import RESEARCH_AREA_NAMES, direction_strengths, local_topic_relevance


def generator_without_client() -> TopicGenerator:
    return object.__new__(TopicGenerator)


def make_teacher(area_list: list[str]) -> Teacher:
    return Teacher(full_name="Тестов Тест Тестович", research_areas=area_list)


def test_taxonomy_has_20_unique_canonical_areas():
    assert len(RESEARCH_AREA_NAMES) == 20
    assert len(set(RESEARCH_AREA_NAMES)) == 20
    assert "Ассистивные технологии" in RESEARCH_AREA_NAMES
    assert "Чат-боты и диалоговые системы" in RESEARCH_AREA_NAMES


def test_assistive_direction_expands_semantically():
    strengths = direction_strengths([("Ассистивные технологии", 1.8)])
    assert strengths.get("Ассистивные технологии", 0) > 4

    accessible = local_topic_relevance(
        "Разработка мобильного приложения навигации для незрячих пользователей внутри здания по опорным точкам с голосовым сопровождением маршрута",
        "Доступность и мобильные технологии",
        research_areas=["Ассистивные технологии"],
    )
    unrelated = local_topic_relevance(
        "Разработка системы оптимального раскроя листового материала на производстве с минимизацией отходов",
        "Алгоритмические и оптимизационные задачи",
        research_areas=["Ассистивные технологии"],
    )
    assert accessible > unrelated + 5


def test_assistive_profile_promotes_expected_bank_topics():
    g = generator_without_client()
    result = g._demo_generate(make_teacher(["Ассистивные технологии"]), 8, include_ready_topics=False)
    titles = [item.title.casefold() for item in result]
    assert any("незряч" in title for title in titles)
    assert any("альтернативного текста" in title or "доступност" in title for title in titles)
    assert any("жестов" in title and "глух" in title for title in titles)


def test_computer_vision_profile_prefers_computer_vision_category():
    g = generator_without_client()
    result = g._demo_generate(make_teacher(["Компьютерное зрение", "Видеоаналитика"]), 5, include_ready_topics=False)
    assert all(item.keywords == ["Компьютерное зрение"] for item in result)


def test_chatbot_profile_prefers_nlp_or_web_topics():
    g = generator_without_client()
    result = g._demo_generate(make_teacher(["Чат-боты и диалоговые системы"]), 8, include_ready_topics=False)
    titles = [item.title.casefold() for item in result]
    assert any("службы поддержки" in title or "службу поддержки" in title for title in titles[:3])
    categories = [item.keywords[0] for item in result]
    assert sum(cat in {"Обработка естественного языка", "Веб-платформы и информационные системы", "Анализ данных и прогнозирование"} for cat in categories) >= 6


def test_multiple_research_areas_are_combined():
    g = generator_without_client()
    result = g._demo_generate(make_teacher(["Робототехника", "Компьютерное зрение"]), 12, include_ready_topics=False)
    categories = {item.keywords[0] for item in result}
    assert "Робототехника и IoT" in categories
    assert "Компьютерное зрение" in categories
