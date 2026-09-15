from app.models import PastTopic, Teacher
from app.services.demo_topics import DEMO_TOPIC_BANK
from app.services.generator import TopicGenerator


def generator_without_client() -> TopicGenerator:
    return object.__new__(TopicGenerator)


def teacher(name: str, past: list[str] | None = None) -> Teacher:
    row = Teacher(full_name=name, research_areas=[])
    for title in past or []:
        row.past_topics.append(PastTopic(title=title))
    return row


def test_demo_bank_has_100_unique_topics():
    titles = [title for title, _ in DEMO_TOPIC_BANK]
    assert len(titles) == 100
    assert len(set(titles)) == 100


def test_demo_never_reuses_teacher_ready_topic_as_new():
    g = generator_without_client()
    ready = "Готовая утверждённая тема преподавателя"
    t = teacher("Иванов Иван Иванович", [ready])
    result = g._demo_generate(t, 2)
    assert ready not in [item.title for item in result]


def test_demo_avoids_repeats_across_selected_teachers_until_bank_exhausted():
    g = generator_without_client()
    first = g._demo_generate(teacher("Иванов Иван Иванович"), 60)
    used = [item.title for item in first]
    second = g._demo_generate(teacher("Петров Пётр Петрович"), 40, used)
    second_titles = [item.title for item in second]
    assert len(set(used)) == 60
    assert len(set(second_titles)) == 40
    assert set(used).isdisjoint(second_titles)

    # После исчерпания всех 100 локальных вариантов повтор уже разрешён.
    all_used = used + second_titles
    third = g._demo_generate(teacher("Сидоров Сидор Сидорович"), 1, all_used)
    assert third[0].title in {title for title, _ in DEMO_TOPIC_BANK}
