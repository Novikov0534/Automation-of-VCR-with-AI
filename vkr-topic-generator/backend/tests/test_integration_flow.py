from io import BytesIO
import json

import pytest

import xlsxwriter
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.main import app, approved_topics
from app.services.generator import GeneratedCandidate, TopicGenerator
from app.models import GeneratedTopic, GenerationBatch, GenerationBatchTopic, PastTopic, Teacher


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def fake_mistral_generation(monkeypatch):
    counter = {"value": 0}

    async def fake_generate(self, teacher, count, extra_avoid=None, focus=None, style_examples=None):
        items = []
        avoid = {str(x).casefold().strip() for x in (extra_avoid or [])}
        avoid.update(str(x.title).casefold().strip() for x in teacher.past_topics)
        while len(items) < count:
            counter["value"] += 1
            title = (
                f"Разработка веб-системы управления учебными проектами {counter['value']} "
                "с аналитикой сроков, уведомлениями и контролем выполнения"
            )
            if title.casefold() in avoid:
                continue
            items.append(GeneratedCandidate(
                title=title,
                rationale="Backend/API, база данных, интерфейс и аналитика. Проверка: функциональные тесты.",
                keywords=["проекты", "аналитика"],
            ))
        return items, "mistral-test", None

    async def fake_generate_batch(self, teacher_requests, extra_avoid=None, focus=None, style_examples=None):
        result = {}
        warnings = []
        reserved = list(extra_avoid or [])
        for teacher, count in teacher_requests:
            items, _, warning = await fake_generate(self, teacher, count, reserved, focus, style_examples)
            result[teacher.id] = items
            reserved.extend(item.title for item in items)
            if warning:
                warnings.append(warning)
        return result, "mistral-batch-test", warnings

    monkeypatch.setattr(TopicGenerator, "generate", fake_generate)
    monkeypatch.setattr(TopicGenerator, "generate_batch", fake_generate_batch)



def reset_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def teacher_payload(name: str, position: str | None, department: str = "САПРиПК"):
    return {
        "full_name": name,
        "topic_count": 2,
        "department": department,
        "position": position,
        "email": None,
        "telegram": None,
        "phone": None,
        "research_areas": ["Информационные системы"],
        "orcid": None,
        "scopus_id": None,
        "past_topics": [
            {"title": f"Разработка информационной системы управления проектами для {name.split()[0]}", "year": 2025}
        ],
    }


def make_history_book() -> bytes:
    buffer = BytesIO()
    workbook = xlsxwriter.Workbook(buffer, {"in_memory": True})
    ws = workbook.add_worksheet("Темы проектов")
    ws.write_row(0, 0, ["№ проекта", "ФИО Руководителя ВКР", "Индивидуальная тема"])
    ws.write_row(1, 0, [17, "Профессоров Павел Петрович", "Разработка системы мониторинга производственных данных с аналитической панелью"])
    ws.write_row(2, 0, [18, "", "Разработка платформы управления проектами с контролем сроков и уведомлениями"])
    workbook.close()
    return buffer.getvalue()


def test_full_workflow_teacher_sort_json_generation_approval_xlsx():
    reset_db()
    positions = [
        ("Беззвания Борис Борисович", None),
        ("Учителей Устин Устинович", "преподаватель"),
        ("Доцентова Дарья Дмитриевна", "доцент"),
        ("Профессоров Павел Петрович", "профессор"),
        ("Профессоров Алексей Алексеевич", "профессор"),
    ]
    ids = {}
    for name, position in positions:
        response = client.post("/api/teachers", json=teacher_payload(name, position))
        assert response.status_code == 201, response.text
        ids[name] = response.json()["id"]

    teachers = client.get("/api/teachers")
    assert teachers.status_code == 200
    ordered = [x["full_name"] for x in teachers.json()]
    assert ordered == [
        "Профессоров Алексей Алексеевич",
        "Профессоров Павел Петрович",
        "Доцентова Дарья Дмитриевна",
        "Учителей Устин Устинович",
        "Беззвания Борис Борисович",
    ]

    backup = client.get("/api/teachers/export-json")
    assert backup.status_code == 200
    data = backup.json()
    assert data["format"] == "vkr-ai-teachers"
    imported = client.post("/api/teachers/import-json", json=data)
    assert imported.status_code == 200, imported.text
    assert imported.json()["updated"] == 5

    generated = client.post(
        "/api/generate/selected",
        json={"selections": [{"teacher_id": ids["Профессоров Павел Петрович"], "count": 2}], "focus": "Практические информационные системы"},
    )
    assert generated.status_code == 200, generated.text
    body = generated.json()
    assert body["created"] == 2
    assert body["batch_id"]
    first = body["topics"][0]
    assert "teacher_similarity_score" in first
    assert "global_similarity_score" in first

    edited = client.patch(f"/api/topics/{first['id']}", json={"title": "Разработка веб-системы мониторинга выполнения проектов с аналитикой сроков и уведомлениями"})
    assert edited.status_code == 200, edited.text
    approved = client.patch(f"/api/topics/{first['id']}/status", json={"status": "approved"})
    assert approved.status_code == 200

    exported = client.get(f"/api/export/xlsx?batch_id={body['batch_id']}")
    assert exported.status_code == 200, exported.text
    assert exported.content[:2] == b"PK"
    assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in exported.headers["content-type"]

    history = client.get("/api/generation-batches")
    assert history.status_code == 200
    assert history.json()[0]["topic_count"] == 2
    assert history.json()[0]["teacher_count"] == 1


def test_history_xlsx_preview_and_confirm():
    reset_db()
    created = client.post(
        "/api/teachers",
        json=teacher_payload("Профессоров Павел Петрович", "профессор"),
    )
    assert created.status_code == 201

    content = make_history_book()
    preview = client.post(
        "/api/history-import/preview",
        files={"file": ("ВКР_24-25.xlsx", content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert preview.status_code == 200, preview.text
    result = preview.json()
    assert result["total_detected"] == 2
    assert result["matched"] == 2
    assert result["unknown"] == 0

    confirmed = client.post(
        "/api/history-import/confirm",
        files={"file": ("ВКР_24-25.xlsx", content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"selected_sheets": json.dumps(["Темы проектов"], ensure_ascii=False), "create_missing_teachers": "false", "refresh_reference": "true"},
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["imported"] == 2

    teachers = client.get("/api/teachers").json()
    assert len(teachers[0]["past_topics"]) == 3  # одна исходная + две из XLSX
    imported_topics = [x for x in teachers[0]["past_topics"] if x["source_file"]]
    assert {x["year"] for x in imported_topics} == {2025}
    assert all(x["source_sheet"] == "Темы проектов" for x in imported_topics)

    # Редактирование профиля не должно стирать provenance тем, пришедших из XLSX.
    teacher = teachers[0]
    update_payload = {
        "full_name": teacher["full_name"],
        "topic_count": teacher["topic_count"],
        "department": teacher["department"],
        "position": teacher["position"],
        "email": "prof@example.com",
        "telegram": teacher["telegram"],
        "phone": teacher["phone"],
        "research_areas": teacher["research_areas"],
        "orcid": teacher["orcid"],
        "scopus_id": teacher["scopus_id"],
        "past_topics": [{"title": x["title"], "year": x["year"]} for x in teacher["past_topics"]],
    }
    edited = client.put(f"/api/teachers/{teacher['id']}", json=update_payload)
    assert edited.status_code == 200, edited.text
    assert len([x for x in edited.json()["past_topics"] if x["source_file"]]) == 2

    backup = client.get("/api/teachers/export-json")
    assert backup.status_code == 200
    exported_past = backup.json()["teachers"][0]["past_topics"]
    assert len([x for x in exported_past if x.get("source_file")]) == 2


def test_verified_integration_status_unconfigured_without_network_calls():
    reset_db()
    response = client.get("/api/integrations/status")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mistral"]["status"] == "unconfigured"
    assert body["google_sheets"]["status"] == "unconfigured"


def test_delete_one_batch_and_clear_history():
    reset_db()
    created = client.post(
        "/api/teachers",
        json=teacher_payload("Профессоров Павел Петрович", "профессор"),
    )
    assert created.status_code == 201
    teacher_id = created.json()["id"]

    batch_ids = []
    for focus in ["Первый набор", "Второй набор"]:
        generated = client.post(
            "/api/generate/selected",
            json={"selections": [{"teacher_id": teacher_id, "count": 2}], "focus": focus},
        )
        assert generated.status_code == 200, generated.text
        batch_ids.append(generated.json()["batch_id"])

    deleted = client.delete(f"/api/generation-batches/{batch_ids[0]}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted_batches": 1, "deleted_topics": 2}
    remaining = client.get("/api/generation-batches").json()
    assert [x["id"] for x in remaining] == [batch_ids[1]]

    cleared = client.delete("/api/generation-batches")
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["deleted_batches"] == 1
    assert cleared.json()["deleted_topics"] == 2
    assert client.get("/api/generation-batches").json() == []
    # История генераций очищается отдельно от справочника преподавателей.
    assert len(client.get("/api/teachers").json()) == 1

    # Полная очистка открывает новый цикл нумерации: следующий набор снова #1.
    generated_after_clear = client.post(
        "/api/generate/selected",
        json={"selections": [{"teacher_id": teacher_id, "count": 1}], "focus": "Новый цикл"},
    )
    assert generated_after_clear.status_code == 200, generated_after_clear.text
    assert generated_after_clear.json()["batch_id"] == 1


def test_legacy_hidden_teacher_columns_receive_defaults():
    reset_db()
    response = client.post(
        "/api/teachers",
        json=teacher_payload("Совместимость Сергей Сергеевич", "доцент"),
    )
    assert response.status_code == 201, response.text
    with TestingSession() as db:
        row = db.execute(text("SELECT courses, technologies, publications FROM teachers WHERE id=:id"), {"id": response.json()["id"]}).one()
        assert all(value is not None for value in row)


def test_explicit_credential_reset_returns_unconfigured():
    reset_db()
    saved = client.put("/api/settings", json={"mistral_api_key": None, "google_service_account_json": None})
    assert saved.status_code == 200, saved.text
    assert saved.json()["mistral_configured"] is False
    assert saved.json()["google_service_account_configured"] is False
    status = client.get("/api/integrations/status")
    assert status.status_code == 200, status.text
    assert status.json()["mistral"]["status"] == "unconfigured"
    assert status.json()["google_sheets"]["status"] == "unconfigured"


def test_regenerate_topic_always_changes_title_in_demo_mode():
    reset_db()
    created = client.post(
        "/api/teachers",
        json=teacher_payload("Перегенераторов Роман Романович", "доцент"),
    )
    assert created.status_code == 201, created.text
    teacher_id = created.json()["id"]

    generated = client.post(
        "/api/generate/selected",
        json={"selections": [{"teacher_id": teacher_id, "count": 2}], "focus": "Практические информационные системы"},
    )
    assert generated.status_code == 200, generated.text
    topic = generated.json()["topics"][0]
    original_title = topic["title"]

    regenerated = client.post(f"/api/topics/{topic['id']}/regenerate")
    assert regenerated.status_code == 200, regenerated.text
    assert regenerated.json()["title"] != original_title
    assert regenerated.json()["status"] == "draft"


def test_new_generation_does_not_reuse_past_topic_verbatim():
    reset_db()
    past_title = "Разработка системы мониторинга учебных проектов с контролем сроков и аналитикой прогресса"
    payload = teacher_payload("Самосравнений Семён Сергеевич", "профессор")
    payload["past_topics"] = [{"title": past_title, "year": 2025}]
    created = client.post("/api/teachers", json=payload)
    assert created.status_code == 201, created.text

    generated = client.post(
        "/api/generate/selected",
        json={"selections": [{"teacher_id": created.json()["id"], "count": 1}], "focus": None},
    )
    assert generated.status_code == 200, generated.text
    topic = generated.json()["topics"][0]
    assert topic["title"] != past_title
    assert topic["teacher_similarity_score"] < 100
    assert topic["global_similarity_score"] < 100


def test_manual_exact_duplicate_still_scores_100():
    reset_db()
    duplicate_title = "Разработка системы мониторинга учебных проектов с контролем сроков и аналитикой прогресса"
    payload = teacher_payload("Дубликатов Денис Дмитриевич", "доцент")
    payload["past_topics"] = [{"title": duplicate_title, "year": 2025}]
    created = client.post("/api/teachers", json=payload)
    assert created.status_code == 201, created.text

    generated = client.post(
        "/api/generate/selected",
        json={"selections": [{"teacher_id": created.json()["id"], "count": 1}], "focus": None},
    )
    assert generated.status_code == 200, generated.text
    topic = generated.json()["topics"][0]
    edited = client.patch(f"/api/topics/{topic['id']}", json={"title": duplicate_title})
    assert edited.status_code == 200, edited.text
    topic = edited.json()
    assert topic["teacher_similarity_score"] == 100
    assert topic["global_similarity_score"] == 100
    assert topic["teacher_similarity_method"] == "exact-duplicate"
    assert topic["global_similarity_method"] == "exact-duplicate"


def test_teacher_can_belong_to_both_departments():
    reset_db()
    response = client.post(
        "/api/teachers",
        json=teacher_payload("ДвеКафедры Дмитрий Дмитриевич", "доцент", "ЭВМиС, САПРиПК"),
    )
    assert response.status_code == 201, response.text
    # API хранит канонический порядок, UI показывает оба выбранных чекбокса.
    assert response.json()["department"] == "САПРиПК, ЭВМиС"


def test_topics_and_generation_result_sorted_by_position_then_name():
    reset_db()
    source = [
        ("Яковлев Ян Янович", "преподаватель"),
        ("Борисов Борис Борисович", "профессор"),
        ("Андреев Андрей Андреевич", "профессор"),
        ("Васильев Василий Васильевич", "доцент"),
    ]
    ids = {}
    for name, position in source:
        payload = teacher_payload(name, position)
        payload["past_topics"] = [{"title": f"Разработка информационной системы контроля проектов кафедры для профиля {name.split()[0]}", "year": 2025}]
        created = client.post("/api/teachers", json=payload)
        assert created.status_code == 201, created.text
        ids[name] = created.json()["id"]

    # Намеренно отправляем преподавателей не в требуемом порядке.
    generated = client.post(
        "/api/generate/selected",
        json={
            "selections": [
                {"teacher_id": ids["Яковлев Ян Янович"], "count": 1},
                {"teacher_id": ids["Васильев Василий Васильевич"], "count": 1},
                {"teacher_id": ids["Борисов Борис Борисович"], "count": 1},
                {"teacher_id": ids["Андреев Андрей Андреевич"], "count": 1},
            ],
            "focus": None,
        },
    )
    assert generated.status_code == 200, generated.text
    batch_id = generated.json()["batch_id"]
    expected = [
        "проф. Андреев А.А.",
        "проф. Борисов Б.Б.",
        "доц. Васильев В.В.",
        "преп. Яковлев Я.Я.",
    ]
    assert [item["teacher_name"] for item in generated.json()["topics"]] == expected
    listed = client.get(f"/api/topics?batch_id={batch_id}")
    assert listed.status_code == 200, listed.text
    assert [item["teacher_name"] for item in listed.json()] == expected

    approved = client.post(f"/api/topics/approve-all?batch_id={batch_id}")
    assert approved.status_code == 200, approved.text
    with TestingSession() as db:
        published = approved_topics(db, batch_id)
        assert [item.teacher.full_name for item in published] == [
            "Андреев Андрей Андреевич",
            "Борисов Борис Борисович",
            "Васильев Василий Васильевич",
            "Яковлев Ян Янович",
        ]


def test_settings_expose_mistral_defaults_and_unconfigured_status():
    reset_db()
    settings = client.get("/api/settings")
    assert settings.status_code == 200, settings.text
    assert settings.json()["mistral_configured"] is False
    assert settings.json()["mistral_chat_model"] == "ministral-8b-2512"
    assert settings.json()["mistral_embedding_model"] == "mistral-embed"
    status = client.get("/api/integrations/status")
    assert status.status_code == 200, status.text
    assert status.json()["mistral"]["status"] == "unconfigured"


def test_teacher_similarity_uses_only_historical_topics_not_current_batch_siblings():
    reset_db()
    created = client.post(
        "/api/teachers",
        json={
            **teacher_payload("Исторический Иван Иванович", "доцент"),
            "past_topics": [{"title": "Разработка мобильного приложения для изучения математики", "year": 2025}],
        },
    )
    assert created.status_code == 201, created.text
    teacher_id = created.json()["id"]

    with TestingSession() as db:
        batch = GenerationBatch(focus=None, selections=[{"teacher_id": teacher_id, "count": 2}])
        db.add(batch)
        db.flush()
        first = GeneratedTopic(
            teacher_id=teacher_id,
            title="Разработка системы анализа изображений опухолей",
            rationale="тест", keywords=[], generation_index=0,
        )
        sibling = GeneratedTopic(
            teacher_id=teacher_id,
            title="Разработка системы анализа изображений опухолей методом нейронных сетей",
            rationale="тест", keywords=[], generation_index=1,
        )
        db.add_all([first, sibling])
        db.flush()
        db.add_all([
            GenerationBatchTopic(batch_id=batch.id, topic_id=first.id),
            GenerationBatchTopic(batch_id=batch.id, topic_id=sibling.id),
        ])
        db.commit()
        batch_id = batch.id
        first_id = first.id

    recalculated = client.post(f"/api/generation-batches/{batch_id}/recalculate-similarity")
    assert recalculated.status_code == 200, recalculated.text
    row = next(item for item in recalculated.json() if item["id"] == first_id)
    assert row["teacher_closest_topic"] == "Разработка мобильного приложения для изучения математики"
    assert row["teacher_closest_topic"] != "Разработка системы анализа изображений опухолей методом нейронных сетей"
    # Глобальная колонка, наоборот, имеет право сравниваться с соседней темой текущего набора.
    assert row["global_closest_topic"] == "Разработка системы анализа изображений опухолей методом нейронных сетей"


def test_generation_groups_mistral_keep_batches_compact():
    from app.main import _generation_groups
    from app.schemas import GenerationSelection

    def groups(n, count=5):
        selections = [GenerationSelection(teacher_id=i + 1, count=count) for i in range(n)]
        return _generation_groups(selections)

    # Free-tier friendly: не более 3 преподавателей и 20 тем в пакете.
    assert [len(group) for group in groups(5)] == [3, 2]
    assert [len(group) for group in groups(7)] == [3, 3, 1]
    assert [sum(x.count for x in group) for group in groups(20)] == [15, 15, 15, 15, 15, 15, 10]

    # 10 преподавателей × 10 тем = 100 тем -> ровно 5 пакетов по 20.
    hundred = groups(10, count=10)
    assert len(hundred) == 5
    assert [sum(x.count for x in group) for group in hundred] == [20, 20, 20, 20, 20]

    # Даже один большой запрос режется на безопасные части.
    large = _generation_groups([GenerationSelection(teacher_id=1, count=50)])
    assert [group[0].count for group in large] == [20, 20, 10]


def test_explicit_demo_generation_marks_source_and_does_not_masquerade_as_ai():
    reset_db()
    created = client.post(
        "/api/teachers",
        json=teacher_payload("Демонстрационный Дмитрий Дмитриевич", "доцент"),
    )
    assert created.status_code == 201, created.text
    teacher_id = created.json()["id"]

    response = client.post(
        "/api/generate/selected/demo",
        json={"selections": [{"teacher_id": teacher_id, "count": 3}], "focus": "веб-системы"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "local-demo"
    assert body["created"] == 3
    assert all(item["generation_source"] == "local-demo" for item in body["topics"])
    assert all(item["generation_model"] is None for item in body["topics"])
    assert "без обращения к Mistral" in body["warning"]


def test_generation_groups_fast_ai_one_teacher_per_request():
    from app.main import _generation_groups
    from app.schemas import GenerationSelection

    selections = [GenerationSelection(teacher_id=i + 1, count=10) for i in range(10)]
    groups = _generation_groups(selections, max_size=1, max_topics=10)
    assert len(groups) == 10
    assert all(len(group) == 1 for group in groups)
    assert [group[0].count for group in groups] == [10] * 10
