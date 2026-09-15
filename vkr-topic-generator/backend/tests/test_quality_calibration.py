import asyncio

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import GeneratedTopic, GenerationBatch, GenerationBatchTopic, PastTopic, Teacher
from app.services.calibration import generate_calibration_pairs, recommend_thresholds
from app.services.evaluation import refresh_reference_set, run_quality_evaluation


def make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'quality.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_quality_reference_set_and_evaluation(tmp_path):
    db = make_session(tmp_path)
    teacher = Teacher(full_name="Иванов Иван Иванович", position="доцент", department="САПРиПК", research_areas=["Информационные системы"])
    db.add(teacher)
    db.flush()
    for i in range(60):
        db.add(PastTopic(
            teacher_id=teacher.id,
            title=f"Разработка информационной системы мониторинга объектов {i} с веб-интерфейсом, базой данных и аналитической отчетностью",
            year=2025,
        ))
    db.commit()
    assert refresh_reference_set(db, 80) >= 50

    batch = GenerationBatch(selections=[{"teacher_id": teacher.id, "count": 1}])
    db.add(batch)
    db.flush()
    topic = GeneratedTopic(
        teacher_id=teacher.id,
        title="Разработка веб-системы контроля выполнения проектов с базой данных, уведомлениями и аналитической панелью",
        rationale="Система включает серверное API, базу данных, интерфейс пользователя, модуль аналитики и набор интеграционных тестов.",
        teacher_similarity_score=35,
        global_similarity_score=28,
    )
    db.add(topic)
    db.flush()
    db.add(GenerationBatchTopic(batch_id=batch.id, topic_id=topic.id))
    db.commit()

    run = asyncio.run(run_quality_evaluation(db, batch.id, 80))
    assert run.reference_count >= 50
    assert run.topic_count == 1
    assert set(run.metrics) >= {"teacher_fit", "specificity", "practicality", "novelty", "overall"}
    assert 0 <= run.metrics["overall"] <= 100


def test_calibration_recommends_thresholds_after_labels(tmp_path):
    db = make_session(tmp_path)
    teacher = Teacher(full_name="Петров Петр Петрович", department="САПРиПК")
    db.add(teacher)
    db.flush()
    titles = [
        "Разработка системы видеоаналитики спортивного матча с отслеживанием игроков",
        "Разработка веб-системы управления проектами с контролем сроков",
        "Разработка сервиса семантического поиска технической документации",
        "Разработка системы распознавания дефектов изделий по изображениям",
        "Разработка информационной системы учета оборудования предприятия",
        "Разработка рекомендательной системы образовательных материалов",
        "Разработка программного комплекса анализа сетевого трафика",
        "Разработка системы прогнозирования спроса на товары",
        "Разработка мобильного приложения мониторинга физической активности",
        "Разработка системы автоматизации обработки договорных документов",
    ]
    for title in titles:
        db.add(PastTopic(teacher_id=teacher.id, title=title, year=2025))
    db.commit()

    rows = asyncio.run(generate_calibration_pairs(db, 18))
    assert len(rows) >= 10
    unlabeled = [x for x in rows if x.label is None]
    # Для теста размечаем несколько крайних пар вручную, имитируя экспертную разметку.
    for row in sorted(unlabeled, key=lambda x: x.similarity_score)[:3]:
        row.label = "different"
    for row in sorted(unlabeled, key=lambda x: x.similarity_score, reverse=True)[:3]:
        row.label = "similar"
    db.commit()

    green, yellow, accuracy, labeled = recommend_thresholds(db)
    assert labeled >= 6
    assert green is not None and yellow is not None and accuracy is not None
    assert 0 <= green < yellow <= 100
    assert 0 <= accuracy <= 100
