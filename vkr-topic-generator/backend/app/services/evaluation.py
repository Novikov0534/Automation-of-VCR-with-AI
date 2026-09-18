from __future__ import annotations

import math
import re
from statistics import mean

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models import GeneratedTopic, GenerationBatchTopic, PastTopic, QualityEvaluationRun, Teacher
from ..utils import teacher_table_name
from .research_taxonomy import profile_relevance_details

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9+#./-]+")

IMPLEMENTATION = ("разработка", "проектирование", "создание", "реализация", "программирован")
PRODUCT = ("систем", "сервис", "платформ", "комплекс", "приложен", "модул", "алгоритм", "инструмент", "конфигурац")
TASK = (
    "монитор", "прогноз", "классификац", "распознаван", "обнаружен", "поиск", "управлен", "учет", "учёт",
    "автоматизац", "рекомендац", "визуализац", "контрол", "обработк", "генерац", "планирован", "сопоставлен",
    "транскриб", "отслежив", "диагност", "оптимизац", "аналитик", "интеграц", "защит",
)
SCOPE = ("база данных", "api", "интерфейс", "веб", "мобиль", "аналит", "уведом", "интеграц", "тест", "модель", "данн")
VAGUE = (
    "применение искусственного интеллекта", "исследование информационных технологий", "система анализа данных",
    "с использованием машинного обучения",
)


def clamp(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def reference_quality_score(title: str) -> float:
    low = " ".join(title.lower().split())
    words = WORD_RE.findall(title)
    score = 0.0
    if 9 <= len(words) <= 30:
        score += 20
    elif 7 <= len(words) <= 34:
        score += 12
    if any(low.startswith(x) for x in IMPLEMENTATION):
        score += 22
    if any(x in low for x in PRODUCT):
        score += 18
    if any(x in low for x in TASK):
        score += 18
    scope_hits = sum(1 for x in SCOPE if x in low)
    score += min(18, scope_hits * 6)
    if " с " in low or " для " in low:
        score += 8
    if any(v in low for v in VAGUE):
        score -= 25
    if low.startswith(("исследование ", "анализ ", "применение ")):
        score -= 18
    return clamp(score)


def practicality_score(title: str, rationale: str | None = None) -> float:
    low = f"{title} {rationale or ''}".lower()
    score = 20.0
    score += 25 if any(x in low for x in IMPLEMENTATION) else 0
    score += 20 if any(x in low for x in PRODUCT) else 0
    score += min(25, sum(1 for x in SCOPE if x in low) * 5)
    if rationale and len(rationale) > 100:
        score += 10
    if low.startswith(("исследование ", "анализ ")):
        score -= 20
    return clamp(score)


def teacher_fit_from_similarity(score: float, has_history: bool) -> float:
    if not has_history:
        return 60.0
    # Умеренная близость к ранее одобренным работам — хороший профильный сигнал;
    # почти дословная близость уже снижает оценку из-за риска повторения.
    anchors = [(0, 35), (15, 65), (30, 90), (45, 100), (60, 86), (75, 58), (100, 20)]
    for (x1, y1), (x2, y2) in zip(anchors, anchors[1:]):
        if x1 <= score <= x2:
            ratio = (score - x1) / (x2 - x1)
            return clamp(y1 + (y2 - y1) * ratio)
    return 35.0


def refresh_reference_set(db: Session, limit: int = 100) -> int:
    topics = db.scalars(select(PastTopic)).all()
    ranked = sorted(topics, key=lambda item: (reference_quality_score(item.title), item.id), reverse=True)
    target_count = min(limit, len(ranked))
    # Стараемся иметь хотя бы 50 эталонов, если историческая база это позволяет.
    good = [item for item in ranked if reference_quality_score(item.title) >= 62]
    selected = good[:target_count]
    if len(selected) < min(50, target_count):
        selected = ranked[:min(50, target_count)]
    selected_ids = {item.id for item in selected[:limit]}
    for item in topics:
        item.is_reference = item.id in selected_ids
    db.commit()
    return len(selected_ids)


def _heuristic_items(topics: list[GeneratedTopic]) -> list[dict]:
    items: list[dict] = []
    for topic in topics:
        history = topic.teacher.past_topics if topic.teacher else []
        specificity = reference_quality_score(topic.title)
        practicality = practicality_score(topic.title, topic.rationale)
        novelty = clamp(100.0 - float(topic.global_similarity_score or topic.similarity_score or 0.0))
        profile = profile_relevance_details(
            topic.title,
            rationale=topic.rationale,
            keywords=topic.keywords or [],
            research_areas=topic.teacher.research_areas or [],
            past_topics=[item.title for item in history],
        )
        teacher_fit = profile.score if ((topic.teacher.research_areas or []) or history) else teacher_fit_from_similarity(
            float(topic.teacher_similarity_score or 0.0), bool(history)
        )
        overall = clamp(mean([teacher_fit, specificity, practicality, novelty]))
        items.append({
            "topic_id": topic.id,
            "title": topic.title,
            "teacher_name": teacher_table_name(topic.teacher.full_name, topic.teacher.position),
            "teacher_fit": teacher_fit,
            "specificity": specificity,
            "practicality": practicality,
            "novelty": novelty,
            "overall": overall,
        })
    return items


async def run_quality_evaluation(db: Session, batch_id: int, reference_limit: int = 100) -> QualityEvaluationRun:
    reference_count = db.scalar(select(PastTopic.id).where(PastTopic.is_reference.is_(True)).limit(1))
    if reference_count is None:
        refresh_reference_set(db, reference_limit)

    references = db.scalars(
        select(PastTopic).where(PastTopic.is_reference.is_(True)).order_by(PastTopic.id.desc()).limit(reference_limit)
    ).all()
    if not references:
        raise ValueError("Сначала импортируйте или добавьте прошлые темы ВКР")

    topics = db.scalars(
        select(GeneratedTopic)
        .join(GenerationBatchTopic, GenerationBatchTopic.topic_id == GeneratedTopic.id)
        .where(GenerationBatchTopic.batch_id == batch_id, GeneratedTopic.status != "rejected")
        .options(selectinload(GeneratedTopic.teacher).selectinload(Teacher.past_topics))
        .order_by(GeneratedTopic.id)
    ).unique().all()
    if not topics:
        raise ValueError("В выбранном наборе нет тем для оценки")

    items = _heuristic_items(topics)
    mode = "heuristic"

    reference_specificity = mean(reference_quality_score(x.title) for x in references)
    reference_practicality = mean(practicality_score(x.title) for x in references)
    metrics = {
        "teacher_fit": round(mean(x["teacher_fit"] for x in items), 1),
        "specificity": round(mean(x["specificity"] for x in items), 1),
        "practicality": round(mean(x["practicality"] for x in items), 1),
        "novelty": round(mean(x["novelty"] for x in items), 1),
        "overall": round(mean(x["overall"] for x in items), 1),
        "reference_specificity": round(reference_specificity, 1),
        "reference_practicality": round(reference_practicality, 1),
    }
    run = QualityEvaluationRun(
        batch_id=batch_id,
        reference_count=len(references),
        topic_count=len(topics),
        mode=mode,
        metrics=metrics,
        items=items,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run
