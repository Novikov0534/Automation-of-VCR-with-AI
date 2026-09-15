from __future__ import annotations

from itertools import combinations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..models import AppSettings, PastTopic, SimilarityCalibrationPair
from ..runtime_settings import load_runtime_settings
from .similarity import SimilarityService, lexical_cosine

LABEL_ORDER = {"different": 0, "similar": 1, "duplicate": 2}


def _pick_evenly(items: list[tuple[float, str, str]], count: int) -> list[tuple[str, str]]:
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return [(a, b) for _, a, b in items]
    result: list[tuple[str, str]] = []
    for i in range(count):
        idx = round(i * (len(items) - 1) / max(1, count - 1))
        _, a, b = items[idx]
        if (a, b) not in result:
            result.append((a, b))
    return result


async def generate_calibration_pairs(db: Session, count: int = 28) -> list[SimilarityCalibrationPair]:
    titles = list(dict.fromkeys(
        db.scalars(
            select(PastTopic.title)
            .order_by(PastTopic.is_reference.desc(), PastTopic.id.desc())
            .limit(140)
        ).all()
    ))
    if len(titles) < 8:
        raise ValueError("Для калибровки нужно хотя бы 8 прошлых тем")

    scored: list[tuple[float, str, str]] = []
    for a, b in combinations(titles, 2):
        scored.append((lexical_cosine(a, b) * 100, a, b))
    scored.sort(key=lambda x: x[0])

    exact_count = min(4, max(2, count // 7))
    low_count = max(5, (count - exact_count) // 3)
    high_count = low_count
    mid_count = max(4, count - exact_count - low_count - high_count)

    low = _pick_evenly(scored[: max(low_count * 8, low_count)], low_count)
    high_pool = [item for item in scored if item[0] < 99.5]
    high = _pick_evenly(high_pool[-max(high_count * 10, high_count):], high_count)
    mid_pool = sorted(scored, key=lambda x: abs(x[0] - 45.0))[: max(mid_count * 10, mid_count)]
    mid = _pick_evenly(sorted(mid_pool, key=lambda x: x[0]), mid_count)
    exact = [(titles[i], titles[i]) for i in range(min(exact_count, len(titles)))]

    pairs = list(dict.fromkeys(exact + high + mid + low))[:count]
    runtime = load_runtime_settings(db)
    scored_actual = await SimilarityService(runtime).score_pairs(pairs)

    db.execute(delete(SimilarityCalibrationPair))
    settings = db.get(AppSettings, 1)
    if not settings:
        settings = AppSettings(id=1)
        db.add(settings)
    settings.similarity_calibrated = False
    created: list[SimilarityCalibrationPair] = []
    for (left, right), (score, method) in zip(pairs, scored_actual):
        row = SimilarityCalibrationPair(
            left_title=left,
            right_title=right,
            similarity_score=score,
            similarity_method=method,
            label="duplicate" if left == right else None,
        )
        db.add(row)
        created.append(row)
    db.commit()
    for row in created:
        db.refresh(row)
    return created


def recommend_thresholds(db: Session) -> tuple[float | None, float | None, float | None, int]:
    rows = db.scalars(
        select(SimilarityCalibrationPair).where(SimilarityCalibrationPair.label.is_not(None))
    ).all()
    if len(rows) < 6:
        return None, None, None, len(rows)

    present = {row.label for row in rows if row.label}
    if not {"different", "similar", "duplicate"}.issubset(present):
        return None, None, None, len(rows)

    values = sorted({round(float(row.similarity_score), 3) for row in rows})
    candidates = {0.0, 100.0}
    candidates.update(values)
    for left, right in zip(values, values[1:]):
        candidates.add(round((left + right) / 2, 3))
    ordered = sorted(candidates)

    best: tuple[float, float, float, float, float] | None = None
    # Оптимизируем balanced accuracy, чтобы многочисленные "разные" пары не
    # вытесняли классы "похожие" и "дубликаты". При равенстве предпочитаем
    # более широкий безопасный коридор между границами.
    for green in ordered:
        for yellow in ordered:
            if yellow <= green:
                continue
            per_class: list[float] = []
            correct = 0
            for class_index, label in enumerate(("different", "similar", "duplicate")):
                class_rows = [row for row in rows if row.label == label]
                if not class_rows:
                    continue
                class_correct = 0
                for row in class_rows:
                    predicted = 0 if row.similarity_score < green else 1 if row.similarity_score < yellow else 2
                    ok = predicted == class_index
                    class_correct += int(ok)
                    correct += int(ok)
                per_class.append(class_correct / len(class_rows))
            balanced = sum(per_class) / len(per_class)
            overall = correct / len(rows)
            width = yellow - green
            # Чуть предпочитаем границы, которые не лежат прямо на размеченной точке.
            min_distance = min(abs(score - green) for score in values) + min(abs(score - yellow) for score in values)
            candidate = (balanced, overall, min_distance, width, -green)
            if best is None or candidate > best:
                best = candidate
                best_green, best_yellow = green, yellow
    if best is None:
        return None, None, None, len(rows)
    return round(float(best_green), 1), round(float(best_yellow), 1), round(best[1] * 100, 1), len(rows)

def apply_recommended_thresholds(db: Session) -> tuple[float, float, float, int]:
    green, yellow, accuracy, labeled = recommend_thresholds(db)
    if green is None or yellow is None or accuracy is None:
        raise ValueError("Разметьте хотя бы 6 пар тем")
    settings = db.get(AppSettings, 1)
    if not settings:
        settings = AppSettings(id=1)
        db.add(settings)
    settings.similarity_green_max = green
    settings.similarity_yellow_max = yellow
    settings.similarity_calibrated = True
    db.commit()
    return green, yellow, accuracy, labeled
