"""Локальная таксономия научных направлений для demo-генератора.

Она не заменяет embeddings/LLM. Задача словаря — дать предсказуемую
"псевдосемантику" без внешнего AI: связать короткое название профиля
преподавателя с терминами и категориями встроенного банка тем.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from .local_embeddings import LocalEmbeddingProvider


@dataclass(frozen=True)
class ResearchDirection:
    name: str
    aliases: tuple[str, ...]
    terms: tuple[str, ...]
    category_weights: dict[str, float]


RESEARCH_DIRECTIONS: tuple[ResearchDirection, ...] = (
    ResearchDirection(
        "Системы искусственного интеллекта",
        ("искусственный интеллект", "системы искусственного интеллекта", "система искусственного интеллекта", "системы искусственный интеллект", "ии", "ai"),
        ("интеллектуальная система", "интеллектуальный анализ", "интеллектуальная обработка", "экспертная система", "машинное обучение", "нейросеть", "классификация", "прогнозирование", "рекомендательная система", "генерация", "адаптивная система", "многоагентная система"),
        {
            "Анализ данных и прогнозирование": 1.6,
            "Обработка естественного языка": 1.15,
            "Компьютерное зрение": 1.15,
            "Игры, симуляции и генеративные системы": 0.6,
            "Аудио и речь": 0.55,
        },
    ),
    ResearchDirection(
        "Машинное обучение и анализ данных",
        ("машинное обучение", "анализ данных", "data science", "ml"),
        ("классификация", "кластеризация", "прогнозирование", "временные ряды", "аномалии", "выявление аномалий", "предиктивная аналитика", "модель"),
        {
            "Анализ данных и прогнозирование": 2.2,
            "Компьютерное зрение": 0.7,
            "Обработка естественного языка": 0.65,
            "Прикладные системы: финансы и образование": 0.5,
        },
    ),
    ResearchDirection(
        "Многоагентные системы",
        ("многоагентные системы", "многоагентная система", "мультиагентные системы", "multi-agent", "multiagent"),
        ("многоагентная система", "мультиагентная система", "агентное взаимодействие", "координация агентов", "автономные агенты", "агентная архитектура"),
        {
            "Веб-платформы и информационные системы": 0.65,
            "Алгоритмические и оптимизационные задачи": 0.55,
            "Прикладные системы: финансы и образование": 0.45,
            "Игры, симуляции и генеративные системы": 0.35,
        },
    ),
    ResearchDirection(
        "Системный анализ",
        ("системный анализ", "анализ систем"),
        ("моделирование систем", "бизнес-процессы", "принятие решений", "управление процессами", "анализ требований", "оптимизация процессов", "жизненный цикл", "мониторинг качества", "управление проектами", "управление историей", "зависимости", "автоматизация процессов", "учебные проекты", "портфель проектов", "риски", "распределение ресурсов", "согласование", "контроль сроков", "узкие места", "приоритизация"),
        {
            "Веб-платформы и информационные системы": 1.4,
            "Алгоритмические и оптимизационные задачи": 0.9,
            "Прикладные системы: финансы и образование": 0.85,
            "Анализ данных и прогнозирование": 0.45,
        },
    ),
    ResearchDirection(
        "Компьютерное зрение",
        ("компьютерное зрение", "computer vision", "распознавание изображений", "обработка изображений"),
        ("изображение", "фотография", "детекция объектов", "сегментация", "классификация изображений", "визуальные признаки", "снимки", "камера", "видеоаналитика", "обнаружение дефектов"),
        {"Компьютерное зрение": 2.8, "Доступность и мобильные технологии": 0.35},
    ),
    ResearchDirection(
        "Видеоаналитика",
        ("видеоаналитика", "анализ видео", "обработка видео", "видеопоток"),
        ("видео", "трекинг", "отслеживание объектов", "камеры", "распознавание действий", "кадр", "видеозапись"),
        {"Компьютерное зрение": 2.65, "Аудио и речь": 0.25},
    ),
    ResearchDirection(
        "Робототехника",
        ("робототехника", "роботы", "робот"),
        ("мобильный робот", "навигация", "лидар", "автономное управление", "планирование движения", "группа роботов", "робот-курьер"),
        {"Робототехника и IoT": 2.8, "Алгоритмические и оптимизационные задачи": 0.35},
    ),
    ResearchDirection(
        "Интернет вещей и встраиваемые системы",
        ("интернет вещей", "iot", "встраиваемые системы", "embedded systems"),
        ("датчик", "сенсор", "микроконтроллер", "телеметрия", "умный дом", "умная теплица", "мониторинг оборудования", "беспроводные устройства"),
        {"Робототехника и IoT": 2.65, "Сети и распределённые системы": 0.45},
    ),
    ResearchDirection(
        "Мультимедийные и игровые технологии",
        ("мультимедийные и игровые технологии", "игровые технологии", "мультимедиа", "компьютерные игры"),
        ("компьютерная игра", "игровая система", "игровой движок", "генерация уровней", "персонажи", "симулятор", "геймдев", "game development"),
        {"Игры, симуляции и генеративные системы": 2.7, "Аудио и речь": 0.45},
    ),
    ResearchDirection(
        "Разработка обучающих игр",
        ("разработка обучающих игр", "обучающие игры", "образовательные игры", "serious games"),
        ("обучающая игра", "геймификация", "образовательная игра", "тренажёр", "учебный симулятор", "обучение"),
        {"Игры, симуляции и генеративные системы": 2.35, "Прикладные системы: финансы и образование": 1.1},
    ),
    ResearchDirection(
        "Ассистивные технологии",
        ("ассистивные технологии", "assistive technologies", "цифровая доступность", "доступные технологии"),
        ("доступность", "овз", "ограниченные возможности", "ограниченными возможностями", "люди с ограниченными возможностями", "инвалидность", "специальные потребности", "незрячие", "слабовидящие", "нарушение зрения", "нарушение слуха", "глухие", "жестовый язык", "субтитры", "голосовое сопровождение", "альтернативный текст", "скринридер"),
        {
            # Категория содержит и обычные мобильные темы, поэтому общий бонус
            # умеренный: реальные assistive-темы должны выигрывать за счёт
            # слов «незрячие», «доступность», «жестовый язык» и т. п.
            "Доступность и мобильные технологии": 0.25,
            "Компьютерное зрение": 0.15,
            "Аудио и речь": 0.20,
            "Геоинформационные и пространственные системы": 0.10,
        },
    ),
    ResearchDirection(
        "Чат-боты и диалоговые системы",
        ("чат-боты и диалоговые системы", "чат-боты", "чат-бот", "диалоговые системы", "conversational ai"),
        ("telegram", "телеграм", "vk", "вконтакте", "discord", "мессенджер", "виртуальный ассистент", "разговорный интерфейс", "диалог", "тикет", "сообщение", "служба поддержки", "nlp"),
        {"Обработка естественного языка": 0.72, "Веб-платформы и информационные системы": 0.62},
    ),
    ResearchDirection(
        "Компьютерная лингвистика и NLP",
        ("компьютерная лингвистика", "nlp", "обработка естественного языка", "natural language processing"),
        ("естественный язык", "обработка текста", "семантический анализ", "извлечение информации", "классификация текста", "реферирование", "терминология", "текстовые документы"),
        {"Обработка естественного языка": 2.9, "Анализ данных и прогнозирование": 0.35},
    ),
    ResearchDirection(
        "Речевые и аудиотехнологии",
        ("речевые и аудиотехнологии", "речевые технологии", "обработка аудио", "speech technologies"),
        ("речь", "аудио", "распознавание речи", "speech-to-text", "синтез речи", "диаризация", "эмоции речи", "обработка звука", "музыка"),
        {"Аудио и речь": 2.95, "Обработка естественного языка": 0.45, "Доступность и мобильные технологии": 0.25},
    ),
    ResearchDirection(
        "Информационные системы и веб-технологии",
        ("информационные системы", "веб-технологии", "web development", "веб-разработка"),
        ("информационная система", "веб-система", "веб-приложение", "веб-сервис", "веб-платформа", "api", "автоматизация", "платформа", "личный кабинет", "workflow", "управление проектами"),
        {"Веб-платформы и информационные системы": 2.75, "Прикладные системы: финансы и образование": 0.45},
    ),
    ResearchDirection(
        "Кибербезопасность",
        ("кибербезопасность", "информационная безопасность", "кибер безопасность", "cybersecurity"),
        ("уязвимость", "фишинг", "cve", "атака", "сетевые угрозы", "аудит безопасности", "компрометация", "целостность", "защита информации"),
        {"Кибербезопасность": 3.0, "Сети и распределённые системы": 0.55},
    ),
    ResearchDirection(
        "Компьютерные сети и распределённые системы",
        ("компьютерные сети", "распределённые системы", "сетевые технологии", "computer networks"),
        ("сеть", "сетевой трафик", "распределённая система", "балансировка нагрузки", "отказоустойчивость", "синхронизация данных", "сетевое оборудование", "узлы"),
        {"Сети и распределённые системы": 2.95, "Кибербезопасность": 0.45},
    ),
    ResearchDirection(
        "Геоинформационные системы",
        ("геоинформационные системы", "гис", "gis", "геоинформатика"),
        ("геоданные", "картография", "пространственные данные", "карта", "координаты", "спутниковые снимки", "геолокация", "изохроны", "рельеф"),
        {"Геоинформационные и пространственные системы": 3.0, "Компьютерное зрение": 0.25},
    ),
    ResearchDirection(
        "Алгоритмы и оптимизация",
        ("алгоритмы и оптимизация", "оптимизация", "алгоритмы", "исследование операций"),
        ("расписание", "распределение ресурсов", "маршрутизация", "оптимальное решение", "планирование", "минимизация", "балансировка", "раскрой"),
        {"Алгоритмические и оптимизационные задачи": 3.0, "Геоинформационные и пространственные системы": 0.4, "Робототехника и IoT": 0.35},
    ),
    ResearchDirection(
        "Мобильные технологии",
        ("мобильные технологии", "мобильная разработка", "mobile development", "мобильные приложения"),
        ("мобильное приложение", "android", "ios", "смартфон", "мобильная система", "приложение для телефона"),
        {"Доступность и мобильные технологии": 2.65, "Анализ данных и прогнозирование": 0.2},
    ),
    ResearchDirection(
        "Базы данных и управление данными",
        ("базы данных", "база данных", "управление данными", "database systems"),
        ("бд", "хранилище", "документы", "поиск", "индекс", "синхронизация", "управление данными", "версионирование", "распределённая база"),
        {"Веб-платформы и информационные системы": 1.35, "Сети и распределённые системы": 1.1, "Обработка естественного языка": 0.25},
    ),
)

RESEARCH_AREA_NAMES: tuple[str, ...] = tuple(item.name for item in RESEARCH_DIRECTIONS if item.name != "Многоагентные системы")

_STOP_TOKENS = {
    "разработка", "разработки", "система", "системы", "систем", "технологии", "технологий",
    "для", "при", "по", "на", "с", "со", "и", "или", "в", "во", "из", "от", "к", "о", "об",
    "автоматическая", "автоматической", "автоматизированная", "автоматизированной", "программный",
    "программного", "данные", "данных", "учетом", "учётом", "метод", "методы", "задача", "задачи",
}


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-zа-яё0-9+#-]+", (value or "").casefold().replace("ё", "е")))


def _token_key(token: str) -> str:
    token = token.casefold().replace("ё", "е").strip()
    # Очень лёгкая русская нормализация без внешней NLP-зависимости.
    # В отличие от усечения по первым буквам она не смешивает, например,
    # «ассистивные» и «ассистент». Делаем до двух проходов, чтобы
    # «жестовый» и «жестов» сошлись к одному основанию «жест».
    if re.search(r"[а-я]", token) and len(token) > 4:
        suffixes = (
            "иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими",
            "иях", "ах", "ях", "ией", "иям", "ием",
            "ение", "ения", "ений",
            "ие", "ые", "ое", "ая", "яя", "ий", "ый", "ой",
            "их", "ых", "ам", "ям", "ом", "ем", "ым", "им", "ов", "ев",
            "ия", "ию", "ью", "ей",
            "ы", "и", "а", "я", "у", "ю", "е", "о", "ь",
        )
        for _ in range(2):
            changed = False
            for suffix in suffixes:
                if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                    token = token[:-len(suffix)]
                    changed = True
                    break
            if not changed:
                break
    return token


def semantic_tokens(value: str) -> set[str]:
    result: set[str] = set()
    for token in re.findall(r"[a-zа-яё0-9+#-]+", (value or "").casefold().replace("ё", "е")):
        if len(token) < 3 or token in _STOP_TOKENS:
            continue
        result.add(_token_key(token))
    return result


def _phrase_present(text: str, phrase: str) -> bool:
    text_n = f" {_normalize_text(text)} "
    phrase_n = _normalize_text(phrase)
    return bool(phrase_n and f" {phrase_n} " in text_n)


def direction_strengths(values: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Определяет, какие направления выражены в наборе текстовых сигналов.

    Значение tuple — (текст, вес источника). Научные области/фокус можно
    передать с большим весом, прошлые темы — с чуть меньшим.
    """
    strengths: dict[str, float] = {}
    for raw, source_weight in values:
        text = (raw or "").strip()
        if not text:
            continue
        text_tokens = semantic_tokens(text)
        for direction in RESEARCH_DIRECTIONS:
            direct = any(_phrase_present(text, alias) for alias in (direction.name, *direction.aliases))
            term_tokens = semantic_tokens(" ".join(direction.terms))
            overlap = len(text_tokens & term_tokens)
            if direct:
                score = 3.0 * source_weight + min(overlap, 4) * 0.25 * source_weight
            elif overlap >= 2:
                score = min(2.0, 0.55 * overlap) * source_weight
            elif overlap == 1 and len(text_tokens) <= 4:
                # Короткое поле вроде "лидар" или "диаризация" тоже должно работать.
                score = 0.55 * source_weight
            else:
                continue
            strengths[direction.name] = strengths.get(direction.name, 0.0) + score
    return strengths


def local_topic_relevance(
    title: str,
    category: str,
    *,
    research_areas: Iterable[str] = (),
    past_topics: Iterable[str] = (),
    focus: str | None = None,
) -> float:
    """Возвращает локальный рейтинг темы без внешней AI-модели."""
    source_values: list[tuple[str, float]] = []
    source_values.extend((value, 1.8) for value in research_areas if value)
    source_values.extend((value, 1.0) for value in past_topics if value)
    if focus and focus.strip():
        source_values.append((focus.strip(), 2.0))

    if not source_values:
        return 0.0

    topic_text = f"{category} {title}"
    topic_tokens = semantic_tokens(topic_text)
    title_tokens = semantic_tokens(title)
    signal_tokens = semantic_tokens(" ".join(value for value, _ in source_values))
    lexical_overlap = len(topic_tokens & signal_tokens)
    score = lexical_overlap * 0.34

    strengths = direction_strengths(source_values)
    by_name = {item.name: item for item in RESEARCH_DIRECTIONS}
    for name, strength in strengths.items():
        direction = by_name[name]
        score += direction.category_weights.get(category, 0.0) * strength
        term_matches = 0
        for term in direction.terms:
            required = semantic_tokens(term)
            if required and required.issubset(title_tokens):
                term_matches += 1
        score += min(term_matches, 5) * 1.05 * min(strength, 5.0)

    return score


@dataclass(frozen=True)
class ProfileRelevanceResult:
    score: float
    matched_research_areas: tuple[str, ...]
    matched_directions: tuple[str, ...]
    dominant_profile_directions: tuple[str, ...]
    foreign_directions: tuple[str, ...] = ()


def _direction_text_score(text: str, direction: ResearchDirection) -> float:
    """Насколько текст выражает конкретное направление (0..8+).

    Эта функция намеренно чувствительнее ``direction_strengths``: для короткой
    темы достаточно одного характерного термина, тогда как профиль строится по
    нескольким более надёжным источникам.
    """
    value = (text or "").strip()
    if not value:
        return 0.0
    score = 0.0
    tokens = semantic_tokens(value)
    aliases = (direction.name, *direction.aliases)
    if any(_phrase_present(value, alias) for alias in aliases):
        score += 4.5
    else:
        # Формы вроде «видеоаналитики»/«информационной системы» не должны
        # теряться из-за точного phrase-match: сравниваем нормализованные корни.
        for alias in aliases:
            alias_tokens = semantic_tokens(alias)
            if alias_tokens and len(tokens & alias_tokens) / len(alias_tokens) >= 0.75:
                score += 2.6
                break
    for term in direction.terms:
        term_tokens = semantic_tokens(term)
        if not term_tokens:
            continue
        if _phrase_present(value, term):
            score += 1.8
        else:
            overlap = len(tokens & term_tokens) / max(1, len(term_tokens))
            if overlap >= 0.66:
                score += 0.8 * overlap
    return score


def topic_direction_scores(text: str) -> dict[str, float]:
    return {
        direction.name: score
        for direction in RESEARCH_DIRECTIONS
        if (score := _direction_text_score(text, direction)) > 0
    }


def profile_direction_scores(
    *,
    research_areas: Iterable[str] = (),
    past_topics: Iterable[str] = (),
    focus: str | None = None,
) -> dict[str, float]:
    """Строит локальный профиль: заявленные области + повторяющиеся мотивы истории."""
    result: dict[str, float] = {}
    by_name = {item.name: item for item in RESEARCH_DIRECTIONS}

    for area in research_areas:
        area = (area or "").strip()
        if not area:
            continue
        area_tokens = semantic_tokens(area)
        direct_matches: list[tuple[float, ResearchDirection]] = []
        for direction in RESEARCH_DIRECTIONS:
            best = 0.0
            for alias in (direction.name, *direction.aliases):
                alias_tokens = semantic_tokens(alias)
                if not alias_tokens:
                    continue
                if _phrase_present(area, alias):
                    best = max(best, 1.0)
                else:
                    best = max(best, len(area_tokens & alias_tokens) / len(alias_tokens))
            if best >= 0.74:
                direct_matches.append((best, direction))
        if direct_matches:
            best_score = max(score for score, _ in direct_matches)
            # Явное research_area не размазываем по случайно похожим направлениям.
            for match_score, direction in direct_matches:
                if match_score >= best_score - 0.08:
                    result[direction.name] = result.get(direction.name, 0.0) + 5.5 * match_score
            continue

        strengths = direction_strengths([(area, 1.0)])
        if strengths:
            for name, value in strengths.items():
                result[name] = result.get(name, 0.0) + max(3.0, value * 1.4)

    # История нужна прежде всего для методологической "подписи": повторяющееся
    # направление (например, многоагентность) быстро накапливает вес.
    for title in past_topics:
        for name, value in topic_direction_scores(title or "").items():
            result[name] = result.get(name, 0.0) + min(3.5, value) * 0.75

    if focus and focus.strip():
        for name, value in topic_direction_scores(focus).items():
            result[name] = result.get(name, 0.0) + min(4.0, value) * 0.9

    return result


def dominant_profile_directions(
    *, research_areas: Iterable[str] = (), past_topics: Iterable[str] = (), focus: str | None = None, limit: int = 5
) -> list[str]:
    scores = profile_direction_scores(research_areas=research_areas, past_topics=past_topics, focus=focus)
    return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]



# Близкие направления считаются одной тематической семьёй при поиске
# «чужой специализации». Это не даёт, например, пометить видеоаналитику как
# чужую для преподавателя компьютерного зрения или обучающую игру как чужую
# для профиля разработки обучающих игр.
_DIRECTION_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"Компьютерное зрение", "Видеоаналитика"}),
    frozenset({"Мультимедийные и игровые технологии", "Разработка обучающих игр"}),
    frozenset({"Системы искусственного интеллекта", "Машинное обучение и анализ данных", "Многоагентные системы"}),
    frozenset({"Компьютерная лингвистика и NLP", "Чат-боты и диалоговые системы", "Речевые и аудиотехнологии"}),
    frozenset({"Системный анализ", "Информационные системы и веб-технологии", "Базы данных и управление данными"}),
    frozenset({"Компьютерные сети и распределённые системы", "Кибербезопасность"}),
    frozenset({"Робототехника", "Интернет вещей и встраиваемые системы"}),
)

def _direction_owned_or_related(name: str, owned: set[str]) -> bool:
    if name in owned:
        return True
    return any(name in family and bool(owned & family) for family in _DIRECTION_FAMILIES)

def _vector_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _semantic_area_signal(text: str, areas: list[str]) -> tuple[float, list[str]]:
    """Дополнительная семантическая проверка research_areas локальными embeddings.

    Не запускает скачивание модели самостоятельно: используется только если
    SimilarityService уже прогрел локальную embedding-модель в этом процессе.
    Это исправляет случаи, где таксономия не знает пользовательскую формулировку
    области (например, инженерная механика/киберфизические системы).
    """
    if not text.strip() or not areas or not LocalEmbeddingProvider.ready():
        return 0.0, []
    vectors = LocalEmbeddingProvider.embed_map([text, *areas])
    if not vectors or text not in vectors:
        return 0.0, []
    scores: list[tuple[str, float]] = []
    for area in areas:
        if area not in vectors:
            continue
        raw = _vector_cosine(vectors[text], vectors[area])
        # Для multilingual MiniLM короткие названия областей имеют заметный
        # базовый cosine даже без прямого совпадения. Нормируем 0.30..0.68 в 0..100.
        normalized = max(0.0, min(100.0, (raw - 0.30) / 0.38 * 100.0))
        scores.append((area, normalized))
    if not scores:
        return 0.0, []
    scores.sort(key=lambda item: item[1], reverse=True)
    best = scores[0][1]
    matched = [area for area, value in scores if value >= 52.0]
    return best, matched


def profile_relevance_details(
    title: str,
    *,
    rationale: str | None = None,
    keywords: Iterable[str] = (),
    research_areas: Iterable[str] = (),
    past_topics: Iterable[str] = (),
    focus: str | None = None,
) -> ProfileRelevanceResult:
    """Оценивает соответствие AI-темы профилю преподавателя в шкале 0..100.

    В отличие от старого ``local_topic_relevance`` эта функция предназначена
    именно для поствалидации ответов LLM и не зависит от категории demo-банка.
    """
    areas = [str(x).strip() for x in research_areas if str(x).strip()]
    history = [str(x).strip() for x in past_topics if str(x).strip()]
    aux_text = " ".join([rationale or "", " ".join(str(x) for x in keywords if str(x).strip())])
    title_scores = topic_direction_scores(title or "")
    topic_scores = dict(title_scores)
    for name, value in topic_direction_scores(aux_text).items():
        topic_scores[name] = topic_scores.get(name, 0.0) + value * 0.22
    text = " ".join([title or "", aux_text])
    profile_scores = profile_direction_scores(research_areas=areas, past_topics=history, focus=focus)

    dominant = [name for name, _ in sorted(profile_scores.items(), key=lambda item: item[1], reverse=True)[:5]]
    if not profile_scores:
        # Нет профиля — нельзя честно назвать тему нерелевантной.
        return ProfileRelevanceResult(60.0, tuple(), tuple(topic_scores), tuple(), tuple())

    keys = set(profile_scores) | set(topic_scores)
    dot = sum(profile_scores.get(k, 0.0) * topic_scores.get(k, 0.0) for k in keys)
    pn = sum(v * v for v in profile_scores.values()) ** 0.5
    tn = sum(v * v for v in topic_scores.values()) ** 0.5
    direction_cosine = dot / (pn * tn) if pn and tn else 0.0

    matched_areas: list[str] = []
    area_strength = 0.0
    for area in areas:
        inferred = direction_strengths([(area, 1.0)])
        names = list(inferred)
        if not names:
            names = [d.name for d in RESEARCH_DIRECTIONS if _direction_text_score(area, d) > 0]
        best = max((topic_scores.get(name, 0.0) for name in names), default=0.0)
        if best >= 0.75:
            matched_areas.append(area)
        area_strength = max(area_strength, min(1.0, best / 1.2))

    # Дополнительный текстовый сигнал нужен для пользовательских research_areas,
    # которых пока нет в таксономии.
    title_tokens = semantic_tokens(text)
    profile_tokens = semantic_tokens(" ".join([*areas, *history[-8:], focus or ""]))
    overlap = len(title_tokens & profile_tokens) / max(1, min(8, len(profile_tokens)))
    token_signal = min(1.0, overlap * 2.0)

    score = 72.0 * direction_cosine + 20.0 * area_strength + 8.0 * token_signal

    # v33: семантика research_areas через уже загруженную локальную embedding-модель.
    # Берём максимум, а не среднее: одной теме достаточно честно соответствовать
    # хотя бы одной области преподавателя; покрытие всех областей контролируется
    # уже на уровне набора тем.
    semantic_text = " ".join([title or "", rationale or "", " ".join(str(x) for x in keywords if str(x).strip())]).strip()
    semantic_score, semantic_areas = _semantic_area_signal(semantic_text, areas)
    if semantic_score:
        score = max(score, semantic_score)
        for area in semantic_areas:
            if area not in matched_areas:
                matched_areas.append(area)

    # Отдельно ловим «тема по технологии подходит, но предметная специализация
    # ушла к другому преподавателю». Пример batch_14: компьютерное зрение +
    # ассистивная задача у преподавателя без ассистивного профиля.
    explicit_profile = profile_direction_scores(research_areas=areas, past_topics=(), focus=None)
    owned = {name for name, value in explicit_profile.items() if value >= 2.5}
    if not owned:
        owned = {name for name, value in profile_scores.items() if value >= 2.5}
    strongest_owned = max((topic_scores.get(name, 0.0) for name in owned), default=0.0)
    foreign = [
        name for name, value in topic_scores.items()
        if not _direction_owned_or_related(name, owned) and value >= 1.8 and value > strongest_owned * 1.05
    ]
    foreign.sort(key=lambda name: topic_scores.get(name, 0.0), reverse=True)
    if foreign:
        # Не зануляем междисциплинарную тему, а делаем её явным warning.
        # В генераторе такой кандидат отклоняется, если итоговый score < 45.
        score -= min(28.0, 8.0 * topic_scores[foreign[0]])

    return ProfileRelevanceResult(
        round(max(0.0, min(100.0, score)), 1),
        tuple(matched_areas),
        tuple(name for name, _ in sorted(topic_scores.items(), key=lambda item: item[1], reverse=True)[:5]),
        tuple(dominant),
        tuple(foreign[:3]),
    )
