import asyncio
import math
import re
from collections import Counter
from difflib import SequenceMatcher

from .local_embeddings import LocalEmbeddingProvider

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)

GENERIC_WORDS = {
    "разработка", "разработки", "разработке", "разработку", "создание", "проектирование", "реализация",
    "система", "системы", "систем", "систему", "системой", "системе",
    "программный", "программная", "программное", "программного", "программной", "программных",
    "модуль", "модуля", "модулей", "приложение", "приложения", "приложений",
    "инструмент", "инструмента", "комплекс", "комплекса", "платформа", "платформы",
    "для", "при", "по", "на", "с", "и", "в", "из", "от", "к", "о", "об", "под", "над", "через",
    "использование", "использованием", "применение", "применением", "основе", "помощью",
    "автоматический", "автоматическая", "автоматическое", "автоматизированный", "автоматизированная",
    "интеллектуальный", "интеллектуальная", "интеллектуального",
}

# Основные пороги для локальной embedding-модели. Для деградированного
# lexical-only режима применяются отдельные (более низкие) границы: его шкала
# распределена иначе и больше не притворяется шкалой embedding-режима.
SIMILARITY_REVIEW_THRESHOLD = 45.0
SIMILARITY_HIGH_THRESHOLD = 70.0
LEXICAL_REVIEW_THRESHOLD = 25.0
LEXICAL_HIGH_THRESHOLD = 50.0

# Небольшой слой доменных понятий нужен не вместо embeddings, а как страховка,
# если локальная ONNX-модель ещё не скачана. Он помогает не терять очевидные
# русскоязычные перефразировки из-за разных словоформ/синонимов.
CONCEPT_GROUPS: dict[str, tuple[str, ...]] = {
    "generation": ("генерац", "автогенерац", "синтез", "формирован"),
    "adaptive": ("адаптив", "персонализ", "индивидуал", "настройк сложност"),
    "assignments": ("задан", "упражнен", "тест", "контрольн работ"),
    "history": ("истори", "версионир", "трассиров", "фиксац", "жизненн цикл"),
    "research_work": ("научн работ", "научн проект", "учебн работ", "разработк проект"),
    "quality_control": ("контрол качеств", "дефект", "брак", "инспекц", "протокол контрол"),
    "construction": ("строител", "стройплощад", "сварн шв", "строительн шв"),
    "education": ("обучен", "учебн", "образован", "курс", "студент"),
    "multiagent": ("многоагент", "мультиагент", "агентн систем", "multi-agent"),
    "assistive": ("ассистив", "овз", "ограниченн возможност", "незряч", "слабовид", "глух", "жестов"),
    "computer_vision": ("компьютерн зрени", "изображен", "фотограф", "видео", "сегментац", "детекц"),
    "robotics": ("робот", "дрон", "беспилот", "навигац", "траектор"),
    "routing": ("маршрутиз", "маршрут", "логист", "доставк"),
}


def normalize_exact_title(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").casefold()).strip()
    return value.rstrip(" .;,:!?")


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _light_stem(token: str) -> str:
    token = token.casefold().replace("ё", "е")
    suffixes = (
        "ирования", "ирование", "ированию", "ированный", "ированная", "ированного",
        "ениями", "аниями", "иями", "ями", "ами", "енной", "енного", "енные", "енный", "енная",
        "ского", "ская", "ский", "ских", "иями", "ого", "ему", "ому", "ыми", "ими",
        "ание", "ания", "ений", "ение", "ения", "ний", "ние", "ция", "ции", "ций",
        "ость", "ости", "остей", "ая", "яя", "ое", "ее", "ые", "ие", "ой", "ий", "ый",
        "ов", "ев", "ам", "ям", "ах", "ях", "ом", "ем", "у", "ю", "а", "я", "ы", "и", "е",
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


def significant_tokens(text: str) -> list[str]:
    result: list[str] = []
    for raw in TOKEN_RE.findall((text or "").casefold()):
        if len(raw) < 3 or raw in GENERIC_WORDS:
            continue
        stem = _light_stem(raw)
        if len(stem) >= 3:
            result.append(stem)
    return result


def lexical_vector(text: str) -> Counter[str]:
    tokens = significant_tokens(text)
    features = tokens + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return Counter(features)


def lexical_cosine(a: str, b: str) -> float:
    if normalize_exact_title(a) and normalize_exact_title(a) == normalize_exact_title(b):
        return 1.0
    va, vb = lexical_vector(a), lexical_vector(b)
    if va == vb and va:
        return 1.0
    keys = set(va) | set(vb)
    dot = sum(va[k] * vb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def lexical_core_similarity(a: str, b: str) -> float:
    ta, tb = significant_tokens(a), significant_tokens(b)
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return 0.0
    jaccard = len(sa & sb) / len(sa | sb)
    sequence = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    return max(0.0, min(1.0, 0.62 * jaccard + 0.38 * sequence))


def _char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Counter[str]:
    normalized = re.sub(r"[^a-zа-яё0-9]+", " ", (text or "").casefold().replace("ё", "е"))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    grams: Counter[str] = Counter()
    for word in normalized.split():
        padded = f" {word} "
        for n in range(n_min, n_max + 1):
            for i in range(max(0, len(padded) - n + 1)):
                grams[padded[i:i+n]] += 1
    return grams


def _counter_cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if not na or not nb:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def char_similarity(a: str, b: str) -> float:
    return _counter_cosine(_char_ngrams(a), _char_ngrams(b))


def semantic_concepts(text: str) -> set[str]:
    low = re.sub(r"\s+", " ", (text or "").casefold().replace("ё", "е"))
    result: set[str] = set()
    for name, markers in CONCEPT_GROUPS.items():
        if any(marker in low for marker in markers):
            result.add(name)
    return result


def concept_similarity(a: str, b: str) -> float:
    sa, sb = semantic_concepts(a), semantic_concepts(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lexical_fallback_similarity(a: str, b: str) -> float:
    """Шкала 0..100 для режима без локальной embedding-модели."""
    if normalize_exact_title(a) and normalize_exact_title(a) == normalize_exact_title(b):
        return 100.0
    lexical = lexical_core_similarity(a, b)
    char = char_similarity(a, b)
    concepts = concept_similarity(a, b)
    # Берём несколько независимых сигналов. max(lexical, char) помогает русским
    # словоформам, concepts — устойчивым перефразировкам доменных действий.
    grounded = max(lexical, char * 0.82)
    score = (0.60 * grounded + 0.25 * concepts + 0.15 * lexical_cosine(a, b)) * 100.0
    return round(max(0.0, min(99.0, score)), 1)


def _semantic_signal(raw_cosine: float) -> float:
    # Для paraphrase-multilingual-MiniLM обычные разные фразы могут иметь
    # ненулевой cosine. Не трактуем 0.4 как «40% дубликата».
    value = max(-1.0, min(1.0, raw_cosine))
    return max(0.0, min(1.0, (value - 0.34) / 0.58))


def hybrid_similarity(a: str, b: str, embedding_cosine: float | None = None) -> float:
    """Гибридная шкала 0..100.

    ``embedding_cosine`` теперь обычно приходит из локального FastEmbed, а не
    из Mistral API. Функция оставлена отдельно для калибровки и regression tests.
    """
    if normalize_exact_title(a) and normalize_exact_title(a) == normalize_exact_title(b):
        return 100.0
    if embedding_cosine is None:
        return lexical_fallback_similarity(a, b)

    lexical = lexical_core_similarity(a, b)
    char = char_similarity(a, b)
    concepts = concept_similarity(a, b)
    semantic = _semantic_signal(embedding_cosine)
    grounding = max(lexical, char * 0.75, concepts * 0.9)

    # Embedding — главный сигнал перефразировки, но без какого-либо локального
    # подтверждения очень высокий cosine не должен превращать разные домены в дубль.
    score = 0.58 * semantic + 0.24 * grounding + 0.18 * concepts
    if concepts == 0 and lexical < 0.20 and char < 0.25:
        score *= 0.60
    elif grounding < 0.08 and concepts == 0:
        score *= 0.45
    return round(max(0.0, min(99.0, score * 100.0)), 1)


def thresholds_for_method(method: str | None) -> tuple[float, float]:
    if method == "local-lexical":
        return LEXICAL_REVIEW_THRESHOLD, LEXICAL_HIGH_THRESHOLD
    return SIMILARITY_REVIEW_THRESHOLD, SIMILARITY_HIGH_THRESHOLD


class SimilarityService:
    """Проверка сходства, независимая от Mistral API.

    1) exact duplicate;
    2) локальная multilingual embedding-модель FastEmbed;
    3) если модель ещё недоступна — честный local-lexical fallback с отдельными порогами.
    """

    def __init__(self, runtime_settings=None) -> None:
        # runtime_settings оставлен в сигнатуре для совместимости старого кода.
        self.runtime_settings = runtime_settings

    async def _embedding_map(self, texts: list[str]) -> tuple[dict[str, list[float]] | None, str]:
        unique = list(dict.fromkeys(text for text in texts if text and text.strip()))
        if not unique:
            return None, "none"
        try:
            result = await asyncio.to_thread(LocalEmbeddingProvider.embed_map, unique)
            if result and len(result) == len(unique):
                return result, "local-embedding"
        except Exception:
            pass
        return None, "local-lexical"

    @staticmethod
    def _exact_index(source: str, candidates: list[str]) -> int | None:
        normalized = normalize_exact_title(source)
        if not normalized:
            return None
        for index, candidate in enumerate(candidates):
            if normalize_exact_title(candidate) == normalized:
                return index
        return None

    @staticmethod
    def _best(source: str, candidates: list[str], embedding_map: dict[str, list[float]] | None) -> tuple[float, int]:
        scored: list[float] = []
        source_vec = embedding_map.get(source) if embedding_map else None
        for candidate in candidates:
            raw = None
            if embedding_map and source_vec is not None and candidate in embedding_map:
                raw = cosine(source_vec, embedding_map[candidate])
            scored.append(hybrid_similarity(source, candidate, raw))
        index = max(range(len(scored)), key=scored.__getitem__)
        return scored[index], index

    async def warm_embeddings(self, texts: list[str]) -> str:
        """Лениво прогревает локальную embedding-модель и кеш для profile scorer."""
        _mapping, method = await self._embedding_map(texts)
        return method

    async def closest(self, new_title: str, past_titles: list[str]) -> tuple[float, str | None, str]:
        if not past_titles:
            return 0.0, None, "none"
        exact = self._exact_index(new_title, past_titles)
        if exact is not None:
            return 100.0, past_titles[exact], "exact-duplicate"
        embedding_map, method = await self._embedding_map([new_title, *past_titles])
        score, index = self._best(new_title, past_titles, embedding_map)
        return score, past_titles[index], method

    async def closest_bulk(self, requests: list[tuple[str, list[str]]]) -> list[tuple[float, str | None, str]]:
        if not requests:
            return []
        texts: list[str] = []
        for source, candidates in requests:
            texts.append(source)
            texts.extend(candidates)
        embedding_map, method = await self._embedding_map(texts)
        results: list[tuple[float, str | None, str]] = []
        for source, candidates in requests:
            if not candidates:
                results.append((0.0, None, "none"))
                continue
            exact = self._exact_index(source, candidates)
            if exact is not None:
                results.append((100.0, candidates[exact], "exact-duplicate"))
                continue
            score, index = self._best(source, candidates, embedding_map)
            results.append((score, candidates[index], method))
        return results

    async def score_pairs(self, pairs: list[tuple[str, str]]) -> list[tuple[float, str]]:
        if not pairs:
            return []
        texts = [text for pair in pairs for text in pair]
        embedding_map, method = await self._embedding_map(texts)
        results: list[tuple[float, str]] = []
        for a, b in pairs:
            if normalize_exact_title(a) == normalize_exact_title(b) and normalize_exact_title(a):
                results.append((100.0, "exact-duplicate"))
                continue
            raw = cosine(embedding_map[a], embedding_map[b]) if embedding_map else None
            results.append((hybrid_similarity(a, b, raw), method))
        return results
