import math
import re
from collections import Counter
from difflib import SequenceMatcher

from .mistral import MistralClient

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)

# Слова, которые встречаются почти в каждой формулировке ВКР и сами по себе
# не означают, что темы действительно похожи.
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

# Фиксированная шкала нашего гибридного индекса. Это НЕ raw cosine Mistral.
# 0–44: различаются; 45–69: есть заметное пересечение; 70–99: высокая близость; 100: дословный дубликат.
SIMILARITY_REVIEW_THRESHOLD = 45.0
SIMILARITY_HIGH_THRESHOLD = 70.0


def normalize_exact_title(text: str) -> str:
    """Нормализация только для фактического текстового дубликата."""
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
    """Лёгкая нормализация русских окончаний без тяжёлой морфологии."""
    token = token.casefold().replace("ё", "е")
    suffixes = (
        "ениями", "аниями", "иями", "ями", "ами", "енной", "енного", "енные", "енный", "енная",
        "ского", "ская", "ский", "ских", "иями", "ого", "ему", "ому", "ыми", "ими",
        "ание", "ания", "ений", "ение", "ения", "ний", "ние", "ция", "ции", "ций",
        "ость", "ости", "остей", "ая", "яя", "ое", "ее", "ые", "ие", "ой", "ий", "ый",
        "ов", "ев", "ам", "ям", "ах", "ях", "ом", "ем", "у", "ю", "а", "я", "ы", "и", "е",
    )
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[:-len(suffix)]
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
    """Насколько совпадает содержательная лексика двух формулировок."""
    ta, tb = significant_tokens(a), significant_tokens(b)
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return 0.0
    jaccard = len(sa & sb) / len(sa | sb)
    sequence = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    # Jaccard важнее: одинаковые общие обороты не должны завышать оценку.
    return max(0.0, min(1.0, 0.70 * jaccard + 0.30 * sequence))


def _semantic_signal(raw_cosine: float) -> float:
    """Преобразует cosine Mistral Embed в вспомогательный сигнал 0..1.

    Raw cosine не показывается пользователю как процент совпадения. Основной
    вес остаётся у содержательной лексики, а embeddings помогают ловить
    близкие перефразировки.
    """
    value = max(-1.0, min(1.0, raw_cosine))
    return max(0.0, min(1.0, (value - 0.50) / 0.45))


def hybrid_similarity(a: str, b: str, embedding_cosine: float | None = None) -> float:
    if normalize_exact_title(a) and normalize_exact_title(a) == normalize_exact_title(b):
        return 100.0
    lexical = lexical_core_similarity(a, b)
    if embedding_cosine is None:
        return round(lexical * 100.0, 1)
    semantic = _semantic_signal(embedding_cosine)
    # Лексика — основной сигнал, embedding помогает ловить близкие перефразировки,
    # но уже не способен сам превратить две разные темы в «90% похожести».
    score = (0.72 * lexical + 0.28 * semantic) * 100.0
    return round(max(0.0, min(99.0, score)), 1)


class SimilarityService:
    def __init__(self, runtime_settings=None) -> None:
        self.mistral = MistralClient(runtime_settings)

    async def _embedding_map(self, texts: list[str]) -> tuple[dict[str, list[float]] | None, str]:
        unique = list(dict.fromkeys(texts))
        if not unique:
            return None, "none"
        if not self.mistral.available:
            return None, "hybrid-lexical"
        try:
            result: dict[str, list[float]] = {}
            for start in range(0, len(unique), 72):
                chunk = unique[start:start + 72]
                vectors = await self.mistral.embeddings(chunk)
                for text, vector in zip(chunk, vectors):
                    result[text] = vector
            if len(result) == len(unique):
                return result, "hybrid-mistral"
        except Exception:
            pass
        return None, "hybrid-lexical"

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
