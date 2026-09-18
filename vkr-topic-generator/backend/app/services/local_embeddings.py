"""Локальные multilingual embeddings для проверки сходства тем.

Модель запускается на CPU через FastEmbed/ONNX и не использует Mistral API.
Если пакет или модель недоступны, вызывающий код получает ``None`` и может
перейти на локальный лексический режим. Модель загружается лениво при первой
проверке; кеш FastEmbed следует хранить в docker volume.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Iterable

DEFAULT_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DEFAULT_CACHE_DIR = os.getenv("LOCAL_EMBEDDING_CACHE_DIR", "/models/fastembed")


class LocalEmbeddingProvider:
    _model = None
    _model_name: str | None = None
    _load_error: str | None = None
    _lock = threading.RLock()
    _cache: "OrderedDict[str, list[float]]" = OrderedDict()
    _cache_limit = 4096

    @classmethod
    def enabled(cls) -> bool:
        raw = os.getenv("LOCAL_EMBEDDING_ENABLED", "true").strip().casefold()
        return raw not in {"0", "false", "no", "off"}

    @classmethod
    def _ensure_model(cls):
        if not cls.enabled():
            return None
        with cls._lock:
            if cls._model is not None:
                return cls._model
            # После ошибки импорта не падаем вместе с приложением. При следующем
            # рестарте backend будет новая попытка (например, после docker rebuild).
            if cls._load_error is not None:
                return None
            try:
                from fastembed import TextEmbedding  # type: ignore

                threads_raw = os.getenv("LOCAL_EMBEDDING_THREADS", "")
                threads = int(threads_raw) if threads_raw.strip().isdigit() else None
                cls._model = TextEmbedding(
                    model_name=DEFAULT_MODEL,
                    cache_dir=DEFAULT_CACHE_DIR,
                    threads=threads,
                    lazy_load=True,
                )
                cls._model_name = DEFAULT_MODEL
                return cls._model
            except Exception as exc:  # pragma: no cover - зависит от окружения/скачивания модели
                cls._load_error = f"{type(exc).__name__}: {exc}"
                return None

    @classmethod
    def ready(cls) -> bool:
        """Модель уже загружена в текущем процессе.

        Используется профильным scorer-ом, чтобы не инициировать отдельную
        тяжёлую загрузку модели только ради рендера страницы.
        """
        return cls._model is not None

    @classmethod
    def model_name(cls) -> str | None:
        return cls._model_name or (DEFAULT_MODEL if cls.enabled() else None)

    @classmethod
    def last_error(cls) -> str | None:
        return cls._load_error

    @classmethod
    def _remember(cls, text: str, vector: list[float]) -> None:
        cls._cache[text] = vector
        cls._cache.move_to_end(text)
        while len(cls._cache) > cls._cache_limit:
            cls._cache.popitem(last=False)

    @classmethod
    def embed_map(cls, texts: Iterable[str]) -> dict[str, list[float]] | None:
        unique = [text for text in dict.fromkeys(texts) if text and text.strip()]
        if not unique:
            return {}
        model = cls._ensure_model()
        if model is None:
            return None

        with cls._lock:
            missing = [text for text in unique if text not in cls._cache]
            if missing:
                try:
                    vectors = list(model.embed(missing, batch_size=min(128, max(1, len(missing)))))
                    if len(vectors) != len(missing):
                        raise RuntimeError("локальная embedding-модель вернула неполный batch")
                    for text, vector in zip(missing, vectors):
                        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
                        cls._remember(text, [float(x) for x in values])
                except Exception as exc:  # pragma: no cover - зависит от ONNX/runtime
                    cls._load_error = f"{type(exc).__name__}: {exc}"
                    return None
            return {text: cls._cache[text] for text in unique if text in cls._cache}
