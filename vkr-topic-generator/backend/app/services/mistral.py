import asyncio
import json
import time
import weakref
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from ..config import get_settings


TOPICS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "rationale", "keywords"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["topics"],
    "additionalProperties": False,
}

BATCH_TOPICS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "teachers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "teacher_id": {"type": "integer"},
                    "topics": TOPICS_RESPONSE_SCHEMA["properties"]["topics"],
                },
                "required": ["teacher_id", "topics"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["teachers"],
    "additionalProperties": False,
}


class MistralAPIError(RuntimeError):
    """Понятная ошибка Mistral API с HTTP-кодом и сообщением сервиса."""

    def __init__(self, status_code: int, message: str, *, endpoint: str | None = None, model: str | None = None):
        self.status_code = status_code
        self.endpoint = endpoint
        self.model = model
        super().__init__(message)


@dataclass
class _RateLimitState:
    """Общая очередь Mistral для одного event loop приложения."""

    lock: asyncio.Lock
    last_request_started_at: float | None = None


_RATE_LIMIT_STATES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _rate_limit_state() -> _RateLimitState:
    loop = asyncio.get_running_loop()
    state = _RATE_LIMIT_STATES.get(loop)
    if state is None:
        state = _RateLimitState(lock=asyncio.Lock())
        _RATE_LIMIT_STATES[loop] = state
    return state


def _mistral_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
            if isinstance(detail, list):
                return json.dumps(detail, ensure_ascii=False)[:1200]
            message = payload.get("message") or payload.get("error")
            if isinstance(message, str) and message.strip():
                return message.strip()
            if isinstance(message, dict):
                nested = message.get("message") or message.get("detail")
                if nested:
                    return str(nested).strip()
    except Exception:
        pass
    text = (getattr(response, "text", "") or "").strip()
    return text[:1200] if text else f"HTTP {response.status_code}"


def _extract_message_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    chunks.append(str(text))
            elif item is not None:
                chunks.append(str(item))
        return "".join(chunks).strip()
    return ""


def _retry_after_seconds(response: httpx.Response, attempt: int) -> float:
    # Четыре попытки дают три паузы: 2, 4 и 8 секунд.
    fallback = min(8.0, float(2 ** (attempt + 1)))
    try:
        raw = (response.headers or {}).get("retry-after")
    except Exception:
        raw = None
    if raw:
        try:
            return max(fallback, min(60.0, float(raw)))
        except (TypeError, ValueError):
            try:
                dt = parsedate_to_datetime(raw)
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                return max(fallback, min(60.0, (dt - now).total_seconds()))
            except Exception:
                pass
    return fallback


class MistralClient:
    """Прямой REST-клиент Mistral API без SDK.

    Используется один внешний AI-провайдер: Mistral. Генерация идёт через
    /v1/chat/completions со строгим JSON Schema, проверка сходства — через
    /v1/embeddings (mistral-embed). Все экземпляры клиента используют одну
    последовательную очередь с паузой между запросами. При 429/5xx выполняется
    автоматический retry 2 -> 4 -> 8 секунд вместо немедленного падения набора.
    """

    MIN_REQUEST_INTERVAL_SECONDS = 1.25
    MAX_ATTEMPTS = 4

    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()
        self.base_url = "https://api.mistral.ai/v1"

    @property
    def available(self) -> bool:
        return bool(getattr(self.settings, "mistral_api_key", None))

    @property
    def chat_model(self) -> str:
        return str(getattr(self.settings, "mistral_chat_model", "mistral-small-latest") or "mistral-small-latest")

    @property
    def embedding_model(self) -> str:
        return str(getattr(self.settings, "mistral_embedding_model", "mistral-embed") or "mistral-embed")

    @property
    def min_request_interval_seconds(self) -> float:
        value = getattr(
            self.settings,
            "mistral_min_request_interval_seconds",
            self.MIN_REQUEST_INTERVAL_SECONDS,
        )
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return self.MIN_REQUEST_INTERVAL_SECONDS

    def _headers(self) -> dict[str, str]:
        key = getattr(self.settings, "mistral_api_key", None)
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is not configured")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _raise_api_error(response: httpx.Response, endpoint: str, model: str | None = None) -> None:
        if response.status_code < 400:
            return
        message = _mistral_error_message(response)
        if response.status_code == 429:
            message = (
                "Mistral API временно недоступен из-за ограничения бесплатного тарифа. "
                "Автоматические повторы через 2, 4 и 8 секунд не помогли; "
                "повторите операцию через несколько секунд."
            )
        raise MistralAPIError(
            response.status_code,
            message,
            endpoint=endpoint,
            model=model,
        )

    async def _wait_for_request_slot(self, state: _RateLimitState) -> None:
        now = time.monotonic()
        if state.last_request_started_at is not None:
            remaining = self.min_request_interval_seconds - (now - state.last_request_started_at)
            if remaining > 0:
                await asyncio.sleep(remaining)
        state.last_request_started_at = time.monotonic()

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        *,
        payload: dict[str, Any] | None = None,
        model: str | None = None,
        timeout: float = 180.0,
    ) -> httpx.Response:
        last_response: httpx.Response | None = None
        last_transport: Exception | None = None
        state = _rate_limit_state()

        # Lock удерживается на протяжении всех попыток одного логического вызова:
        # chat, embeddings и проверка моделей никогда не идут параллельно.
        async with state.lock:
            async with httpx.AsyncClient(timeout=timeout) as client:
                for attempt in range(self.MAX_ATTEMPTS):
                    await self._wait_for_request_slot(state)
                    try:
                        if method == "GET":
                            response = await client.get(endpoint, headers=self._headers())
                        else:
                            response = await client.post(endpoint, headers=self._headers(), json=payload)
                        last_response = response
                        if response.status_code < 400:
                            return response
                        if (
                            response.status_code in {408, 409, 429, 500, 502, 503, 504}
                            and attempt < self.MAX_ATTEMPTS - 1
                        ):
                            await asyncio.sleep(_retry_after_seconds(response, attempt))
                            continue
                        self._raise_api_error(response, endpoint, model)
                    except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:
                        last_transport = exc
                        if attempt < self.MAX_ATTEMPTS - 1:
                            await asyncio.sleep(min(8.0, float(2 ** (attempt + 1))))
                            continue
                        raise RuntimeError(f"Mistral API недоступен: {type(exc).__name__}") from exc

        if last_response is not None:
            self._raise_api_error(last_response, endpoint, model)
        if last_transport is not None:
            raise RuntimeError(f"Mistral API недоступен: {type(last_transport).__name__}") from last_transport
        raise RuntimeError("Mistral API не вернул ответ")

    async def _post_with_retry(self, endpoint: str, payload: dict[str, Any], *, model: str | None = None, timeout: float = 180.0) -> httpx.Response:
        return await self._request_with_retry(
            "POST",
            endpoint,
            payload=payload,
            model=model,
            timeout=timeout,
        )

    async def models(self) -> dict[str, Any]:
        response = await self._request_with_retry("GET", f"{self.base_url}/models", timeout=30.0)
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Mistral Models вернул неожиданный формат ответа")
        return data

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.65,
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 12000,
    ) -> dict[str, Any]:
        selected_model = (model or self.chat_model).strip()
        response_schema = schema or TOPICS_RESPONSE_SCHEMA
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "vkr_topics",
                    "schema": response_schema,
                    "strict": True,
                },
            },
        }
        endpoint = f"{self.base_url}/chat/completions"
        response = await self._post_with_retry(endpoint, payload, model=selected_model)
        data = response.json()
        content = _extract_message_text(data)
        if not content:
            finish_reason = None
            try:
                finish_reason = data.get("choices", [{}])[0].get("finish_reason")
            except Exception:
                pass
            raise RuntimeError(f"Mistral вернул ответ без текста (finish_reason={finish_reason or 'unknown'})")
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Mistral вернул некорректный JSON: {exc.msg}") from exc

    async def embeddings(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        selected_model = (model or self.embedding_model).strip()
        payload = {
            "model": selected_model,
            "input": texts,
            "encoding_format": "float",
        }
        endpoint = f"{self.base_url}/embeddings"
        response = await self._post_with_retry(endpoint, payload, model=selected_model, timeout=120.0)
        data = response.json()
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("Mistral Embeddings вернул неожиданный формат ответа")
        ordered = sorted((row for row in rows if isinstance(row, dict)), key=lambda row: int(row.get("index", 0)))
        vectors = [row.get("embedding") for row in ordered]
        if len(vectors) != len(texts) or any(not isinstance(vector, list) or not vector for vector in vectors):
            raise RuntimeError("Mistral Embeddings вернул неполный набор векторов")
        return [[float(value) for value in vector] for vector in vectors]
