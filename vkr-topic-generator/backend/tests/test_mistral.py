import asyncio
import json
from types import SimpleNamespace

from app.services.mistral import MistralAPIError, MistralClient


class FakeResponse:
    def __init__(self, payload, status_code=200, text=None, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload, ensure_ascii=False)
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeAsyncClient:
    calls = []
    responses = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.__class__.calls.append((url, kwargs))
        payload = self.__class__.responses.pop(0)
        if isinstance(payload, FakeResponse):
            return payload
        return FakeResponse(payload)


def runtime():
    return SimpleNamespace(
        mistral_api_key="mistral-test-key",
        mistral_chat_model="mistral-small-latest",
        mistral_embedding_model="mistral-embed",
        mistral_min_request_interval_seconds=0.0,
    )


def test_mistral_chat_uses_structured_json(monkeypatch):
    from app.services import mistral as module

    FakeAsyncClient.calls = []
    payload_text = json.dumps({"topics": [{"title": "Тема", "rationale": "Описание", "keywords": ["ИИ"]}]}, ensure_ascii=False)
    FakeAsyncClient.responses = [{
        "choices": [{"message": {"content": payload_text}, "finish_reason": "stop"}],
    }]
    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(MistralClient(runtime()).chat_json(system="system", user="user", temperature=0.4))
    assert result["topics"][0]["title"] == "Тема"
    url, kwargs = FakeAsyncClient.calls[0]
    assert url.endswith("/chat/completions")
    assert kwargs["headers"]["Authorization"] == "Bearer mistral-test-key"
    assert kwargs["json"]["model"] == "mistral-small-latest"
    assert kwargs["json"]["messages"][0] == {"role": "system", "content": "system"}
    assert kwargs["json"]["response_format"]["type"] == "json_schema"
    assert kwargs["json"]["response_format"]["json_schema"]["schema"]["required"] == ["topics"]
    assert kwargs["json"]["response_format"]["json_schema"]["strict"] is True


def test_mistral_http_error_preserves_message(monkeypatch):
    from app.services import mistral as module

    FakeAsyncClient.calls = []
    FakeAsyncClient.responses = [FakeResponse({"detail": "Bad request payload"}, status_code=400)]
    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)

    try:
        asyncio.run(MistralClient(runtime()).chat_json(system="system", user="user"))
        assert False, "MistralAPIError expected"
    except MistralAPIError as exc:
        assert exc.status_code == 400
        assert "Bad request payload" in str(exc)


def test_mistral_embeddings(monkeypatch):
    from app.services import mistral as module

    FakeAsyncClient.calls = []
    FakeAsyncClient.responses = [{"data": [
        {"index": 0, "embedding": [1.0, 0.0]},
        {"index": 1, "embedding": [0.5, 0.5]},
    ]}]
    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)

    vectors = asyncio.run(MistralClient(runtime()).embeddings(["первая тема", "вторая тема"]))
    assert vectors == [[1.0, 0.0], [0.5, 0.5]]
    url, kwargs = FakeAsyncClient.calls[0]
    assert url.endswith("/embeddings")
    assert kwargs["json"]["model"] == "mistral-embed"
    assert kwargs["json"]["input"] == ["первая тема", "вторая тема"]


def test_mistral_retries_429(monkeypatch):
    from app.services import mistral as module

    FakeAsyncClient.calls = []
    payload_text = json.dumps({"topics": [{"title": "Новая тема", "rationale": "Описание", "keywords": ["ИИ"]}]}, ensure_ascii=False)
    FakeAsyncClient.responses = [
        FakeResponse({"detail": "rate limit"}, status_code=429, headers={"retry-after": "0.01"}),
        {"choices": [{"message": {"content": payload_text}, "finish_reason": "stop"}]},
    ]
    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)

    sleeps = []

    async def no_sleep(delay):
        sleeps.append(delay)
    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)

    result = asyncio.run(MistralClient(runtime()).chat_json(system="system", user="user"))
    assert result["topics"][0]["title"] == "Новая тема"
    assert len(FakeAsyncClient.calls) == 2
    assert sleeps == [2.0]


def test_mistral_uses_2_4_8_backoff_and_clear_429_error(monkeypatch):
    from app.services import mistral as module

    FakeAsyncClient.calls = []
    FakeAsyncClient.responses = [
        FakeResponse({"detail": "rate limit"}, status_code=429),
        FakeResponse({"detail": "rate limit"}, status_code=429),
        FakeResponse({"detail": "rate limit"}, status_code=429),
        FakeResponse({"detail": "rate limit"}, status_code=429),
    ]
    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)
    sleeps = []

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)

    try:
        asyncio.run(MistralClient(runtime()).chat_json(system="system", user="user"))
        assert False, "MistralAPIError expected"
    except MistralAPIError as exc:
        assert exc.status_code == 429
        assert "ограничения бесплатного тарифа" in str(exc)

    assert len(FakeAsyncClient.calls) == 4
    assert sleeps == [2.0, 4.0, 8.0]


def test_mistral_serializes_chat_and_embeddings_across_clients(monkeypatch):
    from app.services import mistral as module

    real_sleep = asyncio.sleep

    class ConcurrentFakeClient:
        active = 0
        max_active = 0

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            self.__class__.active += 1
            self.__class__.max_active = max(self.__class__.max_active, self.__class__.active)
            await real_sleep(0.01)
            self.__class__.active -= 1
            if url.endswith("/embeddings"):
                return FakeResponse({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
            return FakeResponse({
                "choices": [{
                    "message": {"content": json.dumps({"topics": []})},
                    "finish_reason": "stop",
                }]
            })

    monkeypatch.setattr(module.httpx, "AsyncClient", ConcurrentFakeClient)

    async def run_concurrently():
        first = MistralClient(runtime())
        second = MistralClient(runtime())
        return await asyncio.gather(
            first.chat_json(system="system", user="user"),
            second.embeddings(["тема"]),
        )

    chat, vectors = asyncio.run(run_concurrently())
    assert chat == {"topics": []}
    assert vectors == [[1.0, 0.0]]
    assert ConcurrentFakeClient.max_active == 1


def test_mistral_keeps_minimum_interval_across_clients(monkeypatch):
    from app.services import mistral as module

    FakeAsyncClient.calls = []
    payload = {"choices": [{"message": {"content": json.dumps({"topics": []})}, "finish_reason": "stop"}]}
    FakeAsyncClient.responses = [payload, payload]
    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)

    clock = [100.0]
    sleeps = []

    def monotonic():
        return clock[0]

    async def advance(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(module.time, "monotonic", monotonic)
    monkeypatch.setattr(module.asyncio, "sleep", advance)
    limited_runtime = SimpleNamespace(
        mistral_api_key="mistral-test-key",
        mistral_chat_model="mistral-small-latest",
        mistral_embedding_model="mistral-embed",
        mistral_min_request_interval_seconds=1.25,
    )

    async def run_sequentially():
        await MistralClient(limited_runtime).chat_json(system="system", user="first")
        await MistralClient(limited_runtime).chat_json(system="system", user="second")

    asyncio.run(run_sequentially())
    assert sleeps == [1.25]
    assert len(FakeAsyncClient.calls) == 2


def test_topic_generator_batches_multiple_teachers(monkeypatch):
    from app.services.generator import TopicGenerator

    teachers = [
        SimpleNamespace(id=1, full_name="Иванов Иван Иванович", department="САПРиПК", position="доцент", research_areas=["Компьютерное зрение"], past_topics=[], topics=[]),
        SimpleNamespace(id=2, full_name="Петров Пётр Петрович", department="ЭВМиС", position="профессор", research_areas=["Информационные системы"], past_topics=[], topics=[]),
    ]
    generator = TopicGenerator(runtime())
    calls = []

    async def fake_batch(**kwargs):
        calls.append(kwargs)
        return {"teachers": [
            {"teacher_id": 1, "topics": [{
                "title": "Разработка системы видеоаналитики производственной линии с обнаружением дефектов и визуализацией результатов контроля",
                "rationale": "Backend, обработка видео, интерфейс и тестирование.",
                "keywords": ["видео", "дефекты"],
            }]},
            {"teacher_id": 2, "topics": [{
                "title": "Разработка веб-платформы управления учебными проектами с контролем сроков, уведомлениями и аналитикой выполнения задач",
                "rationale": "API, БД, интерфейс, уведомления и аналитика.",
                "keywords": ["проекты", "аналитика"],
            }]},
        ]}

    monkeypatch.setattr(generator.mistral, "chat_json", fake_batch)
    result, mode, warnings = asyncio.run(generator.generate_batch([(teachers[0], 1), (teachers[1], 1)]))
    assert len(calls) == 1
    assert set(result) == {1, 2}
    assert len(result[1]) == 1 and len(result[2]) == 1
    assert mode == "mistral-batch"
    assert warnings == []
    assert calls[0]["schema"]["required"] == ["teachers"]
