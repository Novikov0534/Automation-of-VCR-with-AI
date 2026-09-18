import asyncio
import json
from types import SimpleNamespace

import pytest

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
        # primary model: 4 attempts
        FakeResponse({"detail": "rate limit"}, status_code=429),
        FakeResponse({"detail": "rate limit"}, status_code=429),
        FakeResponse({"detail": "rate limit"}, status_code=429),
        FakeResponse({"detail": "rate limit"}, status_code=429),
        # free fallback model: 4 attempts
        FakeResponse({"detail": "fallback rate limit"}, status_code=429),
        FakeResponse({"detail": "fallback rate limit"}, status_code=429),
        FakeResponse({"detail": "fallback rate limit"}, status_code=429),
        FakeResponse({"detail": "fallback rate limit"}, status_code=429),
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
        assert "HTTP 429" in str(exc)
        assert "fallback rate limit" in str(exc)

    assert len(FakeAsyncClient.calls) == 8
    assert sleeps == [2.0, 4.0, 8.0, 2.0, 4.0, 8.0]



def test_mistral_falls_back_to_ministral_8b_after_primary_429(monkeypatch):
    from app.services import mistral as module

    FakeAsyncClient.calls = []
    payload_text = json.dumps({"topics": [{"title": "Резервная тема", "rationale": "Описание", "keywords": ["ИИ"]}]}, ensure_ascii=False)
    FakeAsyncClient.responses = [
        FakeResponse({"detail": "rate limit"}, status_code=429),
        FakeResponse({"detail": "rate limit"}, status_code=429),
        FakeResponse({"detail": "rate limit"}, status_code=429),
        FakeResponse({"detail": "rate limit"}, status_code=429),
        {"choices": [{"message": {"content": payload_text}, "finish_reason": "stop"}]},
    ]
    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    result = asyncio.run(MistralClient(runtime()).chat_json(system="system", user="user"))
    assert result["topics"][0]["title"] == "Резервная тема"
    assert FakeAsyncClient.calls[-1][1]["json"]["model"] == "ministral-8b-2512"


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
    assert mode == "mistral-ai"
    assert result[1][0].generation_source == "mistral-ai"
    assert result[2][0].generation_source == "mistral-ai"
    assert warnings == []
    assert calls[0]["schema"]["required"] == ["teachers"]


def test_topic_generator_requires_mistral_key_for_ai_generation():
    from app.services.generator import TopicGenerator

    teacher = SimpleNamespace(
        id=1,
        full_name="Иванов Иван Иванович",
        department="САПРиПК",
        position="доцент",
        research_areas=["Компьютерное зрение"],
        past_topics=[],
        topics=[],
    )
    cfg = runtime()
    cfg.mistral_api_key = None
    generator = TopicGenerator(cfg)

    with pytest.raises(RuntimeError, match="Демо без ИИ"):
        asyncio.run(generator.generate_batch([(teacher, 5)]))

    local, mode, warnings = generator.generate_local_batch([(teacher, 5)])
    assert mode == "local-demo"
    assert warnings == []
    assert len(local[1]) == 5
    assert all(item.generation_source == "local-demo" for item in local[1])


def test_topic_generator_429_does_not_substitute_local_bank(monkeypatch):
    from app.services.generator import TopicGenerator

    teacher = SimpleNamespace(
        id=1,
        full_name="Петров Пётр Петрович",
        department="САПРиПК",
        position="доцент",
        research_areas=["Информационные системы"],
        past_topics=[],
        topics=[],
    )
    generator = TopicGenerator(runtime())

    async def fail_with_429(**kwargs):
        raise MistralAPIError(429, "rate limit")

    monkeypatch.setattr(generator.mistral, "chat_json", fail_with_429)
    with pytest.raises(RuntimeError, match="429"):
        asyncio.run(generator.generate_batch([(teacher, 6)]))


def test_topic_generator_timeout_does_not_substitute_local_bank(monkeypatch):
    from app.services.generator import TopicGenerator

    teacher = SimpleNamespace(
        id=1,
        full_name="Сидоров Сергей Сергеевич",
        department="САПРиПК",
        position="доцент",
        research_areas=["Информационные системы"],
        past_topics=[],
        topics=[],
    )
    generator = TopicGenerator(runtime())
    generator.BULK_AI_TIMEOUT_SECONDS = 0.01
    calls = [0]

    async def very_slow(**kwargs):
        calls[0] += 1
        await asyncio.sleep(1)
        return {"teachers": []}

    monkeypatch.setattr(generator.mistral, "chat_json", very_slow)
    with pytest.raises(RuntimeError, match="Локальные темы не подставлялись"):
        asyncio.run(generator.generate_batch([(teacher, 4)]))
    # v30 quality-first: batch по умолчанию не смешивает 8B и 3B;
    # вторая модель разрешается только MISTRAL_ALLOW_SMALL_FALLBACK=true.
    assert calls[0] == 1



def test_topic_generator_repairs_missing_ai_topics(monkeypatch):
    """Если 1 тема не прошла фильтр, генератор должен добрать её Mistral, а не падать 4/5."""
    from app.services.generator import TopicGenerator

    teacher = SimpleNamespace(
        id=84,
        full_name="Скоробогатченко Дмитрий Анатольевич",
        department="САПРиПК",
        position="профессор",
        research_areas=["Системный анализ", "Системы искусственного интеллекта"],
        past_topics=[
            SimpleNamespace(title="Многоагентная система универсальной автоматизации и адаптивного управления бизнес-процессами университета"),
            SimpleNamespace(title="Многоагентная система накопления и систематизации результатов студенческих практик для обеспечения преемственности научных разработок"),
        ],
        topics=[],
    )
    generator = TopicGenerator(runtime())
    calls = []

    first_topics = [
        {
            "title": "Разработка платформы интеллектуального мониторинга бизнес-процессов университета с выявлением отклонений и рекомендациями по корректировке",
            "rationale": "API, БД, аналитика, интерфейс и оценка качества.",
            "keywords": ["бизнес-процессы", "ИИ"],
        },
        {
            "title": "Разработка системы прогнозирования рисков срыва учебных проектов с анализом истории задач и визуализацией факторов риска",
            "rationale": "Сбор данных, модель, API, интерфейс и тестирование.",
            "keywords": ["риски", "прогнозирование"],
        },
        {
            "title": "Разработка программного комплекса управления портфелем университетских проектов с приоритизацией задач и аналитикой загрузки команд",
            "rationale": "Backend, БД, оптимизация, UI и аналитика.",
            "keywords": ["проекты", "оптимизация"],
        },
        {
            "title": "Разработка системы автоматизации согласования внутренних заявок университета с маршрутизацией, контролем сроков и анализом узких мест",
            "rationale": "Workflow, API, БД, UI и аналитика.",
            "keywords": ["workflow", "автоматизация"],
        },
        # Намеренно слишком короткая/общая — должна быть отклонена валидатором.
        {
            "title": "Разработка системы анализа данных",
            "rationale": "Описание",
            "keywords": ["данные"],
        },
    ]
    repaired_topic = {
        "title": "Разработка системы поддержки принятия решений по распределению ресурсов кафедры с многокритериальной оценкой и сценарным анализом",
        "rationale": "Модель критериев, API, БД, интерфейс и проверка сценариев.",
        "keywords": ["СППР", "ресурсы", "сценарии"],
    }

    async def fake_batch(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"teachers": [{"teacher_id": 84, "topics": first_topics}]}
        return {"teachers": [{"teacher_id": 84, "topics": [repaired_topic]}]}

    monkeypatch.setattr(generator.mistral, "chat_json", fake_batch)
    result, mode, warnings = asyncio.run(generator.generate_batch([(teacher, 5)]))

    assert mode == "mistral-ai"
    assert len(result[84]) == 5
    assert result[84][-1].title == repaired_topic["title"]
    assert all(item.generation_source == "mistral-ai" for item in result[84])
    assert len(calls) == 2
    assert any("автоматически добрал" in warning for warning in warnings)


def test_topic_generator_does_not_force_unsafe_reserve_to_fill_five_of_five(monkeypatch):
    """v33: слабые/дублирующие AI-кандидаты не используются только ради комплекта 5/5."""
    from app.services.generator import TopicGenerator

    teacher = SimpleNamespace(
        id=50,
        full_name="Кравец Алла Григорьевна",
        department="САПРиПК",
        position="профессор",
        research_areas=["Системы искусственного интеллекта"],
        past_topics=[
            SimpleNamespace(title="Разработка интеллектуального ассистента для AI-архитектора"),
            SimpleNamespace(title="Разработка Телеграм-бота для анализа каналов технологических новостей"),
        ],
        topics=[],
    )
    generator = TopicGenerator(runtime())
    calls = []

    topics = [
        {
            "title": "Разработка интеллектуальной системы поддержки принятия решений для выбора архитектуры программного проекта на основе требований",
            "rationale": "ИИ-анализ требований, ранжирование архитектурных вариантов и интерфейс рекомендаций.",
            "keywords": ["ИИ", "архитектура", "СППР"],
        },
        {
            "title": "Разработка сервиса семантического поиска по корпоративной документации с ранжированием результатов и генерацией кратких ответов",
            "rationale": "Эмбеддинги, поиск, генерация ответа и веб-интерфейс.",
            "keywords": ["RAG", "поиск", "LLM"],
        },
        {
            "title": "Разработка интеллектуальной системы классификации обращений пользователей с маршрутизацией запросов и объяснением решения модели",
            "rationale": "Классификация текста, маршрутизация и объяснимость модели.",
            "keywords": ["NLP", "классификация", "ИИ"],
        },
        {
            "title": "Разработка системы автоматического анализа технических требований с выявлением противоречий и формированием рекомендаций по уточнению",
            "rationale": "NLP-анализ требований, поиск конфликтов и формирование рекомендаций.",
            "keywords": ["NLP", "требования", "ИИ"],
        },
        # Пограничная, но технически валидная AI-тема: локальный Quality Gate
        # может посчитать её междисциплинарной. Она должна попасть в резерв,
        # а после неудачных repair — быть возвращена как 5-я тема с warning.
        {
            "title": "Разработка веб-платформы анализа обращений пользователей с автоматической классификацией запросов и формированием аналитических отчётов",
            "rationale": "Классификация обращений, аналитический backend и визуализация результатов.",
            "keywords": ["классификация", "аналитика", "ИИ"],
        },
    ]

    async def fake_batch(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"teachers": [{"teacher_id": 50, "topics": topics}]}
        # Оба repair ничего полезнее не нашли.
        return {"teachers": [{"teacher_id": 50, "topics": []}]}

    monkeypatch.setattr(generator.mistral, "chat_json", fake_batch)
    result, mode, warnings = asyncio.run(generator.generate_batch([(teacher, 5)]))

    assert mode == "mistral-ai"
    assert 1 <= len(result[50]) < 5
    assert all(item.generation_source == "mistral-ai" for item in result[50])
    assert any("Quality Gate не стал заполнять" in warning for warning in warnings)
    assert len(calls) == 3


def test_v33_title_validator_requires_clear_software_project_goal():
    from app.services.generator import TopicGenerator

    assert TopicGenerator._specific_enough(
        "Разработка веб-сервиса мониторинга состояния оборудования с выявлением аномалий и формированием уведомлений"
    )
    assert not TopicGenerator._specific_enough("Разработка интеллектуальной системы анализа данных")
    assert not TopicGenerator._specific_enough("Исследование методов машинного обучения для промышленности")


def test_v34_compact_titles_response_is_normalized(monkeypatch):
    """V34 принимает быстрый titles-only structured response и сохраняет темы как Mistral AI."""
    from app.services.generator import TopicGenerator

    teacher = SimpleNamespace(
        id=901,
        full_name="Тестов Тест Тестович",
        department="САПРиПК",
        position="доцент",
        research_areas=[],
        past_topics=[],
        topics=[],
    )
    generator = TopicGenerator(runtime())
    calls = []

    async def fake_titles_only(**kwargs):
        calls.append(kwargs)
        return {
            "teachers": [{
                "teacher_id": 901,
                "titles": [
                    "Разработка веб-сервиса мониторинга доступности сетевых узлов с журналированием событий и уведомлением администратора",
                    "Разработка информационной системы управления учебными проектами с контролем сроков, ролями пользователей и формированием отчётности",
                    "Разработка приложения анализа журналов информационных систем с поиском аномалий, фильтрацией событий и визуализацией статистики",
                    "Разработка платформы каталогизации открытых наборов данных с полнотекстовым поиском, тегированием и контролем качества метаданных",
                ],
            }]
        }

    monkeypatch.setattr(generator.mistral, "chat_json", fake_titles_only)
    result, mode, _warnings = asyncio.run(generator.generate_batch([(teacher, 3)]))

    assert mode == "mistral-ai"
    assert len(result[901]) == 3
    assert all(item.generation_source == "mistral-ai" for item in result[901])
    assert calls
    schema = calls[0]["schema"]
    teacher_properties = schema["properties"]["teachers"]["items"]["properties"]
    assert "titles" in teacher_properties
    assert "topics" not in teacher_properties
    assert calls[0]["max_tokens"] < 1000
