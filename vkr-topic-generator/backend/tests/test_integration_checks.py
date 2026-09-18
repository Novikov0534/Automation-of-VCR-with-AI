import asyncio

from app.services.integration_checks import check_mistral


def test_mistral_non_ascii_key_returns_error_without_http_request():
    result = asyncio.run(check_mistral("ключ-mistral"))
    assert result.status == "error"
    assert result.message == "неверный API-ключ"
    assert result.detail


def test_mistral_key_with_spaces_returns_error():
    result = asyncio.run(check_mistral("mistral test"))
    assert result.status == "error"
    assert result.message == "неверный API-ключ"


def test_passive_mistral_status_does_not_spend_api_request(monkeypatch):
    from app.services import integration_checks as module

    async def unexpected_models(_self):
        raise AssertionError("passive status must not call Mistral")

    monkeypatch.setattr(module.MistralClient, "models", unexpected_models)
    result = asyncio.run(check_mistral("mistral-test-key"))
    assert result.status == "connected"
    assert result.message == "настроен"
    assert "Проверить Mistral" in result.detail


def test_live_mistral_check_validates_key_without_chat_probe(monkeypatch):
    from app.services import integration_checks as module

    async def fake_models(_self):
        return {"data": [
            {"id": "ministral-8b-2512"},
            {"id": "mistral-embed-2312"},
        ]}

    async def unexpected_chat(_self, **_kwargs):
        raise AssertionError("live check must not spend chat/completions quota")

    monkeypatch.setattr(module.MistralClient, "models", fake_models)
    monkeypatch.setattr(module.MistralClient, "chat_json", unexpected_chat)
    result = asyncio.run(check_mistral(
        "mistral-test-key",
        chat_model="ministral-8b-2512",
        embedding_model="mistral-embed-2312",
        live_probe=True,
    ))
    assert result.status == "connected"
    assert result.message == "подключён"
    assert "Реальный chat-запрос" in result.detail
