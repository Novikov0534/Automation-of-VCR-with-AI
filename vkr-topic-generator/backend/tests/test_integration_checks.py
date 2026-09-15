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
