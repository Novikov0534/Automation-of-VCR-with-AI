from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

import httpx

from .mistral import MistralAPIError, MistralClient

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


@dataclass
class IntegrationCheckResult:
    status: str  # unconfigured | connected | error
    message: str
    detail: str | None = None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


async def check_mistral(
    api_key: str | None,
    *,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    live_probe: bool = False,
) -> IntegrationCheckResult:
    key = _clean(api_key)
    if not key:
        return IntegrationCheckResult("unconfigured", "не настроен", "API-ключ Mistral не указан")

    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        return IntegrationCheckResult("error", "неверный API-ключ", "Ключ Mistral должен содержать только допустимые ASCII-символы.")
    if any(ch.isspace() for ch in key):
        return IntegrationCheckResult("error", "неверный API-ключ", "В API-ключе Mistral не должно быть пробелов или переносов строки.")

    chat = _clean(chat_model) or "ministral-8b-2512"
    embed = _clean(embedding_model) or "mistral-embed"

    # Обычное чтение статуса не должно расходовать бесплатный RPS Mistral.
    # Полная сетевая проверка запускается только явной кнопкой пользователя.
    if not live_probe:
        return IntegrationCheckResult(
            "connected",
            "настроен",
            f"Ключ Mistral сохранён. Генерация: {chat}. "
            "Проверка дублей выполняется локально; для полного теста ключа нажмите «Проверить Mistral».",
        )

    runtime = SimpleNamespace(
        mistral_api_key=key,
        mistral_chat_model=chat,
        mistral_embedding_model=embed,
    )
    mistral = MistralClient(runtime)

    try:
        payload = await mistral.models()
        rows = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(rows, list):
            available = {str(row.get("id")) for row in rows if isinstance(row, dict) and row.get("id")}
            # latest-алиасы могут не присутствовать буквально в списке моделей.
            if chat not in available and not chat.endswith("-latest"):
                return IntegrationCheckResult("error", "модель не найдена", f"Mistral не нашёл модель генерации «{chat}».")
            if embed not in available and embed != "mistral-embed":
                return IntegrationCheckResult("error", "модель не найдена", f"Mistral не нашёл модель embeddings «{embed}».")
    except MistralAPIError as exc:
        if exc.status_code in {401, 403}:
            return IntegrationCheckResult("error", "ошибка авторизации", "Mistral API отклонил ключ. Проверьте API Key в Mistral Studio.")
        if exc.status_code == 429:
            return IntegrationCheckResult("error", "временное ограничение", str(exc))
        return IntegrationCheckResult("error", "ошибка подключения", f"Mistral API вернул HTTP {exc.status_code} при проверке ключа: {exc}")
    except Exception as exc:
        return IntegrationCheckResult("error", "ошибка подключения", f"Проверка Mistral завершилась ошибкой: {type(exc).__name__}: {exc}")

    if live_probe:
        # Лёгкая проверка: GET /models уже подтверждает валидность ключа и доступ к workspace.
        # Не делаем тестовый chat/completions и embeddings: на Free-тарифе такая «проверка»
        # сама расходует RPS/TPM и раньше могла показывать красный 429 при полностью рабочем ключе.
        mass_route = chat if chat.startswith("ministral-") else "ministral-8b-2512"
        return IntegrationCheckResult(
            "connected",
            "подключён",
            f"Ключ подтверждён Mistral API. Модель в настройках: {chat}. "
            f"Маршрут массовой генерации: {mass_route}. Реальный chat-запрос выполняется только при генерации тем. "
            "Проверка похожести выполняется локально, поэтому не зависит от лимитов Mistral embeddings.",
        )

    raise RuntimeError("Недостижимая ветвь проверки Mistral")


def _google_credentials(raw: str):
    from google.oauth2 import service_account

    value = raw.strip()
    if value.startswith("{"):
        info = json.loads(value)
        if info.get("type") != "service_account":
            raise ValueError("JSON должен содержать сервисный аккаунт Google (type=service_account)")
        return service_account.Credentials.from_service_account_info(info, scopes=GOOGLE_SCOPES)
    return service_account.Credentials.from_service_account_file(value, scopes=GOOGLE_SCOPES)


def _check_google_sync(raw: str, spreadsheet_id: str | None) -> IntegrationCheckResult:
    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError as exc:
        return IntegrationCheckResult("error", "модуль Google недоступен", f"Не установлены зависимости Google API: {exc}")

    try:
        credentials = _google_credentials(raw)
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
        return IntegrationCheckResult("error", "неверные данные аккаунта", str(exc))

    try:
        credentials.refresh(GoogleAuthRequest())
    except Exception as exc:
        return IntegrationCheckResult("error", "ошибка авторизации", f"Google не подтвердил сервисный аккаунт: {type(exc).__name__}: {exc}")

    try:
        sheets = build("sheets", "v4", credentials=credentials, cache_discovery=False)
        drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
        drive.about().get(fields="user(emailAddress)").execute()

        target = _clean(spreadsheet_id)
        if target:
            sheet = sheets.spreadsheets().get(
                spreadsheetId=target,
                fields="spreadsheetId,properties(title)",
            ).execute()
            drive_file = drive.files().get(
                fileId=target,
                fields="id,name,capabilities(canEdit)",
            ).execute()
            can_edit = bool((drive_file.get("capabilities") or {}).get("canEdit"))
            if not can_edit:
                return IntegrationCheckResult("error", "нет доступа к таблице", "Сервисный аккаунт видит таблицу, но не имеет права редактирования")
            title = ((sheet.get("properties") or {}).get("title") or drive_file.get("name") or "таблица")
            return IntegrationCheckResult("connected", "подключён", f"Google авторизация подтверждена. Таблица «{title}» доступна на запись")

        return IntegrationCheckResult(
            "connected",
            "подключён",
            "Google авторизация подтверждена. ID таблицы не задан — при публикации будет создана новая таблица",
        )
    except HttpError as exc:
        status = getattr(exc.resp, "status", None)
        text = str(exc)
        if status == 404:
            return IntegrationCheckResult("error", "таблица не найдена", "Проверьте ID Google-таблицы и доступ сервисного аккаунта")
        if status in {401, 403}:
            if spreadsheet_id:
                return IntegrationCheckResult("error", "нет доступа к таблице", "Проверьте права сервисного аккаунта и включённые Google API")
            return IntegrationCheckResult("error", "ошибка подключения", "Проверьте, что Google Drive/Sheets API включены для проекта")
        return IntegrationCheckResult("error", "ошибка подключения", f"Google API вернул HTTP {status or '?'}: {text}")
    except Exception as exc:
        return IntegrationCheckResult("error", "ошибка подключения", f"Google API: {type(exc).__name__}: {exc}")


async def check_google(service_account_json: str | None, spreadsheet_id: str | None = None) -> IntegrationCheckResult:
    raw = _clean(service_account_json)
    if not raw:
        return IntegrationCheckResult("unconfigured", "не настроен", "JSON сервисного аккаунта не указан")
    try:
        return await asyncio.wait_for(asyncio.to_thread(_check_google_sync, raw, spreadsheet_id), timeout=12.0)
    except TimeoutError:
        return IntegrationCheckResult("error", "нет связи", "Проверка Google API превысила 12 секунд")
