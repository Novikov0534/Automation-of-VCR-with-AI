from dataclasses import dataclass
from sqlalchemy.orm import Session

from .config import get_settings
from .models import AppSettings

DISABLED_SECRET = "__VKR_DISABLED__"


@dataclass
class RuntimeSettings:
    mistral_api_key: str | None
    mistral_chat_model: str
    mistral_embedding_model: str
    google_service_account_json: str | None
    google_share_with_email: str | None
    google_spreadsheet_id: str | None
    default_topic_count: int
    default_generation_focus: str | None
    similarity_green_max: float
    similarity_yellow_max: float
    similarity_calibrated: bool
    demo_mode: bool


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _effective_secret(db_value: str | None, env_value: str | None) -> str | None:
    value = _clean(db_value) if db_value is not None else None
    if value == DISABLED_SECRET:
        return None
    if value is not None:
        return value
    return _clean(env_value)


def load_runtime_settings(db: Session) -> RuntimeSettings:
    env = get_settings()
    row = db.get(AppSettings, 1)

    mistral = _effective_secret(row.mistral_api_key if row else None, env.mistral_api_key)
    google = _effective_secret(row.google_service_account_json if row else None, env.google_service_account_json)

    return RuntimeSettings(
        mistral_api_key=mistral,
        mistral_chat_model=_clean(row.mistral_chat_model) if row and row.mistral_chat_model else env.mistral_chat_model,
        mistral_embedding_model=_clean(row.mistral_embedding_model) if row and row.mistral_embedding_model else env.mistral_embedding_model,
        google_service_account_json=google,
        google_share_with_email=_clean(row.google_share_with_email) if row and row.google_share_with_email is not None else _clean(env.google_share_with_email),
        google_spreadsheet_id=_clean(row.google_spreadsheet_id) if row else None,
        default_topic_count=(row.default_topic_count if row and row.default_topic_count else 5),
        default_generation_focus=_clean(row.default_generation_focus) if row else None,
        similarity_green_max=45.0,
        similarity_yellow_max=70.0,
        similarity_calibrated=True,
        demo_mode=env.demo_mode,
    )


def secret_hint(value: str | None, *, label: str = "ключ") -> str | None:
    value = _clean(value)
    if not value:
        return None
    if len(value) <= 8:
        return f"{label} настроен"
    return f"••••{value[-4:]}"
