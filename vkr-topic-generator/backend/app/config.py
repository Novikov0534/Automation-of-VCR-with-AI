from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VKR Topic Generator"
    database_url: str = "sqlite:///./vkr.db"
    cors_origins: str = "http://localhost:3000"

    mistral_api_key: str | None = None
    mistral_chat_model: str = "ministral-8b-2512"
    mistral_embedding_model: str = "mistral-embed"

    google_service_account_json: str | None = None
    google_share_with_email: str | None = None

    demo_mode: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
