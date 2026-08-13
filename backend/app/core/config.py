from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]


import urllib.parse
from pydantic import model_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "Kubernetes RAG Assistant"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb"
    JWT_SECRET: str = "supersecretkey"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    OPENAI_API_KEY: str = ""

    @model_validator(mode="after")
    def clean_database_url(self) -> "Settings":
        url = self.DATABASE_URL
        if url:
            prefix = ""
            for p in ["postgresql+asyncpg://", "postgresql://"]:
                if url.startswith(p):
                    prefix = p
                    url = url[len(p):]
                    break
            if prefix and "@" in url:
                creds, host_part = url.rsplit("@", 1)
                if ":" in creds:
                    user, password = creds.split(":", 1)
                    # Decode first to prevent double encoding, then encode safely
                    unquoted_password = urllib.parse.unquote(password)
                    encoded_password = urllib.parse.quote(unquoted_password, safe="")
                    self.DATABASE_URL = f"{prefix}{user}:{encoded_password}@{host_part}"
        return self


settings = Settings()
