"""Runtime configuration — env-driven, no secrets in code."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Process-level settings.

    Every field has a safe default for local dev. Production values
    come from environment variables (or `/root/al-fardan/chatbot.env`
    on the VPS, loaded by docker-compose).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # HTTP server
    host: str = "0.0.0.0"
    port: int = 8001

    # Shared secret the CRM uses to authenticate calls to this service.
    # The chatbot requires it on /admin/* endpoints. /chat is public so
    # the widget can call it from the browser.
    chatbot_secret: str = "dev-secret-change-me"

    # Outbound secret for calls FROM the chatbot TO the CRM (e.g.
    # POST /api/chatbot/lead when the bot completes a signup flow).
    crm_base_url: str = "https://al-fardan-crm.vercel.app"
    crm_api_key: str = ""

    # Retrieval
    # Messages with a cosine similarity below this threshold are
    # treated as "the bot doesn't know" and get captured as an
    # unanswered query.
    confidence_threshold: float = 0.22
    top_k: int = 3

    # Data paths
    data_dir: Path = REPO_ROOT / "app" / "data"

    # CORS — comma-separated list of allowed origins for the widget
    cors_origins: str = (
        "https://al-fardan-crm.vercel.app,"
        "http://localhost:3000"
    )

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.pkl"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chat.db"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
