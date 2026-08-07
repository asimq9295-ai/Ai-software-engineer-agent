from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_num_predict: int = 64
    ollama_read_timeout: int = 900
    database_url: str = "sqlite:///agent_runs.db"

    @property
    def sqlite_path(self) -> str:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("Only sqlite:/// DATABASE_URL values are supported.")
        return self.database_url.removeprefix(prefix)


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    load_dotenv(project_root / "ai_software_engineer_agent" / ".env", override=False)
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "64")),
        ollama_read_timeout=int(os.getenv("OLLAMA_READ_TIMEOUT", "900")),
        database_url=os.getenv("DATABASE_URL", "sqlite:///agent_runs.db"),
    )
