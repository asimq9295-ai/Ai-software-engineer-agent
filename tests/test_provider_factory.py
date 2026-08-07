from __future__ import annotations

from ai_software_engineer_agent.domain.entities import AIProvider
from ai_software_engineer_agent.infrastructure.config import Settings
from ai_software_engineer_agent.infrastructure.ollama_gateway import OllamaLLMGateway
from ai_software_engineer_agent.infrastructure.provider_factory import build_llm_gateway


def test_provider_factory_builds_ollama_gateway_with_configured_model() -> None:
    settings = Settings(
        google_api_key="",
        gemini_model="gemini-2.5-flash",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        database_url="sqlite:///agent_runs.db",
    )

    gateway = build_llm_gateway(AIProvider.OLLAMA, settings)

    assert isinstance(gateway, OllamaLLMGateway)
    assert gateway.model == "llama3"


def test_provider_factory_allows_model_override() -> None:
    settings = Settings(
        google_api_key="",
        gemini_model="gemini-2.5-flash",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        database_url="sqlite:///agent_runs.db",
    )

    gateway = build_llm_gateway(AIProvider.OLLAMA, settings, model="mistral")

    assert gateway.model == "mistral"
