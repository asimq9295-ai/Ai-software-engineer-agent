from __future__ import annotations

from ai_software_engineer_agent.domain.entities import AIProvider
from ai_software_engineer_agent.domain.ports import LLMGateway
from ai_software_engineer_agent.infrastructure.config import Settings
from ai_software_engineer_agent.infrastructure.gemini_gateway import GeminiLLMGateway
from ai_software_engineer_agent.infrastructure.ollama_gateway import OllamaLLMGateway


def build_llm_gateway(
    provider: AIProvider,
    settings: Settings,
    model: str | None = None,
) -> LLMGateway:
    if provider == AIProvider.GEMINI:
        return GeminiLLMGateway(
            api_key=settings.google_api_key,
            model=model or settings.gemini_model,
        )
    if provider == AIProvider.OLLAMA:
        return OllamaLLMGateway(
            base_url=settings.ollama_base_url,
            model=model or settings.ollama_model,
            num_predict=settings.ollama_num_predict,
            read_timeout=settings.ollama_read_timeout,
        )
    raise ValueError(f"Unsupported AI provider: {provider}")
