from __future__ import annotations

from typing import Protocol

from ai_software_engineer_agent.domain.entities import AgentRun


class LLMGateway(Protocol):
    @property
    def model(self) -> str:
        """Return the configured model name."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an AI response from a system and user prompt."""

    def list_models(self) -> list[str]:
        """Return available model names for this provider."""

    def health_check(self) -> tuple[bool, str]:
        """Return provider health status and a concise message."""


class AgentRunRepository(Protocol):
    def save(self, run: AgentRun) -> None:
        """Persist a completed agent run."""

    def list_recent(self, limit: int = 20) -> list[AgentRun]:
        """Return recent agent runs, newest first."""
