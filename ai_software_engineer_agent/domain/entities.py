from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4


class AgentFeature(StrEnum):
    PROJECT_PLANNER = "Project Planner"
    CODE_GENERATOR = "Code Generator"
    BUG_FIXER = "Bug Fixer"
    TEST_GENERATOR = "Test Generator"
    DOCUMENTATION_WRITER = "Documentation Writer"


class AIProvider(StrEnum):
    GEMINI = "Gemini"
    OLLAMA = "Ollama"


@dataclass(frozen=True)
class AgentRequest:
    feature: AgentFeature
    prompt: str
    context: str = ""

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("Prompt is required.")


@dataclass(frozen=True)
class AgentRun:
    id: str
    feature: AgentFeature
    prompt: str
    context: str
    response: str
    created_at: datetime

    @classmethod
    def create(
        cls,
        feature: AgentFeature,
        prompt: str,
        context: str,
        response: str,
    ) -> "AgentRun":
        return cls(
            id=str(uuid4()),
            feature=feature,
            prompt=prompt,
            context=context,
            response=response,
            created_at=datetime.now(timezone.utc),
        )
