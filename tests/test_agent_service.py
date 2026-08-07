from __future__ import annotations

import pytest

from ai_software_engineer_agent.application.dto import AgentInput
from ai_software_engineer_agent.application.services import AgentService
from ai_software_engineer_agent.domain.entities import AgentFeature, AgentRun


class FakeLLMGateway:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        assert "Task mode" in system_prompt
        assert "User request" in user_prompt
        return "Generated answer"


class InMemoryRepository:
    def __init__(self) -> None:
        self.runs: list[AgentRun] = []

    def save(self, run: AgentRun) -> None:
        self.runs.append(run)

    def list_recent(self, limit: int = 20) -> list[AgentRun]:
        return self.runs[-limit:][::-1]


def test_agent_service_generates_and_persists_run() -> None:
    repository = InMemoryRepository()
    service = AgentService(FakeLLMGateway(), repository)

    output = service.run(
        AgentInput(
            feature=AgentFeature.CODE_GENERATOR,
            prompt="Create a FastAPI endpoint",
            context="Use SQLite",
        )
    )

    assert output.response == "Generated answer"
    assert output.run_id == repository.runs[0].id
    assert repository.runs[0].feature == AgentFeature.CODE_GENERATOR


@pytest.mark.parametrize("feature", list(AgentFeature))
def test_agent_service_supports_all_capabilities(feature: AgentFeature) -> None:
    repository = InMemoryRepository()
    service = AgentService(FakeLLMGateway(), repository)

    output = service.run(
        AgentInput(
            feature=feature,
            prompt=f"Run {feature.value}",
            context="Use the selected provider",
        )
    )

    assert output.response == "Generated answer"
    assert repository.runs[0].feature == feature


def test_agent_service_rejects_blank_prompt() -> None:
    service = AgentService(FakeLLMGateway(), InMemoryRepository())

    with pytest.raises(ValueError, match="Prompt is required"):
        service.run(
            AgentInput(
                feature=AgentFeature.PROJECT_PLANNER,
                prompt=" ",
            )
        )
