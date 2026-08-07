from __future__ import annotations

from dataclasses import dataclass

from ai_software_engineer_agent.domain.entities import AgentFeature


@dataclass(frozen=True)
class AgentInput:
    feature: AgentFeature
    prompt: str
    context: str = ""


@dataclass(frozen=True)
class AgentOutput:
    run_id: str
    response: str
