from __future__ import annotations

from ai_software_engineer_agent.application.dto import AgentInput, AgentOutput
from ai_software_engineer_agent.domain.entities import AgentRequest, AgentRun
from ai_software_engineer_agent.domain.ports import AgentRunRepository, LLMGateway
from ai_software_engineer_agent.domain.prompts import build_system_prompt


class AgentService:
    def __init__(self, llm_gateway: LLMGateway, repository: AgentRunRepository) -> None:
        self._llm_gateway = llm_gateway
        self._repository = repository

    def run(self, request_input: AgentInput) -> AgentOutput:
        request = AgentRequest(
            feature=request_input.feature,
            prompt=request_input.prompt,
            context=request_input.context,
        )
        request.validate()

        system_prompt = build_system_prompt(request.feature)
        user_prompt = self._build_user_prompt(request)
        response = self._llm_gateway.generate(system_prompt, user_prompt)

        run = AgentRun.create(
            feature=request.feature,
            prompt=request.prompt,
            context=request.context,
            response=response,
        )
        self._repository.save(run)
        return AgentOutput(run_id=run.id, response=response)

    def recent_runs(self, limit: int = 20) -> list[AgentRun]:
        return self._repository.list_recent(limit=limit)

    @staticmethod
    def _build_user_prompt(request: AgentRequest) -> str:
        parts = [f"User request:\n{request.prompt.strip()}"]
        if request.context.strip():
            parts.append(f"Additional project context:\n{request.context.strip()}")
        return "\n\n".join(parts)
