from __future__ import annotations

from unittest.mock import Mock, patch

from ai_software_engineer_agent.infrastructure.ollama_gateway import OllamaLLMGateway


def test_ollama_gateway_lists_available_models() -> None:
    response = Mock()
    response.json.return_value = {"models": [{"name": "llama3"}, {"name": "mistral"}]}
    response.raise_for_status.return_value = None

    with patch("ai_software_engineer_agent.infrastructure.ollama_gateway.requests.get") as get:
        get.return_value = response
        gateway = OllamaLLMGateway("http://localhost:11434", "llama3")

        assert gateway.list_models() == ["llama3", "mistral"]


def test_ollama_gateway_generates_text() -> None:
    response = Mock()
    response.json.return_value = {"response": "Generated answer"}
    response.raise_for_status.return_value = None

    with patch("ai_software_engineer_agent.infrastructure.ollama_gateway.requests.post") as post:
        post.return_value = response
        gateway = OllamaLLMGateway("http://localhost:11434", "llama3")

        assert gateway.generate("system", "prompt") == "Generated answer"
        post.assert_called_once()
