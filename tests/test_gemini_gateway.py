from __future__ import annotations

import pytest

from ai_software_engineer_agent.infrastructure.gemini_gateway import GeminiLLMGateway


def test_gemini_gateway_requires_google_api_key() -> None:
    with pytest.raises(ValueError, match="GOOGLE_API_KEY is missing"):
        GeminiLLMGateway(api_key="", model="gemini-2.5-flash")
