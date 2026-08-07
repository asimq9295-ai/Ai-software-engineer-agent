from __future__ import annotations

import time

from google import genai

from ai_software_engineer_agent.domain.exceptions import TemporaryProviderError


class GeminiLLMGateway:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Add your Google AI Studio API key to .env."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(2):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user_prompt,
                    config={
                        "system_instruction": system_prompt,
                        "temperature": 0.2,
                    },
                )
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("Gemini returned an empty response.")
                return text
            except Exception as exc:
                if self._is_temporary_error(exc):
                    if attempt == 0:
                        time.sleep(1)
                        continue
                    raise TemporaryProviderError(
                        "Gemini is temporarily unavailable. You can retry later or switch to Ollama."
                    ) from exc
                raise
        raise TemporaryProviderError("Gemini is temporarily unavailable.")

    def list_models(self) -> list[str]:
        try:
            models = self._client.models.list()
            names = []
            for model in models:
                name = getattr(model, "name", "")
                if name.startswith("models/"):
                    name = name.removeprefix("models/")
                if "gemini" in name.lower():
                    names.append(name)
            return sorted(set(names)) or [self._model]
        except Exception:
            return [self._model]

    def health_check(self) -> tuple[bool, str]:
        try:
            self.generate("Return exactly OK.", "Connection test")
            return True, f"Gemini is reachable using {self._model}."
        except Exception as exc:
            return False, f"Gemini connection failed: {exc}"

    @staticmethod
    def _is_temporary_error(exc: Exception) -> bool:
        message = str(exc).upper()
        return any(
            marker in message
            for marker in ("503", "UNAVAILABLE", "TEMPORARILY", "OVERLOADED")
        )
