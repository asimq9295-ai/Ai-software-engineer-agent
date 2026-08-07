from __future__ import annotations

import requests


class OllamaLLMGateway:
    _CONNECT_TIMEOUT_SECONDS = 10
    def __init__(
        self,
        base_url: str,
        model: str = "qwen2.5:1.5b",
        num_predict: int = 256,
        read_timeout: int = 900,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model or "qwen2.5:1.5b"
        self._num_predict = num_predict
        self._read_timeout = read_timeout

    @property
    def model(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {
                    "num_predict": self._num_predict,
                    "temperature": 0.2,
                },
            },
            timeout=(self._CONNECT_TIMEOUT_SECONDS, self._read_timeout),
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("response", "")
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text

    def list_models(self) -> list[str]:
        response = requests.get(
            f"{self._base_url}/api/tags",
            timeout=(self._CONNECT_TIMEOUT_SECONDS, 30),
        )
        response.raise_for_status()
        payload = response.json()
        names = [model.get("name", "") for model in payload.get("models", [])]
        names = sorted(name for name in names if name)
        return names or [self._model]

    def health_check(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
            if self._model not in models:
                return (
                    False,
                    f"Ollama is reachable, but model '{self._model}' was not found. Run: ollama pull {self._model}",
                )
            return True, f"Ollama is reachable using {self._model}."
        except Exception as exc:
            return (
                False,
                f"Ollama connection failed. Start Ollama and verify {self._base_url}: {exc}",
            )
