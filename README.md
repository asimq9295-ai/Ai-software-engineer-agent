# AI Software Engineer Agent

A production-ready Streamlit application for AI-assisted software engineering workflows.

## Features

- Project Planner
- Code Generator
- Bug Fixer
- Test Generator
- Documentation Writer
- Gemini and Ollama provider selection
- Dynamic model selection per provider
- Provider connection testing
- Gemini retry with optional Ollama fallback
- SQLite-backed run history
- Clean architecture with separable domain, application, infrastructure, and presentation layers

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

For Gemini, set `GOOGLE_API_KEY` in `.env` with your Google AI Studio API key.
For Ollama, install Ollama locally and pull the default model:

```powershell
ollama pull qwen2.5:1.5b
```

Then run:

```powershell
streamlit run app.py
```

Use the sidebar to choose `Gemini` or `Ollama`, pick an available model, and test the connection before running an agent task.

## Project Structure

```text
ai_software_engineer_agent/
  application/
    dto.py
    services.py
  domain/
    entities.py
    exceptions.py
    ports.py
    prompts.py
  infrastructure/
    config.py
    gemini_gateway.py
    ollama_gateway.py
    provider_factory.py
    sqlite_repository.py
  presentation/
    streamlit_app.py
tests/
  test_agent_service.py
app.py
requirements.txt
```

## Environment

Gemini:

- `GOOGLE_API_KEY`
- `GEMINI_MODEL`, default `gemini-2.5-flash`

Ollama:

- `OLLAMA_BASE_URL`, default `http://localhost:11434`
- `OLLAMA_MODEL`, default `qwen2.5:1.5b`
- `OLLAMA_NUM_PREDICT`, default `64`
- `OLLAMA_READ_TIMEOUT`, default `900`

Storage:

- `DATABASE_URL`, default `sqlite:///agent_runs.db`

If `GOOGLE_API_KEY` is missing, Gemini displays a clear configuration error. Ollama requires the local Ollama service to be running and the selected model to be available.

## Provider Behavior

Gemini uses the configured `GEMINI_MODEL` by default and lists Gemini models when the API key is available. Ollama lists local models from `/api/tags` and uses `qwen2.5:1.5b` by default.

The `Test Connection` button runs the selected provider health check. If Gemini returns a temporary outage such as `503 UNAVAILABLE`, the app retries once and can fall back to Ollama when the fallback checkbox is enabled.

## Testing

```powershell
pytest
```
