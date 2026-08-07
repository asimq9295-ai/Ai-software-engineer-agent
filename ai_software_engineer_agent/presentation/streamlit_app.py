from __future__ import annotations

import traceback

import streamlit as st

from ai_software_engineer_agent.application.dto import AgentInput
from ai_software_engineer_agent.application.services import AgentService
from ai_software_engineer_agent.domain.entities import AIProvider, AgentFeature
from ai_software_engineer_agent.domain.exceptions import TemporaryProviderError
from ai_software_engineer_agent.infrastructure.config import load_settings
from ai_software_engineer_agent.infrastructure.provider_factory import build_llm_gateway
from ai_software_engineer_agent.infrastructure.sqlite_repository import SQLiteAgentRunRepository


@st.cache_resource(show_spinner=False)
def build_repository() -> SQLiteAgentRunRepository:
    settings = load_settings()
    return SQLiteAgentRunRepository(settings.sqlite_path)


def build_service(provider: AIProvider, model: str) -> AgentService:
    llm_gateway = build_llm_gateway(provider, load_settings(), model=model)
    return AgentService(llm_gateway=llm_gateway, repository=build_repository())


def list_provider_models(provider: AIProvider) -> list[str]:
    settings = load_settings()
    if provider == AIProvider.GEMINI and not settings.google_api_key:
        return [settings.gemini_model]
    try:
        return build_llm_gateway(provider, settings).list_models()
    except Exception:
        if provider == AIProvider.GEMINI:
            return [settings.gemini_model]
        return [settings.ollama_model]


def main() -> None:
    st.set_page_config(
        page_title="AI Software Engineer Agent",
        layout="wide",
    )

    st.title("AI Software Engineer Agent")
    st.caption("Plan, generate, fix, test, and document software projects.")

    with st.sidebar:
        st.header("Workspace")
        settings = load_settings()
        provider_label = st.selectbox(
            "AI Provider",
            options=[provider.value for provider in AIProvider],
        )
        provider = AIProvider(provider_label)
        available_models = list_provider_models(provider)
        default_model = _default_model_for_provider(provider, available_models)
        model = st.selectbox(
            "Model",
            options=available_models,
            index=available_models.index(default_model),
        )
        fallback_to_ollama = st.checkbox(
            "Fallback to Ollama",
            value=True,
            disabled=provider != AIProvider.GEMINI,
        )
        if st.button("Test Connection", use_container_width=True):
            _test_connection(provider, model)
        with st.expander("Provider Diagnostics"):
            st.write(f"Selected provider: {provider.value}")
            st.write(f"Selected model: {model}")
            st.write(f"Gemini key configured: {bool(settings.google_api_key)}")
            st.write(f"Gemini model: {settings.gemini_model}")
            st.write(f"Ollama URL: {settings.ollama_base_url}")
            st.write(f"Ollama default model: {settings.ollama_model}")
        st.divider()
        feature_label = st.selectbox(
            "Capability",
            options=[feature.value for feature in AgentFeature],
        )
        st.divider()
        st.subheader("Recent Runs")
        _render_recent_runs()

    feature = AgentFeature(feature_label)
    prompt = st.text_area(
        "Request",
        height=180,
        placeholder="Describe the project, code, bug, test need, or documentation task.",
    )
    context = st.text_area(
        "Project Context",
        height=160,
        placeholder="Paste relevant code, requirements, errors, architecture notes, or constraints.",
    )

    submitted = st.button("Run Agent", type="primary", use_container_width=True)
    if submitted:
        _handle_submission(
            feature=feature,
            prompt=prompt,
            context=context,
            provider=provider,
            model=model,
            fallback_to_ollama=fallback_to_ollama,
        )


def _handle_submission(
    feature: AgentFeature,
    prompt: str,
    context: str,
    provider: AIProvider,
    model: str,
    fallback_to_ollama: bool,
) -> None:
    try:
        service = build_service(provider, model)
        with st.spinner("Engineering response..."):
            output = service.run(
                AgentInput(
                    feature=feature,
                    prompt=prompt,
                    context=context,
                )
            )
        st.success(f"Completed run {output.run_id}")
        st.markdown(output.response)
    except TemporaryProviderError as exc:
        if provider == AIProvider.GEMINI and fallback_to_ollama:
            st.warning(f"{exc} Trying Ollama instead.")
            _show_traceback(exc)
            _run_with_ollama_fallback(feature, prompt, context)
            return
        st.error(str(exc))
        _show_traceback(exc)
    except Exception as exc:
        st.error(str(exc))
        _show_traceback(exc)


def _run_with_ollama_fallback(
    feature: AgentFeature,
    prompt: str,
    context: str,
) -> None:
    settings = load_settings()
    try:
        service = build_service(AIProvider.OLLAMA, settings.ollama_model)
        with st.spinner("Fallback response from Ollama..."):
            output = service.run(
                AgentInput(
                    feature=feature,
                    prompt=prompt,
                    context=context,
                )
            )
        st.success(f"Completed fallback run {output.run_id}")
        st.markdown(output.response)
    except Exception as exc:
        st.error(f"Ollama fallback failed: {exc}")
        _show_traceback(exc)


def _test_connection(provider: AIProvider, model: str) -> None:
    try:
        gateway = build_llm_gateway(provider, load_settings(), model=model)
        healthy, message = gateway.health_check()
        if healthy:
            st.success(message)
        else:
            st.error(message)
    except Exception as exc:
        st.error(str(exc))
        _show_traceback(exc)


def _show_traceback(exc: Exception) -> None:
    with st.expander("Exception Traceback", expanded=True):
        st.code("".join(traceback.format_exception(exc)), language="text")


def _default_model_for_provider(provider: AIProvider, models: list[str]) -> str:
    settings = load_settings()
    configured = (
        settings.gemini_model if provider == AIProvider.GEMINI else settings.ollama_model
    )
    if configured in models:
        return configured
    return models[0]


def _render_recent_runs() -> None:
    try:
        runs = build_repository().list_recent(limit=10)
    except Exception as exc:
        st.info(f"History unavailable: {exc}")
        return

    if not runs:
        st.caption("No runs yet.")
        return

    for run in runs:
        with st.expander(f"{run.feature.value} - {run.created_at:%Y-%m-%d %H:%M}"):
            st.write(run.prompt)
            st.markdown(run.response)
