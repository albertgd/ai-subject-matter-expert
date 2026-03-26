"""Shared LLM factory — used across processors and researchers."""

from typing import Optional
from src.config import PRIMARY_LLM, active_llm_provider


def build_llm(provider: str = "auto", model_name: Optional[str] = None):
    """Return a LangChain chat model for the active provider."""
    if provider == "auto":
        provider = active_llm_provider()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name or PRIMARY_LLM,
            temperature=0.1,
            max_tokens=4096,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.5-flash",
            temperature=0.1,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=0.1,
        )
