"""Shared LLM factory — used across processors and researchers.

Add new providers here only. All callers (ContentStructurer, CaseStructurer,
research scripts) delegate to build_llm so there is a single place to update.
"""

from typing import Optional
from src.config import PRIMARY_LLM, GROQ_API_KEY, active_llm_provider


def build_llm(provider: str = "auto", model_name: Optional[str] = None,
              temperature: float = 0.1, max_tokens: int = 4096):
    """Return a LangChain chat model for the active provider."""
    if provider == "auto":
        provider = active_llm_provider()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name or PRIMARY_LLM,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.0-flash",
            temperature=temperature,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name or "llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=GROQ_API_KEY,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
        )
