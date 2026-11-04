"""LLM provider construction."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import DEFAULT_MODELS, PROVIDER_MODELS


def create_llm(provider, model, api_key=None, temperature=0.1):
    if provider == "Gemini":
        if not api_key:
            raise ValueError("A Google API key is required for Gemini generation.")
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=temperature, timeout=45
        )
    if provider == "OpenRouter":
        if not api_key:
            raise ValueError(
                "An OpenRouter API key is required for OpenRouter generation."
            )
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            timeout=45,
        )
    if provider == "Ollama":
        return ChatOllama(model=model, temperature=temperature)
    raise ValueError(f"Unsupported AI provider: {provider}")
