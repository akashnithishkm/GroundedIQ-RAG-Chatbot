"""Embedding provider construction."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import EMBEDDING_MODEL


def google_embeddings(api_key: str):
    if not api_key:
        raise ValueError(
            "A Google API key is required for document embeddings and retrieval."
        )
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
