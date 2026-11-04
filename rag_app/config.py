"""Central paths and application settings."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
SQLITE_PATH = BASE_DIR / "conversations.sqlite3"
COLLECTION_NAME = "pdf_documents"  # Keep the existing collection intact.
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "gemini-embedding-001"
HISTORY_TURNS = 6

PROVIDER_MODELS = {
    "Gemini": ["gemini-2.5-flash", "gemini-2.0-flash"],
    "OpenRouter": ["openrouter/free"],
    "Ollama": ["llama3.2:3b", "phi3:mini", "mistral:7b"],
}
DEFAULT_MODELS = {name: models[0] for name, models in PROVIDER_MODELS.items()}
