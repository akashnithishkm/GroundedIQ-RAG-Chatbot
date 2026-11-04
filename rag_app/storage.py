"""Chroma operations and safe document lifecycle management."""

from pathlib import Path
import shutil
import chromadb
from langchain_chroma import Chroma
from .config import BASE_DIR, CHROMA_DIR, COLLECTION_NAME
from .embeddings import google_embeddings


def vectorstore(api_key: str) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=google_embeddings(api_key),
        persist_directory=str(CHROMA_DIR),
    )


def indexed_documents(api_key: str | None = None) -> tuple[int, dict[str, dict]]:
    if not CHROMA_DIR.exists():
        return 0, {}
    with chromadb.PersistentClient(path=str(CHROMA_DIR)) as client:
        if COLLECTION_NAME not in {c.name for c in client.list_collections()}:
            return 0, {}
        rows = client.get_collection(COLLECTION_NAME).get(include=["metadatas"])
    files: dict[str, dict] = {}
    for meta in rows.get("metadatas") or []:
        if not meta:
            continue
        name = meta.get("source", "Unknown")
        item = files.setdefault(
            name,
            {
                "chunks": 0,
                "document_id": meta.get("document_id", ""),
                "embedding_model": meta.get("embedding_model", "legacy/unknown"),
            },
        )
        item["chunks"] += 1
    return len(rows.get("ids") or []), files


def delete_document(filename: str, api_key: str) -> None:
    del api_key  # Kept in the signature for compatibility with older callers.
    with chromadb.PersistentClient(path=str(CHROMA_DIR)) as client:
        if COLLECTION_NAME not in {c.name for c in client.list_collections()}:
            return
        collection = client.get_collection(COLLECTION_NAME)
        existing = collection.get(where={"source": filename})
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])


def clear_index() -> None:
    if not CHROMA_DIR.exists():
        return
    with chromadb.PersistentClient(path=str(CHROMA_DIR)) as client:
        if COLLECTION_NAME in {c.name for c in client.list_collections()}:
            client.delete_collection(COLLECTION_NAME)


def delete_chroma_database() -> None:
    """Remove the configured Chroma directory, never an arbitrary user path."""
    root = BASE_DIR.resolve()
    target = CHROMA_DIR.resolve()
    expected = (root / "chroma_db").resolve()
    if CHROMA_DIR.is_symlink() or target != expected or target.parent != root:
        raise ValueError(
            "Refusing to delete a Chroma path outside the configured project data folder."
        )
    if not target.exists():
        return
    # Close Chroma's SQLite handles before deleting its files on Windows.
    if target.is_dir():
        with chromadb.PersistentClient(path=str(target)):
            pass
        shutil.rmtree(target)


def remove_upload(filename: str) -> bool:
    """Delete only a named file directly inside the configured documents folder."""
    root = Path(
        __import__("rag_app.config", fromlist=["DOCUMENTS_DIR"]).DOCUMENTS_DIR
    ).resolve()
    target = (root / Path(filename).name).resolve()
    if target.parent != root or not target.is_file():
        return False
    target.unlink()
    return True
