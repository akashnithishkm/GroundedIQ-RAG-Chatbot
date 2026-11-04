"""Validated multi-format ingestion with hash-based idempotency and replacement."""

import csv, hashlib
from pathlib import Path
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MAX_UPLOAD_BYTES,
    SUPPORTED_EXTENSIONS,
)
from .security import safe_filename
from .storage import vectorstore


def extract_documents(path: Path, digest: str) -> list[Document]:
    ext = path.suffix.lower()
    docs: list[Document] = []
    if ext == ".pdf":
        with pymupdf.open(path) as pdf:
            docs = [
                Document(page_content=p.get_text(), metadata={"page": i})
                for i, p in enumerate(pdf, 1)
                if p.get_text().strip()
            ]
    elif ext == ".txt":
        docs = [
            Document(page_content=path.read_text(encoding="utf-8-sig"), metadata={})
        ]
    elif ext == ".docx":
        from docx import Document as WordDocument

        word = WordDocument(path)
        paragraphs = [p.text for p in word.paragraphs if p.text.strip()]
        docs = [Document(page_content="\n".join(paragraphs), metadata={})]
        for i, table in enumerate(word.tables, 1):
            docs.append(
                Document(
                    page_content="\n".join(
                        " | ".join(c.text for c in row.cells) for row in table.rows
                    ),
                    metadata={"section": f"Table {i}"},
                )
            )
    elif ext == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            for index, row in enumerate(csv.DictReader(f), 1):
                docs.append(
                    Document(
                        page_content=" | ".join(
                            f"{k}: {v}" for k, v in row.items() if v
                        ),
                        metadata={"row": index},
                    )
                )
    else:
        raise ValueError("Supported file types are PDF, TXT, DOCX, and CSV.")
    for doc in docs:
        doc.metadata.update(
            {
                "source": safe_filename(path.name),
                "document_id": digest,
                "embedding_model": EMBEDDING_MODEL,
            }
        )
    return docs


def ingest_file(file_path: str | Path, api_key: str, replace: bool = False) -> dict:
    path = Path(file_path).resolve()
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported file types are PDF, TXT, DOCX, and CSV.")
    if not path.is_file():
        raise FileNotFoundError("Selected document was not found.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("File exceeds the 25 MB upload limit.")
    safe_filename(path.name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    store = vectorstore(api_key)
    existing = store.get(where={"source": path.name}, include=["metadatas"])
    if existing.get("ids"):
        metas = existing.get("metadatas") or []
        if metas and metas[0].get("document_id") == digest:
            return {
                "filename": path.name,
                "chunks": len(existing["ids"]),
                "document_id": digest,
                "status": "duplicate",
            }
        if not replace:
            raise ValueError(
                "A different version is indexed. Choose reindex to replace it safely."
            )
    raw = extract_documents(path, digest)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(raw)
    if not chunks:
        raise ValueError(
            "No readable text found. Scanned PDF OCR is not enabled; see README for optional setup."
        )
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{digest}_{index}"
    ids = [f"{digest}_{i}" for i in range(len(chunks))]
    # Add the new version first so a provider/embedding failure leaves the old
    # indexed version usable. Content hashes make the new IDs non-colliding.
    store.add_documents(chunks, ids=ids)
    if existing.get("ids"):
        store.delete(ids=existing["ids"])
    return {
        "filename": path.name,
        "chunks": len(chunks),
        "pages": len(raw),
        "document_id": digest,
        "status": "indexed",
    }
