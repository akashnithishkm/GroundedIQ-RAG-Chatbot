import hashlib
from pathlib import Path

import pymupdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "pdf_documents"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def get_vectorstore(api_key):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )


def ingest_pdf(file_path, api_key):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    print(f"\nProcessing: {file_path.name}")

    # 1. Calculate a unique fingerprint for the PDF
    file_bytes = file_path.read_bytes()
    document_id = hashlib.sha256(file_bytes).hexdigest()

    # 2. Extract text page by page
    documents = []

    with pymupdf.open(file_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text()

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "page": page_number,
                            "document_id": document_id
                        }
                    )
                )

    if not documents:
        raise ValueError(
            "No readable text found. This may be a scanned PDF."
        )

    print("Pages extracted:", len(documents))

    # 3. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    print("Chunks created:", len(chunks))

    # 4. Connect to ChromaDB
    vectorstore = get_vectorstore(api_key)

    # 5. Remove previous version of the same filename
    existing = vectorstore.get(
        where={"source": file_path.name}
    )

    if existing["ids"]:

        vectorstore.delete(
            ids=existing["ids"]
        )

        print("Previous version removed.")

    # 6. Generate unique IDs for chunks
    ids = [
        f"{document_id}_{index}"
        for index in range(len(chunks))
    ]

    # 7. Generate embeddings and store chunks
    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    print("Embeddings generated and stored.")

    print("PDF ingestion completed successfully!")

    return {
        "filename": file_path.name,
        "pages": len(documents),
        "chunks": len(chunks),
        "document_id": document_id
    }


if __name__ == "__main__":

    from getpass import getpass

    api_key = getpass("Google API key: ")

    if not api_key:
        raise ValueError("A Google API key is required.")

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found.")

    for pdf_file in pdf_files:
        ingest_pdf(pdf_file, api_key)
