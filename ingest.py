"""Command-line entry point for indexing supported files in ``documents/``.

Examples:
    python ingest.py path/to/report.pdf
    python ingest.py documents  # index every supported file in that folder

Google credentials can be supplied through GOOGLE_API_KEY or entered at the
prompt. The Streamlit upload/library workflow remains available in app.py.
"""

import argparse
import os
from getpass import getpass
from pathlib import Path

from rag_app.config import DOCUMENTS_DIR, SUPPORTED_EXTENSIONS
from rag_app.ingestion import ingest_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or folders to index (defaults to documents/)",
    )
    args = parser.parse_args()
    paths = args.paths or [DOCUMENTS_DIR]
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(
                candidate
                for candidate in sorted(path.iterdir())
                if candidate.is_file()
                and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        else:
            files.append(path)

    if not files:
        parser.error("No supported documents found (PDF, TXT, DOCX, CSV).")

    api_key = os.environ.get("GOOGLE_API_KEY") or getpass("Google API key: ")
    if not api_key:
        parser.error("A Google API key is required for document embeddings.")
    for path in files:
        print(ingest_file(path, api_key, replace=True))


if __name__ == "__main__":
    main()
