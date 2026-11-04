from pathlib import Path
from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from rag_app import ingestion, workflow
from rag_app.security import safe_error, safe_filename


def test_safe_filename_and_secret_redaction():
    assert safe_filename("../../report?.pdf") == "report_.pdf"
    assert "[redacted]" in safe_error(
        RuntimeError("provider said sk-secret"), ["sk-secret"]
    )


def test_txt_metadata(monkeypatch, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Known facts", encoding="utf-8")
    docs = ingestion.extract_documents(path, "abc123")
    assert docs[0].page_content == "Known facts"
    assert docs[0].metadata["source"] == "notes.txt"
    assert docs[0].metadata["document_id"] == "abc123"
    assert docs[0].metadata["embedding_model"]


def test_duplicate_does_not_rewrite(monkeypatch, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("same", encoding="utf-8")
    store = MagicMock()
    store.get.return_value = {
        "ids": ["old"],
        "metadatas": [
            {"document_id": __import__("hashlib").sha256(b"same").hexdigest()}
        ],
    }
    monkeypatch.setattr(ingestion, "vectorstore", lambda key: store)
    result = ingestion.ingest_file(path, "mock")
    assert result["status"] == "duplicate"
    store.add_documents.assert_not_called()


def test_reindex_replaces_only_after_extract(monkeypatch, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("new text", encoding="utf-8")
    store = MagicMock()
    store.get.return_value = {
        "ids": ["old"],
        "metadatas": [{"document_id": "old-hash"}],
    }
    monkeypatch.setattr(ingestion, "vectorstore", lambda key: store)
    monkeypatch.setattr(
        ingestion,
        "extract_documents",
        lambda p, d: [
            Document(
                page_content="new text", metadata={"source": p.name, "document_id": d}
            )
        ],
    )
    result = ingestion.ingest_file(path, "mock", replace=True)
    store.delete.assert_called_once_with(ids=["old"])
    store.add_documents.assert_called_once()
    assert result["status"] == "indexed"


@pytest.mark.parametrize("mode", ["Similarity", "MMR", "Hybrid"])
def test_retrieval_modes(monkeypatch, mode):
    from rag_app import retrieval

    doc = Document(page_content="evidence", metadata={"source": "a.pdf", "page": 1})
    fake = MagicMock()
    fake.similarity_search.return_value = [doc]
    fake.max_marginal_relevance_search.return_value = [doc]
    monkeypatch.setattr(retrieval, "vectorstore", lambda key: fake)
    assert retrieval.retrieve("q", "key", mode=mode)


def test_citations_are_from_retrieved_documents():
    source = Document(
        page_content="Passage excerpt", metadata={"source": "book.pdf", "page": 7}
    )
    citation = workflow.format_sources([source])[0]
    assert citation["source"] == "book.pdf" and citation["page"] == 7
    assert citation["snippet"] == "Passage excerpt"


def test_no_evidence_short_circuits_generation(monkeypatch):
    monkeypatch.setattr(workflow, "retrieve", lambda *a, **k: [])
    graph = workflow.build_rag_graph("key")
    result = graph.invoke({"question": "unknown", "history": []})
    assert result["answer"] == workflow.NO_EVIDENCE
    assert result["active_provider"] == "Not used"


def test_followup_rewrites_query_from_recent_history(monkeypatch):
    from rag_app import workflow as wf

    seen = []

    class FakeLLM:
        def invoke(self, messages):
            return MagicMock(content="When did the founder leave the company?")

    monkeypatch.setattr(wf, "create_llm", lambda *a, **k: FakeLLM())
    monkeypatch.setattr(wf, "retrieve", lambda q, *a, **k: (seen.append(q) or []))
    graph = wf.build_rag_graph("key")
    result = graph.invoke(
        {
            "question": "When did they leave?",
            "history": [
                {"role": "user", "content": "Tell me about the founder"},
                {"role": "assistant", "content": "They led a company."},
            ],
        }
    )
    assert seen == ["When did the founder leave the company?"]
    assert result["answer"] == wf.NO_EVIDENCE
