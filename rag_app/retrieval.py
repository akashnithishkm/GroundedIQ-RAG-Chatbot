"""Configurable dense, MMR, and reciprocal-rank hybrid retrieval."""

from langchain_community.retrievers import BM25Retriever

from .storage import vectorstore


def retrieve(question, api_key, top_k=4, mode="Similarity", source_filter=None):
    store = vectorstore(api_key)
    metadata_filter = {"source": source_filter} if source_filter else None

    if mode == "MMR":
        return store.max_marginal_relevance_search(
            question,
            k=top_k,
            fetch_k=max(20, top_k * 4),
            filter=metadata_filter,
        )

    dense_documents = store.similarity_search(question, k=top_k, filter=metadata_filter)
    if mode != "Hybrid":
        return dense_documents

    candidates = store.similarity_search(
        question, k=max(top_k * 4, 20), filter=metadata_filter
    )
    if not candidates:
        return []

    try:
        sparse_retriever = BM25Retriever.from_documents(candidates)
        sparse_retriever.k = top_k
        lexical_documents = sparse_retriever.invoke(question)
    except Exception:
        return dense_documents

    fused_ranks = {}
    for rank, document in enumerate(dense_documents + lexical_documents, start=1):
        document_key = document.metadata.get("chunk_id") or (
            document.metadata.get("source"),
            document.metadata.get("page"),
            document.page_content[:100],
        )
        current_score, _ = fused_ranks.get(document_key, (0, document))
        fused_ranks[document_key] = current_score + 1 / (60 + rank), document

    ranked_documents = sorted(
        fused_ranks.values(), key=lambda item: item[0], reverse=True
    )
    return [document for _, document in ranked_documents[:top_k]]
