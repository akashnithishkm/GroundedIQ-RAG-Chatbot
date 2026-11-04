"""Evaluate retrieval against labels supplied for the user's own documents.

Edit ``EVALUATION_CASES`` after indexing files. These metrics assess whether an
expected source/page was retrieved; they do not measure answer accuracy.
"""

from .retrieval import retrieve

# Add entries shaped like:
# {"question": "...", "expected_source": "my-file.pdf",
#  "expected_page": 2, "review": "Check the answer against the passage."}
# Use expected_page=None when any page from the expected source is acceptable.
EVALUATION_CASES: list[dict] = []


def run_evaluation(api_key, k=4, mode="Similarity"):
    """Return labeled retrieval results, or an explanation when no labels exist."""
    if not EVALUATION_CASES:
        return [
            {
                "question": "No evaluation cases configured",
                "expected_source": "—",
                "expected_page": "—",
                "retrieved_sources": "—",
                f"Recall@{k}": "—",
                "Reciprocal rank": "—",
                "Human answer-quality review": (
                    "Index your files, then add questions and expected source/page "
                    "labels to EVALUATION_CASES in rag_app/evaluation.py. Retrieval "
                    "metrics measure source retrieval only, not answer accuracy."
                ),
            }
        ]

    rows = []
    recalls = []
    reciprocal_ranks = []

    for case in EVALUATION_CASES:
        documents = retrieve(case["question"], api_key, k, mode)
        expected_source = case["expected_source"]
        expected_page = case.get("expected_page")
        matching_ranks = [
            rank
            for rank, document in enumerate(documents, start=1)
            if document.metadata.get("source") == expected_source
            and (
                expected_page is None
                or str(document.metadata.get("page")) == str(expected_page)
            )
        ]
        reciprocal_rank = 1 / matching_ranks[0] if matching_ranks else 0
        recalls.append(int(bool(matching_ranks)))
        reciprocal_ranks.append(reciprocal_rank)
        rows.append(
            {
                "question": case["question"],
                "expected_source": expected_source,
                "expected_page": expected_page if expected_page is not None else "Any",
                "retrieved_sources": ", ".join(
                    dict.fromkeys(
                        document.metadata.get("source", "?") for document in documents
                    )
                ),
                f"Recall@{k}": int(bool(matching_ranks)),
                "Reciprocal rank": round(reciprocal_rank, 3),
                "Human answer-quality review": case.get(
                    "review", "Manually compare any generated answer with its evidence."
                ),
            }
        )

    count = len(EVALUATION_CASES)
    rows.append(
        {
            "question": "MEAN",
            "expected_source": "—",
            "expected_page": "—",
            "retrieved_sources": "—",
            f"Recall@{k}": round(sum(recalls) / count, 3),
            "Reciprocal rank": round(sum(reciprocal_ranks) / count, 3),
            "Human answer-quality review": (
                "Retrieval metrics only; they do not measure answer accuracy. "
                "Review answers against the retrieved passages."
            ),
        }
    )
    return rows
