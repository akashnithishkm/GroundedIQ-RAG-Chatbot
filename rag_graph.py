"""Interactive command-line chat entry point for the LangGraph RAG workflow."""

from getpass import getpass
from rag_app.workflow import build_rag_graph

if __name__ == "__main__":
    key = getpass("Google API key: ")
    graph = build_rag_graph(key)
    question = input("Ask a question: ")
    result = graph.invoke({"question": question, "history": []})
    print(result.get("answer", ""))
