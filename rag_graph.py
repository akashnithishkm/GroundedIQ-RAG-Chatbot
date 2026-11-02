from getpass import getpass
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ingest import get_vectorstore


PROVIDER_MODELS = {
    "Gemini": [
        "gemini-3.8-flash",
        "gemini-2.5-flash",
    ],
    "OpenRouter": [
        "openrouter/free",
    ],
    "Ollama": [
        "llama3.2:3b",
        "phi3:mini",
        "mistral:7b",
    ],
}

DEFAULT_MODELS = {
    provider: models[0]
    for provider, models in PROVIDER_MODELS.items()
}


class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    sources: list
    active_provider: str
    active_model: str


def extract_response_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts)

    return str(content)


def create_llm(provider, model, api_key=None):
    if provider == "Gemini":
        if not api_key:
            raise ValueError(
                "A Google API key is required for Gemini generation."
            )

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.1,
        )

    if provider == "OpenRouter":
        if not api_key:
            raise ValueError(
                "An OpenRouter API key is required for OpenRouter generation."
            )

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
        )

    if provider == "Ollama":
        return ChatOllama(
            model=model,
            temperature=0.1,
        )

    raise ValueError(f"Unsupported AI provider: {provider}")


def build_rag_graph(
    api_key,
    top_k=3,
    provider="Gemini",
    model=None,
    provider_api_key=None,
    auto_fallback=False,
):
    if not api_key:
        raise ValueError(
            "A Google API key is required for PDF embeddings and retrieval."
        )

    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Unsupported AI provider: {provider}")

    selected_model = model or DEFAULT_MODELS[provider]
    if selected_model not in PROVIDER_MODELS[provider]:
        raise ValueError(
            f"Unsupported model for {provider}: {selected_model}"
        )

    vectorstore = get_vectorstore(api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    provider_keys = {
        "Gemini": api_key,
        "OpenRouter": provider_api_key,
        "Ollama": None,
    }

    provider_order = list(PROVIDER_MODELS)
    providers_to_try = [provider]
    if auto_fallback:
        providers_to_try.extend(
            candidate
            for candidate in provider_order
            if candidate != provider
        )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF question-answering assistant.

Rules:
1. Answer only using the supplied context.
2. Do not invent facts.
3. If the context does not contain the answer, say so clearly.
4. Give a clear and easy-to-understand answer.
5. Do not claim information comes from the PDF unless it is supported by the context.

Context:
{context}

Question:
{question}

Answer:
""")

    def retrieve_node(state: RAGState):
        print("\n[RETRIEVAL NODE] Searching ChromaDB...")
        documents = retriever.invoke(state["question"])

        sources = []
        context_parts = []
        for document in documents:
            context_parts.append(document.page_content)
            sources.append(
                {
                    "source": document.metadata.get("source"),
                    "page": document.metadata.get("page"),
                    "document_id": document.metadata.get("document_id"),
                }
            )

        context = "\n\n".join(context_parts)
        print("Retrieved chunks:", len(documents))
        return {"context": context, "sources": sources}

    def generate_node(state: RAGState):
        if not state["context"].strip():
            return {
                "answer": "I couldn't find relevant information in the uploaded PDFs.",
                "active_provider": "Not used",
                "active_model": "",
            }

        failures = []
        for current_provider in providers_to_try:
            current_model = (
                selected_model
                if current_provider == provider
                else DEFAULT_MODELS[current_provider]
            )

            try:
                llm = create_llm(
                    current_provider,
                    current_model,
                    provider_keys[current_provider],
                )
                response = (prompt | llm).invoke(
                    {
                        "context": state["context"],
                        "question": state["question"],
                    }
                )
                return {
                    "answer": extract_response_text(response),
                    "active_provider": current_provider,
                    "active_model": current_model,
                }
            except Exception as error:
                failures.append(
                    f"{current_provider} ({current_model}): {error}"
                )
                if not auto_fallback:
                    raise RuntimeError(failures[-1]) from error

        raise RuntimeError(
            "All configured generation providers failed: "
            + " | ".join(failures)
        )

    graph_builder = StateGraph(RAGState)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()


if __name__ == "__main__":
    api_key = getpass("Google API key: ")
    if not api_key:
        raise ValueError("A Google API key is required.")

    rag_graph = build_rag_graph(api_key)
    question = input("\nAsk a question about your PDFs: ")
    result = rag_graph.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
            "sources": [],
            "active_provider": "",
            "active_model": "",
        }
    )

    print("\nQUESTION:")
    print(question)
    print("\nANSWER:")
    print(result["answer"])
    print(f"\nPROVIDER: {result['active_provider']} ({result['active_model']})")
    print("\nSOURCES:")
    for source in result["sources"]:
        print(source)
