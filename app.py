
from pathlib import Path

import chromadb
import streamlit as st

from ingest import CHROMA_DIR, COLLECTION_NAME, ingest_pdf, get_vectorstore
from rag_graph import DEFAULT_MODELS, PROVIDER_MODELS, build_rag_graph


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG | GenAI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. CUSTOM CSS - PREMIUM DARK UI
# ============================================================

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

.stApp {
    background:
        radial-gradient(ellipse at 80% 0%, rgba(91, 54, 190, 0.20), transparent 35%),
        radial-gradient(ellipse at 40% 60%, rgba(0, 140, 255, 0.08), transparent 40%),
        #080d20;
    color: #edf1ff;
}

header[data-testid="stHeader"] {
    background: rgba(8, 13, 32, 0.8);
}

.block-container {
    max-width: 1600px;
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11152e 0%, #171638 100%);
    border-right: 1px solid rgba(130, 140, 255, 0.20);
}

section[data-testid="stSidebar"] * {
    color: #edf1ff;
}

section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"],
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-caption {
    color: #a6afd0;
    font-size: 12px;
    margin-top: -8px;
    margin-bottom: 25px;
}

/* ---------- HEADINGS ---------- */

.hero-title {
    font-size: clamp(25px, 3vw, 39px);
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -1.3px;
    color: #f7f8ff;
}

.gradient-text {
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subheading {
    color: #a9b4d4;
    font-size: 14px;
    margin-top: 8px;
    margin-bottom: 20px;
}

/* ---------- STAT CARDS ---------- */

.stat-card {
    background: linear-gradient(
        135deg,
        rgba(36, 48, 91, 0.80),
        rgba(30, 25, 73, 0.75)
    );
    border: 1px solid rgba(123, 151, 255, 0.30);
    border-radius: 16px;
    padding: 15px 13px;
    min-height: 88px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.stat-label {
    color: #aab7d8;
    font-size: 12px;
}

.stat-value {
    font-size: 25px;
    font-weight: 800;
    color: #f7f8ff;
    margin-top: 5px;
}

/* ---------- PANELS ---------- */

.glass-panel {
    background: linear-gradient(
        145deg,
        rgba(21, 29, 58, 0.96),
        rgba(16, 23, 47, 0.94)
    );
    border: 1px solid rgba(112, 137, 220, 0.35);
    border-radius: 19px;
    padding: 19px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.20);
    margin-bottom: 15px;
}

.panel-title {
    color: #f4f6ff;
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 5px;
}

.panel-caption {
    color: #9caaca;
    font-size: 12px;
    margin-bottom: 15px;
}

/* ---------- CHAT ---------- */

.welcome-card {
    background: linear-gradient(
        120deg,
        rgba(35, 49, 91, 0.92),
        rgba(42, 38, 91, 0.85)
    );
    border: 1px solid rgba(128, 151, 255, 0.32);
    border-radius: 17px;
    padding: 21px;
    margin-bottom: 17px;
}

.welcome-title {
    font-size: 19px;
    font-weight: 750;
    color: #ffffff;
}

.welcome-text {
    color: #c6d0ed;
    font-size: 13px;
    line-height: 1.7;
}

.feature-pill {
    display: inline-block;
    background: rgba(72, 93, 160, 0.25);
    border: 1px solid rgba(128, 157, 255, 0.30);
    color: #c6d9ff;
    padding: 7px 10px;
    border-radius: 20px;
    font-size: 11px;
    margin: 5px 4px 0 0;
}

.user-bubble {
    background: linear-gradient(110deg, #793ce0, #4245c9);
    border: 1px solid #a879ff;
    border-radius: 17px 17px 4px 17px;
    padding: 13px 17px;
    color: white;
    margin: 8px 0;
}

.assistant-bubble {
    background: linear-gradient(
        130deg,
        rgba(31, 43, 78, 0.98),
        rgba(26, 35, 68, 0.96)
    );
    border: 1px solid rgba(104, 143, 230, 0.35);
    border-radius: 17px;
    padding: 17px;
    color: #edf2ff;
    margin: 8px 0;
}

/* ---------- DOCUMENT CARDS ---------- */

.document-card {
    background: rgba(39, 52, 91, 0.65);
    border: 1px solid rgba(117, 146, 225, 0.32);
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
}

.document-name {
    color: #f3f5ff;
    font-size: 12px;
    font-weight: 650;
    overflow-wrap: anywhere;
}

.document-meta {
    color: #a9b7d9;
    font-size: 11px;
    margin-top: 5px;
}

.source-card {
    background: rgba(39, 64, 112, 0.55);
    border: 1px solid rgba(94, 167, 255, 0.35);
    border-radius: 10px;
    padding: 10px;
    margin-top: 8px;
    color: #dceaff;
    font-size: 11px;
}

/* ---------- BUTTONS ---------- */

.stButton > button {
    background: linear-gradient(100deg, #a34cf1, #397ff0);
    color: white !important;
    border: 1px solid rgba(193, 151, 255, 0.5);
    border-radius: 11px;
    font-weight: 700;
    min-height: 43px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #c6a0ff;
    box-shadow: 0 0 20px rgba(140, 90, 255, 0.35);
    transform: translateY(-1px);
}

/* ---------- INPUTS ---------- */

.stChatInput textarea {
    background: #192342 !important;
    color: white !important;
    border-radius: 12px !important;
}

[data-testid="stTextInput"] [data-baseweb="input"] {
    background: #ffffff !important;
    border: 1px solid rgba(124, 150, 225, 0.65) !important;
    border-radius: 12px !important;
}

[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 1px #a78bfa !important;
}

[data-testid="stTextInput"] input {
    background: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
    border: 0 !important;
    border-radius: 12px !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    opacity: 1 !important;
}

[data-testid="stTextInput"] button {
    background: #ffffff !important;
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    border: 0 !important;
}

[data-testid="stTextInput"] button * {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
}

[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stFormSubmitButton"] button:focus,
.stFormSubmitButton > button,
.stFormSubmitButton > button:hover,
.stFormSubmitButton > button:focus {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

[data-testid="stFileUploader"] {
    background: rgba(32, 44, 79, 0.55);
    border: 1px dashed rgba(130, 164, 255, 0.65);
    border-radius: 14px;
    padding: 8px;
}

[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
}

/* ---------- MISC ---------- */

hr {
    border-color: rgba(132, 150, 210, 0.20);
}

div[data-testid="stExpander"] {
    background: rgba(35, 47, 83, 0.45);
    border: 1px solid rgba(120, 145, 215, 0.25);
    border-radius: 12px;
}

.small-muted {
    color: #94a4c8;
    font-size: 11px;
}

.stCaption {
    color: #aab8d6;
}

/* PDF uploader text and button styling */

/* Make "Choose PDF files" text white */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label span,
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #ffffff !important;
}

/* Make the Upload button black */
[data-testid="stFileUploader"] button {
    background: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #555555 !important;
    border-radius: 8px !important;
}

/* Upload button hover */
[data-testid="stFileUploader"] button:hover {
    background: #222222 !important;
    border-color: #a78bfa !important;
    color: #ffffff !important;
}

/* Chat input placeholder color */
.stChatInput textarea::placeholder {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Typed text color */
.stChatInput textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* Chat input background */
.stChatInput textarea {
    background-color: #192342 !important;
}

/* ===================================
   FORCE ALL CHAT TEXT TO WHITE
=================================== */

/* User and assistant chat messages */
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong {
    color: #ffffff !important;
}

/* Chat input placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Chat input typed text */
[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* Loading and spinner messages */
[data-testid="stSpinner"],
[data-testid="stSpinner"] *,
.stSpinner,
.stSpinner * {
    color: #ffffff !important;
}

/* Status and informational messages */
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] *,
.stAlert,
.stAlert * {
    color: #ffffff !important;
}

/* Chat timestamps and captions */
[data-testid="stChatMessage"] small,
[data-testid="stChatMessage"] .stCaption {
    color: #ffffff !important;
}

/* Chat buttons and icons */
[data-testid="stChatMessage"] button {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "last_processing_results" not in st.session_state:
    st.session_state.last_processing_results = []


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_database_stats():

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collections = client.list_collections()
    if COLLECTION_NAME not in {collection.name for collection in collections}:
        return 0, {}

    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    document_counts = {}
    for metadata in metadatas:
        if metadata:
            filename = metadata.get("source", "Unknown")
            document_counts[filename] = document_counts.get(filename, 0) + 1

    return len(data.get("ids", [])), document_counts


def process_pdf_paths(file_paths):
    if not api_key:
        st.error("Enter a Google API key to generate PDF embeddings.")
        return

    processing_results = []
    progress_bar = st.progress(0)

    for index, file_path in enumerate(file_paths):
        try:
            with st.status(
                f"Processing {file_path.name}...",
                expanded=True
            ) as status:
                result = ingest_pdf(file_path=file_path, api_key=api_key)
                st.write(f"Pages: {result['pages']}")
                st.write(f"Chunks: {result['chunks']}")
                status.update(
                    label=f"Completed: {file_path.name}",
                    state="complete"
                )
            processing_results.append(result)
        except Exception as error:
            st.error(
                f"Could not process {file_path.name}: {sanitize_error(error)}"
            )

        progress_bar.progress((index + 1) / len(file_paths))

    st.session_state.last_processing_results = processing_results
    if processing_results:
        st.success(f"Successfully processed {len(processing_results)} PDF(s)!")
        st.rerun()


def format_sources(sources):

    if not sources:
        return

    st.markdown("**📎 Sources used**")

    columns = st.columns(min(len(sources), 3))

    for index, source in enumerate(sources):

        with columns[index % len(columns)]:

            filename = source.get("source", "Unknown PDF")
            page = source.get("page", "?")

            st.markdown(
                f"""
                <div class="source-card">
                    📄 <b>{filename}</b><br>
                    <span class="small-muted">Page {page}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


def sanitize_error(error):
    error_text = str(error)
    for secret in (api_key, openrouter_api_key):
        if secret:
            error_text = error_text.replace(secret, "[redacted]")
    return error_text


def clear_api_keys():
    st.session_state.pop("google_api_key", None)
    st.session_state.pop("openrouter_api_key", None)


def show_error(error):

    error_text = sanitize_error(error)
    normalized_error = error_text.lower()

    if "openrouter" in normalized_error and (
        "401" in normalized_error
        or "unauthorized" in normalized_error
        or "user not found" in normalized_error
    ):

        st.error(
            "OpenRouter rejected the API key (401 Unauthorized). Create a "
            "new key in your OpenRouter account, enter it in the API keys "
            "section, and retry. Do not paste the key into chat."
        )

    elif "ollama" in normalized_error and (
        "winerror 10061" in normalized_error
        or "connection refused" in normalized_error
        or "actively refused" in normalized_error
    ):

        st.error(
            "Cannot connect to Ollama at http://localhost:11434. Install "
            "Ollama for Windows and start the Ollama app (or run `ollama "
            "serve` in a terminal). Then download the selected model with "
            f"`ollama pull {selected_model}` and retry. You can verify it "
            "with `ollama list`."
        )

    elif "ollama" in normalized_error and (
        "not found" in normalized_error or "status code: 404" in normalized_error
    ):

        st.error(
            f"The Ollama model `{selected_model}` is not installed. "
            f"Install it with `ollama pull {selected_model}`, verify with "
            "`ollama list`, and retry."
        )

    elif "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

        st.error(
            "A provider quota or rate limit was reached during generation "
            "or Google-powered document retrieval. Check the relevant "
            "provider quota and try again. "
            f"Details: {error_text}"
        )

    elif "403" in error_text or "PERMISSION_DENIED" in error_text:

        st.error(
            "API permission denied. Check the selected provider's API key "
            "and the Google API key used for document embeddings. "
            f"Details: {error_text}"
        )

    else:

        st.error(f"Application error: {error_text}")


# ============================================================
# 5. SIDEBAR / BRANDING
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <div class="brand">📚 PDF RAG</div>
            <div class="brand">CHATBOT</div>
            <div class="brand-caption">
                Chat with your Documents ✨
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 Choose your AI provider")
    selected_provider = st.selectbox(
        "Generation provider",
        options=list(PROVIDER_MODELS),
        key="generation_provider",
    )
    selected_model = st.selectbox(
        "Model",
        options=PROVIDER_MODELS[selected_provider],
        index=PROVIDER_MODELS[selected_provider].index(
            DEFAULT_MODELS[selected_provider]
        ),
        key=f"generation_model_{selected_provider}",
    )

    provider_descriptions = {
        "Gemini": "Primary · Google cloud generation",
        "OpenRouter": "Backup · Cloud models, including free options",
        "Ollama": "Local · Runs on your laptop; no API key",
    }
    st.caption(provider_descriptions[selected_provider])

    auto_fallback = st.checkbox(
        "Enable automatic fallback",
        value=False,
        help=(
            "Try your selected provider first. If it fails, try the remaining "
            "providers in this order: Gemini, OpenRouter, Ollama."
        ),
    )

    st.markdown("### 🔐 API keys")
    st.caption(
        "Keys are kept in this Streamlit session only and are not written "
        "to project files. Only enter keys on an app instance you trust."
    )
    openrouter_required = (
        selected_provider == "OpenRouter" or auto_fallback
    )
    with st.form("api_keys"):
        st.text_input(
            "Google API key (PDF embeddings and retrieval)",
            type="password",
            placeholder="Paste your Google API key",
            key="google_api_key",
            help=(
                "Required for PDF embeddings and retrieval, regardless of "
                "which provider generates answers."
            ),
        )
        st.text_input(
            "OpenRouter API key "
            f"({'required' if openrouter_required else 'optional'})",
            type="password",
            placeholder="Paste your OpenRouter API key",
            key="openrouter_api_key",
            help=(
                "OpenRouter requires an API key, including for its free "
                "models. Needed when selected or enabled as a fallback."
            ),
        )
        keys_submitted = st.form_submit_button("Save API keys")

    api_key = st.session_state.get("google_api_key", "").strip()
    openrouter_api_key = st.session_state.get(
        "openrouter_api_key", ""
    ).strip()

    if keys_submitted:
        if api_key:
            st.success("API keys saved for this session.")
        else:
            st.warning("Enter a Google API key to process and search PDFs.")

    st.button(
        "Clear API keys",
        key="clear_api_keys",
        on_click=clear_api_keys,
    )

    if not api_key:
        st.warning(
            "Google is always needed to embed and search PDFs. The selected "
            "provider (Gemini, OpenRouter, or Ollama) generates the answer."
        )
    elif openrouter_required and not openrouter_api_key:
        st.warning(
            "Enter an OpenRouter API key above to use OpenRouter"
            + (
                " as a fallback."
                if auto_fallback and selected_provider != "OpenRouter"
                else "."
            )
            + " Google API access is still used for document embeddings."
        )
    elif selected_provider == "Ollama":
        st.info(
            "Install Ollama for Windows and keep the Ollama app running. "
            f"Download this model once with `ollama pull {selected_model}`. "
            "The local API should be available at localhost:11434."
        )

    st.divider()

    st.markdown("### 💬 Workspace")

    st.markdown(
        """
        <div class="document-card">
            💬 <b>AI Document Chat</b><br>
            <span class="small-muted">Ask questions from your PDFs</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### ⚡ Capabilities")

    st.markdown(
        """
        <div style="line-height:2.3; color:#d8e2ff; font-size:13px;">
            ✅ Multiple PDF uploads<br>
            ✅ Gemini, OpenRouter, or local Ollama answers<br>
            ✅ Source page references<br>
            ✅ Vector similarity search<br>
            ✅ Persistent document storage<br>
            ✅ LangGraph workflow
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🧠 Technology Stack")

    st.markdown(
        """
        <div class="small-muted" style="line-height:2;">
            🐍 Python<br>
            🔗 LangChain<br>
            🔀 LangGraph<br>
            ✨ Gemini · OpenRouter · Ollama<br>
            🗄️ ChromaDB<br>
            🎈 Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div class="small-muted">
            Built as a Generative AI RAG project.<br>
            Version 1.0
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 6. MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-title">
        💬 Chat with your PDFs using
        <span class="gradient-text">Gen AI</span>
    </div>
    <div class="subheading">
        Upload your documents, ask questions, and get
        contextual answers with source references.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 7. LIVE DATABASE STATISTICS
# ============================================================

total_chunks, document_counts = get_database_stats()
stored_files = sorted(document_counts)

stat1, stat2, stat3 = st.columns(3)

with stat1:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">📄 Indexed Documents</div>
            <div class="stat-value">{len(stored_files)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with stat2:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">🧩 Stored Chunks</div>
            <div class="stat-value">{total_chunks}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with stat3:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">🗄️ Vector Database</div>
            <div class="stat-value">ChromaDB</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# 8. THREE-PANEL MAIN WORKSPACE
# ============================================================

chat_column, document_column = st.columns(
    [0.69, 0.31],
    gap="medium"
)


# ============================================================
# 9. CHAT WORKSPACE
# ============================================================

with chat_column:

    st.markdown(
        """
        <div class="glass-panel">
            <div class="welcome-title">
                🤖 Hello! I'm your PDF RAG Assistant.
            </div>
            <div class="welcome-text">
                Upload one or more PDF documents and ask
                questions about their content. I'll search
                your documents and provide contextual answers.
            </div>
            <br>
            <span class="feature-pill">🔍 Search & Answer</span>
            <span class="feature-pill">📄 Page References</span>
            <span class="feature-pill">🧠 Context Retrieval</span>
            <span class="feature-pill">⚡ Gemini · OpenRouter · Ollama</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display previous messages
    for message in st.session_state.messages:

        avatar = "🤖" if message["role"] == "assistant" else "👤"

        with st.chat_message(
            message["role"],
            avatar=avatar
        ):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                if message.get("provider"):
                    st.caption(
                        f"Provider: {message['provider']} · "
                        f"Model: {message.get('model', 'unknown')}"
                    )
                    if (
                        message.get("requested_provider")
                        and message["provider"] != message["requested_provider"]
                    ):
                        st.info(
                            f"Fallback used: {message['requested_provider']} "
                            f"was unavailable; answered with {message['provider']}."
                        )

                format_sources(
                    message.get("sources", [])
                )

    # Chat input
    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:

        if not api_key:

            st.warning(
                "Enter a Google API key in the sidebar to search your PDFs."
            )

        elif total_chunks == 0:

            st.warning(
                "Upload and process at least one PDF before asking questions."
            )

        else:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user", avatar="👤"):
                st.markdown(question)

            try:

                with st.chat_message("assistant", avatar="🤖"):

                    with st.spinner(
                        "✨ Searching documents and thinking..."
                    ):

                        rag_graph = build_rag_graph(
                            api_key=api_key,
                            top_k=4,
                            provider=selected_provider,
                            model=selected_model,
                            provider_api_key=openrouter_api_key,
                            auto_fallback=auto_fallback,
                        )

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

                        answer = result["answer"]
                        sources = result["sources"]
                        active_provider = result["active_provider"]
                        active_model = result["active_model"]

                    st.markdown(answer)
                    st.caption(
                        f"Provider: {active_provider} · Model: "
                        f"{active_model or 'not used'}"
                    )
                    if active_provider not in {
                        selected_provider,
                        "Not used",
                    }:
                        st.info(
                            f"Fallback used: {selected_provider} was "
                            f"unavailable; answered with {active_provider}."
                        )

                    format_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "provider": active_provider,
                        "model": active_model,
                        "requested_provider": selected_provider,
                    }
                )

            except Exception as error:

                show_error(error)

                st.session_state.messages.pop()

# ============================================================
# 10. DOCUMENT MANAGEMENT PANEL
# ============================================================

with document_column:

    st.markdown(
        """
        <div class="glass-panel">
            <div class="panel-title">☁️ Upload Documents</div>
            <div class="panel-caption">
                Add PDF documents to your knowledge base.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if st.button("⚙️ Process Uploaded PDFs", use_container_width=True):

        if not uploaded_files:

            st.warning("Please select at least one PDF.")

        else:
            local_files = []
            for uploaded_file in uploaded_files:
                safe_name = Path(uploaded_file.name).name
                file_path = DOCUMENTS_DIR / safe_name
                file_path.write_bytes(uploaded_file.getvalue())
                local_files.append(file_path)

            process_pdf_paths(local_files)

    local_pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    unindexed_local_files = [
        file_path
        for file_path in local_pdf_files
        if file_path.name not in document_counts
    ]

    if local_pdf_files:
        st.markdown("**PDFs saved in `documents/`**")
        for file_path in local_pdf_files:
            chunk_count = document_counts.get(file_path.name, 0)
            status = (
                f"Indexed · {chunk_count} chunks"
                if chunk_count
                else "Not indexed yet"
            )
            st.caption(f"{file_path.name} — {status}")

        if unindexed_local_files and st.button(
            "⚙️ Process PDFs already in documents/",
            key="process_saved_pdfs",
            use_container_width=True,
        ):
            process_pdf_paths(unindexed_local_files)


    # ========================================================
    # 11. INDEXED DOCUMENTS
    # ========================================================

    st.write("")

    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="panel-title">
                🗂️ Indexed Documents ({len(stored_files)})
            </div>
            <div class="panel-caption">
                Documents currently stored in ChromaDB.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if stored_files:

        for filename in stored_files:

            chunk_count = document_counts.get(
                filename,
                0
            )

            st.markdown(
                f"""
                <div class="document-card">
                    <div class="document-name">
                        📕 {filename}
                    </div>
                    <div class="document-meta">
                        🧩 {chunk_count} chunks
                        &nbsp;&nbsp; <span style="color:#34d399;">● Indexed</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            """
            <div class="document-card">
                <div class="document-meta">
                    No indexed documents yet.<br>
                    Upload a PDF to get started.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 12. FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <div style="
        text-align:center;
        color:#7685aa;
        font-size:11px;
        padding:15px;
        border-top:1px solid rgba(130,150,220,0.15);
    ">
        Powered by LangChain · LangGraph · Gemini · OpenRouter · Ollama · ChromaDB
        <br>
        PDF RAG Assistant | Generative AI Project
    </div>
    """,
    unsafe_allow_html=True
)
