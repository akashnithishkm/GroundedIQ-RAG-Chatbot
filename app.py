"""Polished Streamlit interface for the local document RAG assistant."""

import logging, os, time
from pathlib import Path
import streamlit as st
from rag_app.config import (
    BASE_DIR,
    DOCUMENTS_DIR,
    MAX_UPLOAD_BYTES,
    PROVIDER_MODELS,
    DEFAULT_MODELS,
)
from rag_app.history import (
    add_message,
    create_conversation,
    delete_conversation,
    get_messages,
    list_conversations,
    rename_conversation,
)
from rag_app.ingestion import ingest_file
from rag_app.security import safe_error, safe_filename
from rag_app.storage import (
    clear_index,
    delete_chroma_database,
    delete_document,
    indexed_documents,
)
from rag_app.workflow import NO_EVIDENCE, build_rag_graph

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("rag_app")
st.set_page_config(
    page_title="GroundedIQ RAG Chatbot · Conversational Document Intelligence",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#15233a;--muted:#66758a;--line:#e5ebf2;--blue:#315be8;--wash:#f5f8fc}
.stApp{background:linear-gradient(145deg,#f9fbfe 0%,#f2f6fb 58%,#edf3fa 100%);color:var(--ink);font-family:'DM Sans',sans-serif}
.block-container{padding-top:2rem;max-width:1440px}.hero{padding:1.55rem 1.8rem;border:1px solid #e6ebf4;border-radius:23px;background:linear-gradient(115deg,#fff 4%,#f4f7ff 70%,#f0f5ff);box-shadow:0 12px 38px #2946750b;margin-bottom:1.2rem}
.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#526ce0;font-weight:700}.hero h1{font:800 clamp(28px,3vw,41px) 'Manrope',sans-serif;letter-spacing:-1.7px;margin:7px 0 5px;color:#16243d}.hero p{margin:0;color:#65748b;font-size:15px}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e7edf5}.stButton>button,.stFormSubmitButton>button{border-radius:11px;border:1px solid #dbe3ee;font-weight:600;min-height:42px}.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:#315be8;border-color:#315be8;color:white}
[data-testid="stChatMessage"]{background:white;border:1px solid #e7edf3;border-radius:16px;padding:12px 16px;box-shadow:0 4px 16px #203d6608}.stChatInput textarea{border-radius:14px}.metric-label{color:#76859a;font-size:12px}.metric-value{font:700 24px Manrope;color:#182944}.stCaption{color:#738198}.section-note{padding:12px 14px;border-radius:12px;background:#f4f7fc;color:#65748a;font-size:13px;border:1px solid #e7edf5}
</style>""",
    unsafe_allow_html=True,
)
DOCUMENTS_DIR.mkdir(exist_ok=True)


def read_secret(name):
    value = os.getenv(name, "")
    if not value:
        try:
            value = st.secrets.get(name, "")
        except Exception:
            pass
    return value or ""


with st.sidebar:
    st.markdown("## :material/auto_awesome: GroundedIQ RAG Chatbot")
    st.caption("Private document intelligence")
    st.divider()
    conversations = list_conversations()
    if (
        "conversation_id" not in st.session_state
        or st.session_state.conversation_id not in {r[0] for r in conversations}
    ):
        st.session_state.conversation_id = create_conversation()
        st.rerun()
    if st.button(":material/add: New chat", width="stretch", type="primary"):
        st.session_state.conversation_id = create_conversation()
        st.rerun()
    conversations = list_conversations()
    selected = st.selectbox(
        "Conversations",
        options=[r[0] for r in conversations],
        format_func=lambda cid: next(
            (r[1] for r in conversations if r[0] == cid), "Conversation"
        ),
        index=next(
            (
                i
                for i, r in enumerate(conversations)
                if r[0] == st.session_state.conversation_id
            ),
            0,
        ),
    )
    if selected != st.session_state.conversation_id:
        st.session_state.conversation_id = selected
        st.rerun()
    with st.expander("Conversation actions"):
        current_name = next((r[1] for r in conversations if r[0] == selected), "")
        new_name = st.text_input(
            "Rename conversation", value=current_name, key="rename_value"
        )
        if st.button("Save name", key="rename_chat"):
            rename_conversation(selected, new_name)
            st.rerun()
        if st.button("Delete this conversation", key="delete_chat"):
            st.session_state.confirm_delete_chat = True
        if st.session_state.get("confirm_delete_chat"):
            st.warning("This permanently deletes this conversation and its messages.")
            a, b = st.columns(2)
            if a.button("Confirm delete", key="yes_delete_chat"):
                delete_conversation(selected)
                st.session_state.conversation_id = create_conversation()
                st.session_state.confirm_delete_chat = False
                st.rerun()
            if b.button("Cancel", key="no_delete_chat"):
                st.session_state.confirm_delete_chat = False
                st.rerun()
    st.divider()
    st.subheader("Answer engine")
    provider = st.selectbox("Provider", list(PROVIDER_MODELS), key="provider")
    model = st.selectbox(
        "Model",
        PROVIDER_MODELS[provider],
        index=PROVIDER_MODELS[provider].index(DEFAULT_MODELS[provider]),
        key=f"model_{provider}",
    )
    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.1,
        0.05,
        help="Lower values produce more consistent document-grounded answers.",
    )
    google_key = st.text_input(
        "Google API key · embeddings and Gemini",
        value=read_secret("GOOGLE_API_KEY"),
        type="password",
        help="Session-only input; environment variable or Streamlit secrets also work.",
    )
    openrouter_key = st.text_input(
        "OpenRouter API key", value=read_secret("OPENROUTER_API_KEY"), type="password"
    )
    st.caption("Secrets are not stored in conversation history or project files.")
    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Passages to retrieve", 1, 12, 4)
    retrieval_mode = st.selectbox("Search mode", ["Similarity", "MMR", "Hybrid"])
    st.caption(
        {
            "Similarity": "Ranks by embedding similarity. Fast and precise for direct questions.",
            "MMR": "Balances similarity with passage diversity.",
            "Hybrid": "Combines vector and keyword ranks with reciprocal-rank fusion.",
        }[retrieval_mode]
    )
    st.caption(
        "Embeddings use Google gemini-embedding-001. Changing embedding models requires a separate collection and full reindex."
    )

total_chunks, files = indexed_documents()
st.markdown(
    '<div class="hero"><div class="eyebrow">Your knowledge, in context</div><h1>Document intelligence, grounded.</h1><p>Ask across your files, follow the thread, and inspect the passages behind every answer.</p></div>',
    unsafe_allow_html=True,
)
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Indexed files", len(files))
with m2:
    st.metric("Searchable passages", total_chunks)
with m3:
    st.metric("Search strategy", retrieval_mode)
chat_tab, library_tab, eval_tab = st.tabs(["Chat", "Library & indexing", "Evaluation"])

with chat_tab:
    msgs = get_messages(st.session_state.conversation_id)
    st.markdown("### :material/smart_toy: GroundedIQ assistant")
    st.caption(
        "Ask naturally, explore follow-ups, and open the source cards to verify each response."
    )
    if not msgs:
        with st.container(border=True):
            st.markdown(
                "### Welcome to GroundedIQ RAG Chatbot. What would you like to learn?"
            )
            st.write(
                "I can search your documents, summarize key ideas, compare details, and keep track of follow-up questions."
            )
            st.caption(
                "Answers are grounded in your indexed files, with source passages you can inspect."
            )
        if files:
            st.markdown("**Try a starter question**")
            suggestions = [
                "Summarize the key points",
                "Find the important dates and numbers",
                "What does this document say about…?",
            ]
            cols = st.columns(3)
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(suggestion, key=f"chat_starter_{i}", width="stretch"):
                    st.session_state.pending_prompt = suggestion
                    st.rerun()
        else:
            st.info(
                "Add a PDF, Word file, text file, or CSV in **Library & indexing** to start chatting."
            )
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("provider"):
                    st.caption(
                        f"{msg['provider']} · {msg.get('model','')} · {msg.get('latency',0):.1f}s"
                    )
                if msg.get("sources"):
                    with st.expander(f"Sources · {len(msg['sources'])}"):
                        for i, src in enumerate(msg["sources"], 1):
                            location = (
                                f"Page {src['page']}"
                                if src.get("page")
                                else (
                                    f"Row {src['row']}"
                                    if src.get("row")
                                    else src.get("section") or "Passage"
                                )
                            )
                            st.markdown(
                                f"**[{i}] {src['source']}** · {location}\n\n> {src['snippet']}"
                            )
    typed_question = st.chat_input("Ask GroundedIQ about your indexed documents…")
    question = st.session_state.pop("pending_prompt", None) or typed_question
    if question:
        add_message(st.session_state.conversation_id, "user", question)
        if not files:
            add_message(st.session_state.conversation_id, "assistant", NO_EVIDENCE)
            st.rerun()
        if not google_key:
            st.error(
                "Add a Google API key in the sidebar to search and embed documents."
            )
        elif provider == "OpenRouter" and not openrouter_key:
            st.error("Add an OpenRouter API key in the sidebar to use that provider.")
        else:
            start = time.perf_counter()
            try:
                with st.spinner(
                    "GroundedIQ is searching your documents and preparing a grounded answer…"
                ):
                    graph = build_rag_graph(
                        google_key,
                        top_k,
                        provider,
                        model,
                        openrouter_key,
                        temperature,
                        retrieval_mode,
                    )
                    result = graph.invoke(
                        {
                            "question": question,
                            "history": msgs,
                            "provider": provider,
                            "model": model,
                            "retrieval_mode": retrieval_mode,
                        }
                    )
                elapsed = time.perf_counter() - start
                add_message(
                    st.session_state.conversation_id,
                    "assistant",
                    result["answer"],
                    result.get("sources"),
                    result.get("active_provider"),
                    result.get("active_model"),
                    elapsed,
                )
                st.rerun()
            except Exception as exc:
                log.warning("RAG request failed (%s)", type(exc).__name__)
                st.error(
                    f"Request failed: {safe_error(exc,[google_key,openrouter_key])}"
                )

with library_tab:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Add documents")
        uploads = st.file_uploader(
            "PDF, TXT, DOCX, or CSV · up to 25 MB each",
            type=["pdf", "txt", "docx", "csv"],
            accept_multiple_files=True,
        )
        if st.button("Index selected files", type="primary", disabled=not uploads):
            if not google_key:
                st.error("Add a Google API key in the sidebar to create embeddings.")
            else:
                for upload in uploads:
                    if upload.size > MAX_UPLOAD_BYTES:
                        st.error(f"{upload.name}: file exceeds 25 MB.")
                        continue
                    try:
                        name = safe_filename(upload.name)
                        target = (DOCUMENTS_DIR / name).resolve()
                        if target.parent != DOCUMENTS_DIR.resolve():
                            raise ValueError("Unsafe filename.")
                        target.write_bytes(upload.getvalue())
                        result = ingest_file(target, google_key)
                        (st.info if result["status"] == "duplicate" else st.success)(
                            f"{result['filename']}: {result['status']} · {result['chunks']} chunks"
                        )
                    except Exception as exc:
                        st.error(
                            f"{upload.name}: {safe_error(exc,[google_key,openrouter_key])}"
                        )
                st.rerun()
        st.caption(
            "Uploads are saved in the project documents folder. Reindexing replaces that filename's vectors after successful extraction."
        )
        local = sorted(
            p
            for p in DOCUMENTS_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".docx", ".csv"}
        )
        unindexed = [p for p in local if p.name not in files]
        if unindexed and st.button(
            "Index files already in documents/", width="stretch"
        ):
            if not google_key:
                st.error("Add a Google API key first.")
            else:
                for p in unindexed:
                    try:
                        st.success(f"{p.name}: {ingest_file(p,google_key)['status']}")
                    except Exception as exc:
                        st.error(f"{p.name}: {safe_error(exc,[google_key])}")
                st.rerun()
    with right:
        st.subheader(f"Indexed documents · {len(files)}")
        if not files:
            st.info(
                "No indexed documents yet. Upload a file or index an existing local document."
            )
        for name, meta in files.items():
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(
                    f"{meta['chunks']} chunks · {meta['embedding_model']} · indexed"
                )
                c1, c2 = st.columns(2)
                if c1.button("Reindex", key=f"reindex_{name}"):
                    path = DOCUMENTS_DIR / Path(name).name
                    if not google_key:
                        st.error("Add a Google API key first.")
                    elif path.exists():
                        try:
                            st.success(str(ingest_file(path, google_key, replace=True)))
                            st.rerun()
                        except Exception as exc:
                            st.error(safe_error(exc, [google_key]))
                    else:
                        st.warning(
                            "Local upload is missing. Upload the file again before reindexing."
                        )
                if c2.button("Remove from index", key=f"rm_index_{name}"):
                    st.session_state.pending_index_delete = name
                if st.session_state.get("pending_index_delete") == name:
                    st.warning(
                        f"This removes vectors for {name}; the uploaded file stays in documents/."
                    )
                    a, b = st.columns(2)
                    if a.button("Confirm removal", key=f"confirm_rm_{name}"):
                        delete_document(name, google_key)
                        st.session_state.pop("pending_index_delete")
                        st.rerun()
                    if b.button("Cancel", key=f"cancel_rm_{name}"):
                        st.session_state.pop("pending_index_delete")
                        st.rerun()
                if name in {p.name for p in local} and st.button(
                    "Delete uploaded file only", key=f"rm_upload_{name}"
                ):
                    st.session_state.pending_upload_delete = name
                if st.session_state.get("pending_upload_delete") == name:
                    st.warning(
                        f"This permanently deletes documents/{name}. Indexed vectors remain unchanged."
                    )
                    a, b = st.columns(2)
                    if a.button("Confirm file deletion", key=f"yes_upload_{name}"):
                        from rag_app.storage import remove_upload

                        remove_upload(name)
                        st.session_state.pop("pending_upload_delete")
                        st.rerun()
                    if b.button("Cancel", key=f"no_upload_{name}"):
                        st.session_state.pop("pending_upload_delete")
                        st.rerun()
    st.divider()
    st.subheader("Start fresh")
    st.caption(
        "These actions only affect the configured documents/ folder and chroma_db/ database. Conversation history and API keys remain untouched."
    )
    with st.expander("Review fresh-start actions"):
        files_local = [p.name for p in local]
        st.markdown(
            f"**Remove local uploads:** deletes {len(files_local)} file(s) from documents/: {', '.join(files_local) if files_local else 'none'}. The vector index remains unchanged."
        )
        if st.button("Remove all uploaded local documents", key="fresh_uploads"):
            st.session_state.fresh_action = "uploads"
        st.markdown(
            f"**Clear indexed vectors:** deletes all {total_chunks} passages from Chroma collection `pdf_documents`. Uploaded files and the Chroma database folder remain unchanged."
        )
        if st.button("Clear all indexed vectors", key="fresh_index"):
            st.session_state.fresh_action = "index"
        st.markdown(
            f"**Delete Chroma database folder:** deletes all files inside `chroma_db/` (including vectors and database metadata). Uploaded files remain unchanged."
        )
        if st.button("Delete Chroma database files", key="fresh_database"):
            st.session_state.fresh_action = "database"
        st.markdown(
            f"**Remove uploads and clear index:** deletes the {len(files_local)} listed local upload(s) and clears all {total_chunks} indexed passages, while keeping the Chroma database folder."
        )
        if st.button("Remove uploads and clear index", key="fresh_both"):
            st.session_state.fresh_action = "both"
        st.markdown(
            f"**Delete uploads and Chroma database:** deletes the {len(files_local)} listed local upload(s) and the entire configured `chroma_db/` folder."
        )
        if st.button("Delete uploads and Chroma database", key="fresh_full"):
            st.session_state.fresh_action = "full"
        action = st.session_state.get("fresh_action")
        if action:
            labels = {
                "uploads": "local uploads only",
                "index": "all indexed vectors only (the Chroma database folder stays)",
                "database": "the entire chroma_db/ folder and all files in it",
                "both": "local uploads and all indexed vectors (the Chroma database folder stays)",
                "full": "local uploads and the entire chroma_db/ folder",
            }
            st.warning(
                f"Confirm: this will permanently delete {labels[action]}. Conversations and API keys will remain."
            )
            confirm = st.text_input(
                "Type DELETE to confirm", key=f"fresh_confirm_{action}"
            )
            a, b = st.columns(2)
            if a.button(
                "Confirm start fresh",
                key=f"fresh_yes_{action}",
                disabled=confirm != "DELETE",
            ):
                if action in {"uploads", "both", "full"}:
                    for filename in files_local:
                        try:
                            from rag_app.storage import remove_upload

                            remove_upload(filename)
                        except OSError:
                            log.warning("Could not remove configured upload")
                if action in {"index", "both"}:
                    clear_index()
                if action in {"database", "full"}:
                    delete_chroma_database()
                st.session_state.pop("fresh_action")
                st.rerun()
            if b.button("Cancel", key=f"fresh_no_{action}"):
                st.session_state.pop("fresh_action")
                st.rerun()

with eval_tab:
    st.subheader("Retrieval evaluation")
    st.caption(
        "Add questions and expected source/page labels for your own indexed files in rag_app/evaluation.py. Retrieval metrics do not measure answer accuracy."
    )
    from rag_app.evaluation import run_evaluation

    if st.button("Run retrieval evaluation", disabled=not files or not google_key):
        try:
            st.session_state.eval_results = run_evaluation(
                google_key, top_k, retrieval_mode
            )
        except Exception as exc:
            st.error(safe_error(exc, [google_key, openrouter_key]))
    if st.session_state.get("eval_results"):
        st.dataframe(st.session_state.eval_results, width="stretch", hide_index=True)
        st.caption(
            "Review the retrieved passages yourself. Recall@k and reciprocal rank measure source retrieval only, not answer accuracy."
        )
    else:
        st.info(
            "No evaluation cases are configured. Add your own questions and expected sources in rag_app/evaluation.py."
        )
    st.caption(
        "Optional reranking is not enabled in this build. Scores are not compared across embedding models."
    )

st.divider()
st.caption(
    "GroundedIQ RAG Chatbot · LangGraph workflow · Chroma vector search · local SQLite conversation history"
)
