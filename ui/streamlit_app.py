import sys
import time
from pathlib import Path
import streamlit as st

# Add workspace root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import config
from app.services.ingestion import load_pdf
from app.services.chunking import chunk_document
from app.services.embeddings import generate_embeddings
from app.services.vector_store import (
    add_chunks_to_vector_store,
    get_collection_stats,
    reset_collection,
)
from app.services.retrieval import retrieve_relevant_chunks
from app.services.generation import generate_answer_stream


# ==============================================================================
# Page Configuration & Modern Dark UI Styling
# ==============================================================================
st.set_page_config(
    page_title="DocuSense — Document QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 950px;
    }

    /* Header */
    .docusense-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60A5FA 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .docusense-subtitle {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }

    /* ── Upload Dropzone: flatten card, remove inner box shadow ── */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    /* The dashed drop-area itself — keep a very subtle border, no fill */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    /* "Upload PDFs to Index" label + "200MB per file" helper — mute to 55% */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] p,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] label {
        color: rgba(203,213,225,0.55) !important;
        font-size: 0.82rem !important;
    }
    /* Document icon block: solid white -> semi-transparent tint */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] svg,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span[data-testid="stIconMaterial"] {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 6px !important;
        padding: 4px !important;
        color: rgba(148,163,184,0.7) !important;
    }
    /* Uploaded file item rows: dark + semi-transparent, no stacked box */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
        background: rgba(15,23,42,0.6) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
    }
    /* File name text inside rows */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
        color: rgba(226,232,240,0.85) !important;
        font-size: 0.82rem !important;
    }
    /* X / clear button: invisible until hover */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] button {
        opacity: 0.25 !important;
        transition: opacity 0.2s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] button:hover {
        opacity: 0.9 !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #3B82F6, #8B5CF6);
    }

    /* Source Citation Cards */
    .source-card {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 0.90rem;
        color: #E2E8F0;
        line-height: 1.5;
    }
    .source-card-header {
        font-weight: 600;
        color: #60A5FA;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .source-badge {
        background-color: #334155;
        color: #93C5FD;
        font-size: 0.78rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 500;
    }

    /* Empty State */
    .empty-state-card {
        background-color: #1E293B;
        border: 1px dashed #334155;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .empty-state-title {
        color: #F8FAFC;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .empty-state-desc {
        color: #94A3B8;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Session State Initialization
# ==============================================================================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = config.top_k
    if "relevance_pct" not in st.session_state:
        st.session_state.relevance_pct = int(round((1.0 - config.similarity_threshold) * 100))
    if "upload_processed" not in st.session_state:
        st.session_state.upload_processed = set()
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "ollama"
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = config.ollama_model
    if "groq_model" not in st.session_state:
        st.session_state.groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    if "groq_api_key" not in st.session_state:
        import os
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")


init_session_state()


# ==============================================================================
# Helper Functions
# ==============================================================================
def render_inspection_pane(sources, latency: float):
    """Renders the 3-tab source citation and inspection pane."""
    if not sources:
        return

    with st.expander("📚 Source Citations & Inspection", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Retrieved Context", "Relevance & Metadata", "Latency"])

        with tab1:
            for idx, src in enumerate(sources, 1):
                sim_pct = f"{src.get('similarity_score', 0.0) * 100:.1f}%"
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-card-header">
                            <span>[Source {idx}] {src['filename']} — Page {src['page_number']}</span>
                            <span class="source-badge">{sim_pct} Match</span>
                        </div>
                        <div>{src['text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab2:
            metadata_table = [
                {
                    "Source": f"Source {idx}",
                    "Document": src["filename"],
                    "Page": f"Page {src['page_number']}",
                    "Relevance Match": f"{src.get('similarity_score', 0.0) * 100:.1f}%",
                }
                for idx, src in enumerate(sources, 1)
            ]
            st.dataframe(metadata_table, use_container_width=True, hide_index=True)

            with st.expander("🛠️ Raw Chunk IDs & Cosine Metrics"):
                for idx, src in enumerate(sources, 1):
                    st.json(
                        {
                            "source_index": idx,
                            "chunk_id": src["chunk_id"],
                            "cosine_distance": src.get("distance", 0.0),
                            "similarity_score": src.get("similarity_score", 0.0),
                        }
                    )

        with tab3:
            st.metric(label="Total Latency", value=f"{latency:.2f}s")


def process_uploaded_pdf(uploaded_file, progress_bar, status_text):
    """Runs the full ingestion -> chunking -> embedding -> vector store pipeline."""
    temp_dir = ROOT_DIR / "data" / "raw"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / uploaded_file.name

    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except (PermissionError, OSError):
        if not temp_path.exists():
            alt_path = temp_dir / f"upload_{uploaded_file.name}"
            with open(alt_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_path = alt_path

    status_text.markdown("🔄 **Step 1/4**: Ingesting PDF & extracting pages (300 DPI OCR)...")
    progress_bar.progress(25)
    pages = load_pdf(str(temp_path), dpi=300)

    status_text.markdown("✂️ **Step 2/4**: Section-aware text chunking...")
    progress_bar.progress(50)
    chunks = chunk_document(pages)

    status_text.markdown("🧬 **Step 3/4**: Generating FastEmbed 384D vector embeddings...")
    progress_bar.progress(75)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)

    status_text.markdown("🗄️ **Step 4/4**: Upserting vectors to ChromaDB...")
    progress_bar.progress(95)
    add_chunks_to_vector_store(chunks, embeddings)

    progress_bar.progress(100)
    status_text.markdown(f"✅ **Indexed {len(chunks)} chunks** from `{uploaded_file.name}`.")
    time.sleep(0.5)

    st.session_state.indexed_files.append(uploaded_file.name)
    st.session_state.upload_processed.add(uploaded_file.name)


# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown("### 📄 DocuSense AI")
    st.markdown("---")

    # 1. Document Upload
    st.markdown("#### Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs to Index",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload digital or scanned PDFs. 300 DPI OCR fallback runs automatically for scanned pages.",
    )

    if uploaded_files:
        for u_file in uploaded_files:
            if u_file.name not in st.session_state.upload_processed:
                st.info(f"Processing: **{u_file.name}**")
                prog_bar = st.progress(0)
                stat_txt = st.empty()
                process_uploaded_pdf(u_file, prog_bar, stat_txt)
                prog_bar.empty()
                stat_txt.empty()
                st.rerun()

    stats = get_collection_stats()
    n_chunks = stats["total_vector_count"]
    n_files  = len(st.session_state.indexed_files)
    if n_files == 0:
        st.caption("No documents indexed yet.")
    else:
        st.caption(f"{n_chunks:,} chunks · {n_files} {'file' if n_files == 1 else 'files'}")

    st.markdown("---")

    # 2. Model Selector
    st.markdown("#### 🤖 Language Model")
    provider_choice = st.radio(
        "Backend",
        options=["🦙 Local Ollama", "⚡ Groq API"],
        index=0 if st.session_state.llm_provider == "ollama" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.llm_provider = "ollama" if "Ollama" in provider_choice else "groq"

    if st.session_state.llm_provider == "ollama":
        st.session_state.ollama_model = st.text_input(
            "Ollama Model Name",
            value=st.session_state.ollama_model,
            placeholder="e.g. llama3.2:3b",
            help="Exact model tag as shown by `ollama list`. Make sure it's pulled locally.",
        ).strip()
        active_model = st.session_state.ollama_model or config.ollama_model
    else:
        st.session_state.groq_api_key = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
            help="Your Groq Cloud API key. Get one at console.groq.com.",
        ).strip()
        st.session_state.groq_model = st.text_input(
            "Groq Model Name",
            value=st.session_state.groq_model,
            placeholder="e.g. meta-llama/llama-4-scout-17b-16e-instruct",
            help="Exact model ID from console.groq.com/docs/models (e.g. qwen-qwq-32b, meta-llama/llama-4-maverick-17b-128e-instruct).",
        ).strip()
        active_model = st.session_state.groq_model or "meta-llama/llama-4-scout-17b-16e-instruct"

    st.markdown("---")

    # 3. RAG Hyperparameters
    with st.expander("⚙️ RAG Hyperparameters", expanded=False):
        st.session_state.top_k = st.slider(
            "Top-K Retrieved Chunks",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Number of top relevant text chunks to retrieve from ChromaDB.",
        )

        st.session_state.relevance_pct = st.slider(
            "Minimum Relevance Match %",
            min_value=0,
            max_value=100,
            value=st.session_state.relevance_pct,
            step=5,
            help="Minimum similarity % required for a chunk to be included in context.",
        )

        computed_threshold = round(1.0 - (st.session_state.relevance_pct / 100.0), 2)
        st.caption(f"🎯 *Cosine distance threshold*: `d <= {computed_threshold}`")
        st.markdown("**Embedding**: `FastEmbed` (384D ONNX)")

    st.markdown("---")

    # 4. System Actions
    st.markdown("#### 🧹 Actions")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_btn2:
        if st.button("Reset DB", use_container_width=True, type="secondary"):
            st.session_state.confirm_reset = True

    if st.session_state.confirm_reset:
        st.warning("⚠️ Delete all vectors & reset workspace?")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Confirm Wipe", type="primary", use_container_width=True):
                reset_collection()
                st.session_state.messages = []
                st.session_state.indexed_files = []
                st.session_state.upload_processed = set()
                st.session_state.confirm_reset = False
                st.success("Database reset!")
                time.sleep(0.5)
                st.rerun()
        with col_c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()

    st.markdown("---")
    provider_label = "Ollama (local)" if st.session_state.llm_provider == "ollama" else "Groq API"
    st.caption(f"DocuSense v1.0 · {provider_label} · {active_model}")


# ==============================================================================
# Main Chat Area
# ==============================================================================
st.markdown('<div class="docusense-title">DocuSense</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="docusense-subtitle">Ask anything about your uploaded documents</div>',
    unsafe_allow_html=True,
)

# Empty state
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state-card">
            <div class="empty-state-title">Get started 📄</div>
            <div class="empty-state-desc">Upload one or more PDFs from the sidebar, then ask any question.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_inspection_pane(message["sources"], message.get("latency", 0.0))

# Chat input
prompt_input = st.chat_input("Ask anything about your documents...")

if prompt_input:
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    active_distance_threshold = round(1.0 - (st.session_state.relevance_pct / 100.0), 2)

    with st.chat_message("assistant"):
        start_time = time.time()

        # Retrieval — fast vector search, wrapped in a spinner
        with st.spinner("Thinking..."):
            retrieved_chunks = retrieve_relevant_chunks(
                query=prompt_input,
                top_k=st.session_state.top_k,
                threshold=active_distance_threshold,
            )

        # Stream LLM response — spinner disappears on first token
        response_placeholder = st.empty()
        full_response = response_placeholder.write_stream(
            generate_answer_stream(
                query=prompt_input,
                retrieved_chunks=retrieved_chunks,
                model_name=active_model,
                provider=st.session_state.llm_provider,
                groq_api_key=st.session_state.get("groq_api_key", "") or None,
            )
        )
        end_time = time.time()
        total_latency = end_time - start_time

        # Persist to session
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_chunks,
                "latency": total_latency,
            }
        )

        # Citations below answer
        if retrieved_chunks:
            render_inspection_pane(retrieved_chunks, total_latency)
