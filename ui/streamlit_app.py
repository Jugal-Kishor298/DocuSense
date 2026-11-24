import os
import sys
import time
from pathlib import Path
import streamlit as st

# Add workspace root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.api_client import DocuSenseAPIClient

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

    /* Status Pill */
    .status-pill-online {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 9999px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .status-pill-offline {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 9999px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }

    /* Upload Dropzone */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] p,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] label {
        color: rgba(203,213,225,0.55) !important;
        font-size: 0.82rem !important;
    }

    /* Source Citation Cards */
    .source-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        color: #E2E8F0;
    }
    .source-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-weight: 600;
        color: #93C5FD;
        font-size: 0.82rem;
    }
    .source-badge {
        background: rgba(96, 165, 250, 0.15);
        color: #60A5FA;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.75rem;
    }

    /* Empty State Card */
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
# Client & Session State Initialization
# ==============================================================================
def resolve_backend_url() -> str:
    """Resolves backend API URL from Streamlit secrets or environment variables."""
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"]
    except Exception:
        pass
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


if "backend_url" not in st.session_state:
    st.session_state.backend_url = resolve_backend_url()

api_client = DocuSenseAPIClient(base_url=st.session_state.backend_url)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "upload_processed" not in st.session_state:
        st.session_state.upload_processed = set()
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False
    if "backend_online" not in st.session_state:
        st.session_state.backend_online = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "remaining_queries" not in st.session_state:
        st.session_state.remaining_queries = 8
    if "max_queries" not in st.session_state:
        st.session_state.max_queries = 8
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "enable_reranking" not in st.session_state:
        st.session_state.enable_reranking = True
    if "custom_groq_key" not in st.session_state:
        st.session_state.custom_groq_key = ""
    if "groq_model" not in st.session_state:
        st.session_state.groq_model = "openai/gpt-oss-120b"
    if "last_metadata" not in st.session_state:
        st.session_state.last_metadata = None


init_session_state()


# ==============================================================================
# Cold-Start Wakeup & Health Verification
# ==============================================================================
def ensure_backend_connected():
    """Checks backend health; if sleeping (Render free tier), displays wake-up progress banner."""
    if not st.session_state.backend_online:
        is_healthy, _ = api_client.check_health(timeout=2.0)
        if is_healthy:
            st.session_state.backend_online = True
        else:
            # Render free-tier cold-start handler
            with st.status("🚀 Connecting to DocuSense Backend (Render free tier waking from sleep ~45s)...", expanded=True) as status_box:
                progress_bar = st.progress(0)
                for status in api_client.poll_for_wakeup(max_seconds=70, interval=2.5):
                    pct = min(100, int((status["elapsed"] / 50.0) * 100))
                    progress_bar.progress(pct)
                    status_box.write(f"⏳ Waiting for backend container... ({status['elapsed']}s elapsed)")
                    if status.get("online"):
                        status_box.update(label="✅ DocuSense Backend is online and ready!", state="complete", expanded=False)
                        st.session_state.backend_online = True
                        break
                    if status.get("error"):
                        status_box.update(label="❌ Backend wake-up timed out. Please refresh in a moment.", state="error")
                        break

    # Initialize backend session if not present
    if st.session_state.backend_online and not st.session_state.session_id:
        try:
            sess_info = api_client.create_session()
            st.session_state.session_id = sess_info["session_id"]
            st.session_state.remaining_queries = sess_info.get("remaining_queries", 8)
            st.session_state.max_queries = sess_info.get("max_queries", 8)
        except Exception as e:
            st.sidebar.error(f"Failed to create session: {e}")


ensure_backend_connected()


# ==============================================================================
# Helper Functions
# ==============================================================================
def render_inspection_pane(sources, timings=None, latency=None):
    """Renders the 3-tab source citation, relevance metadata, and latency breakdown pane."""
    if not sources:
        return

    total_time = latency if latency is not None else (timings.get("total_ms", 0) / 1000.0 if timings else 0.0)

    with st.expander("📚 Source Citations & Inspection", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Retrieved Context", "Relevance & Metadata", "Latency Breakdown"])

        with tab1:
            for idx, src in enumerate(sources, 1):
                sim_score = src.get("similarity_score", 0.0)
                sim_pct = f"{sim_score * 100:.1f}%" if sim_score <= 1.0 else f"{sim_score:.2f}"
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-card-header">
                            <span>[Source {idx}] {src.get('filename', 'unknown')} — Page {src.get('page_number', 1)}</span>
                            <span class="source-badge">{sim_pct} Match</span>
                        </div>
                        <div>{src.get('text', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab2:
            metadata_table = [
                {
                    "Source": f"Source {idx}",
                    "Document": src.get("filename", "unknown"),
                    "Page": f"Page {src.get('page_number', 1)}",
                    "Similarity": f"{src.get('similarity_score', 0.0) * 100:.1f}%",
                    "Distance": src.get("distance", 0.0),
                }
                for idx, src in enumerate(sources, 1)
            ]
            st.dataframe(metadata_table, use_container_width=True, hide_index=True)

        with tab3:
            if timings:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retrieval (Qdrant)", f"{timings.get('retrieval_ms', 0):.1f} ms")
                col2.metric("Rerank (FlashRank)", f"{timings.get('rerank_ms', 0):.1f} ms")
                col3.metric("LLM Inference (Groq)", f"{timings.get('generation_ms', 0):.1f} ms")
                col4.metric("Total Latency", f"{timings.get('total_ms', 0):.1f} ms")
            else:
                st.metric(label="Total Latency", value=f"{total_time:.2f}s")


# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown("### 📄 DocuSense AI")

    # Backend Connection Status Badge
    if st.session_state.backend_online:
        st.markdown('<div class="status-pill-online">● Backend Online (Qdrant Cloud)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill-offline">● Backend Offline / Connecting</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 1. Document Upload
    st.markdown("#### Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs to Index",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload digital PDFs (up to 10 MB per file). Text is extracted, embedded, and indexed into Qdrant Cloud.",
    )

    if uploaded_files and st.session_state.session_id:
        for u_file in uploaded_files:
            if u_file.name not in st.session_state.upload_processed:
                st.info(f"Uploading & Indexing: **{u_file.name}**...")
                try:
                    with st.spinner("Processing PDF on backend (Chunking + FastEmbed 384D ONNX)..."):
                        result = api_client.upload_document(
                            session_id=st.session_state.session_id,
                            filename=u_file.name,
                            file_bytes=u_file.getvalue(),
                        )
                    st.success(f"✅ Indexed {result.get('indexed_chunks', 0)} chunks!")
                    st.session_state.indexed_files.append(u_file.name)
                    st.session_state.upload_processed.add(u_file.name)
                    time.sleep(0.4)
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    # Display session file summary
    n_files = len(st.session_state.indexed_files)
    if n_files == 0:
        st.caption("No documents indexed in this session yet.")
    else:
        st.caption(f"📁 {n_files} {'document' if n_files == 1 else 'documents'} indexed in session.")

    st.markdown("---")

    # 2. Quota & Rate Limit Tracker
    st.markdown("#### 🎯 Session Demo Quota")
    quota_left = st.session_state.remaining_queries
    quota_max = st.session_state.max_queries
    quota_pct = max(0.0, min(1.0, quota_left / float(quota_max))) if quota_max > 0 else 0.0

    st.progress(quota_pct)
    st.caption(f"**{quota_left} / {quota_max}** free queries remaining.")

    if quota_left <= 0:
        st.warning("⚠️ Demo quota exhausted. Enter your own Groq API Key below to continue querying.")

    st.markdown("---")

    # 3. Model & API Key Configuration
    st.markdown("#### 🤖 Language Model")
    st.session_state.groq_model = st.text_input(
        "Groq Model ID",
        value=st.session_state.groq_model,
        placeholder="e.g. openai/gpt-oss-120b",
        help="Target Groq model ID (default: openai/gpt-oss-120b).",
    ).strip() or "openai/gpt-oss-120b"

    st.session_state.custom_groq_key = st.text_input(
        "Groq API Key (Optional)",
        value=st.session_state.custom_groq_key,
        type="password",
        placeholder="gsk_...",
        help="Optional: Enter your own Groq key from console.groq.com to bypass the demo query quota.",
    ).strip()

    st.markdown("---")

    # 4. RAG Hyperparameters
    with st.expander("⚙️ RAG Hyperparameters", expanded=False):
        st.session_state.top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Number of top reranked chunks supplied to LLM context.",
        )
        st.session_state.enable_reranking = st.checkbox(
            "Enable FlashRank Cross-Encoder",
            value=st.session_state.enable_reranking,
            help="Re-scores candidates using FlashRank TinyBERT joint attention.",
        )
        st.markdown("**Vector Store**: `Qdrant Cloud`")
        st.markdown("**Embedding**: `FastEmbed (384D ONNX)`")

    st.markdown("---")

    # 5. Session Actions
    st.markdown("#### 🧹 Session Actions")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_btn2:
        if st.button("Reset DB", use_container_width=True, type="secondary"):
            st.session_state.confirm_reset = True

    if st.session_state.confirm_reset:
        st.warning("⚠️ Purge session vectors from Qdrant?")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Confirm", type="primary", use_container_width=True):
                if st.session_state.session_id:
                    try:
                        api_client.delete_session(st.session_state.session_id)
                    except Exception:
                        pass
                st.session_state.messages = []
                st.session_state.indexed_files = []
                st.session_state.upload_processed = set()
                st.session_state.session_id = None
                st.session_state.confirm_reset = False
                ensure_backend_connected()
                st.success("Session reset!")
                time.sleep(0.5)
                st.rerun()
        with col_c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()

    st.markdown("---")
    st.caption(f"DocuSense v2.0 · Decoupled RAG · {st.session_state.groq_model}")


# ==============================================================================
# Main Chat Area
# ==============================================================================
st.markdown('<div class="docusense-title">DocuSense</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="docusense-subtitle">Ask anything about your uploaded documents (Grounded RAG with Citations)</div>',
    unsafe_allow_html=True,
)

# Empty state
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state-card">
            <div class="empty-state-title">Get started 📄</div>
            <div class="empty-state-desc">Upload one or more PDFs from the sidebar, then ask any question about their content.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_inspection_pane(
                sources=message["sources"],
                timings=message.get("timings"),
                latency=message.get("latency"),
            )

# Chat Input & Streaming Execution
prompt_input = st.chat_input("Ask anything about your documents...")

if prompt_input:
    if not st.session_state.backend_online:
        st.error("Backend is currently offline. Please wait for connection or refresh.")
        st.stop()

    if not st.session_state.session_id:
        ensure_backend_connected()

    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Render streaming assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        collected_tokens = []
        captured_metadata = {}

        def token_stream_generator():
            start_stream_time = time.time()
            for event in api_client.stream_query(
                query=prompt_input,
                session_id=st.session_state.session_id,
                top_k=st.session_state.top_k,
                enable_reranking=st.session_state.enable_reranking,
                custom_groq_key=st.session_state.custom_groq_key or None,
                model=st.session_state.groq_model,
            ):
                event_type = event.get("type")
                if event_type == "token":
                    content = event.get("content", "")
                    collected_tokens.append(content)
                    yield content
                elif event_type == "done":
                    captured_metadata.update(event)
                    if "remaining_queries" in event and event["remaining_queries"] is not None:
                        st.session_state.remaining_queries = event["remaining_queries"]
                elif event_type == "error":
                    err_msg = f"\n\n⚠️ **Error**: {event.get('content', 'Query failed')}"
                    collected_tokens.append(err_msg)
                    yield err_msg

        with st.spinner("Retrieving from Qdrant Cloud & reranking..."):
            full_response = response_placeholder.write_stream(token_stream_generator())

        sources = captured_metadata.get("citations", [])
        timings = captured_metadata.get("timings", {})

        # Persist response to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "timings": timings,
            }
        )

        # Render citations pane directly below assistant answer
        if sources:
            render_inspection_pane(sources, timings=timings)

        # Update remaining query count in sidebar
        if "remaining_queries" in captured_metadata and captured_metadata["remaining_queries"] is not None:
            st.session_state.remaining_queries = captured_metadata["remaining_queries"]
