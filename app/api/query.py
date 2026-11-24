"""
app/api/query.py
----------------
FastAPI router for grounded RAG query execution.
Provides both full JSON responses (/query) and real-time token streaming (/query/stream).
Orchestrates:
  1. Two-stage retrieval: Qdrant Cosine Vector Search + FlashRank Cross-Encoder Reranking
  2. Multi-tenant session filtering
  3. Grounded context formatting & system prompt construction
  4. Resilient Groq LLM inference with session rate limiting
  5. Per-stage latency metrics and source citation metadata
"""

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from groq import Groq

from app.api.rate_limit import check_and_consume_quota, get_remaining_queries
from app.config import config
from app.schemas import Citation, QueryRequest, QueryResponse, TimingsBreakdown
from app.services.embeddings import generate_query_embedding
from app.services.retrieval.reranker import rerank_chunks
from app.services.retrieval.retriever import format_retrieved_context
from app.services.vector_store import search_qdrant
from prompts.templates import build_grounded_prompt

router = APIRouter(prefix="", tags=["query"])


def _retrieve_and_rerank(
    query: str,
    session_id: Optional[str],
    top_k: int,
    enable_reranking: bool,
) -> tuple[List[Dict[str, Any]], float, float]:
    """Synchronous CPU-bound retrieval and reranking routine.
    Returns (chunks, retrieval_ms, rerank_ms).
    """
    t0 = time.perf_counter()

    # Stage 1: Generate 384D query embedding & search Qdrant
    query_vector = generate_query_embedding(query)
    candidate_k = max(top_k * 3, 15) if enable_reranking else top_k

    candidates = search_qdrant(
        query_vector=query_vector,
        session_id=session_id,
        top_k=candidate_k,
    )
    retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Stage 2: Cross-Encoder Reranking (FlashRank ONNX)
    t_rerank = time.perf_counter()
    if enable_reranking and len(candidates) > 1:
        final_chunks = rerank_chunks(query=query, chunks=candidates, top_k=top_k)
    else:
        final_chunks = candidates[:top_k]
    rerank_ms = round((time.perf_counter() - t_rerank) * 1000.0, 2)

    return final_chunks, retrieval_ms, rerank_ms


def _get_active_groq_key(x_groq_api_key: Optional[str]) -> tuple[str, bool]:
    """Returns (api_key, is_custom_user_key)."""
    custom_key = (x_groq_api_key or "").strip()
    if custom_key:
        return custom_key, True
    server_key = config.groq_api_key.strip()
    if not server_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No Groq API key configured on server. Please supply 'X-Groq-API-Key' header.",
        )
    return server_key, False


@router.post("/query", response_model=QueryResponse)
async def query_rag_endpoint(
    req: QueryRequest,
    x_groq_api_key: Optional[str] = Header(None, alias="X-Groq-API-Key"),
):
    """Executes full RAG pipeline and returns grounded answer with citations and timings."""
    clean_q = req.query.strip()
    if not clean_q:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

    # 1. Resolve API key and apply rate limiting if using demo key
    api_key, is_custom_key = _get_active_groq_key(x_groq_api_key)
    remaining_budget = None

    if not is_custom_key:
        if not req.session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'session_id'. Required for demo query rate tracking.",
            )
        allowed, remaining_budget = check_and_consume_quota(req.session_id)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Demo quota reached for this session ({config.rate_limit_demo_queries} queries max). "
                    "Please enter your own free Groq API key in the sidebar to continue querying."
                ),
            )

    t_total_start = time.perf_counter()

    # 2. Execute retrieval & reranking in thread pool to prevent blocking event loop
    loop = asyncio.get_running_loop()
    chunks, retrieval_ms, rerank_ms = await loop.run_in_executor(
        None,
        _retrieve_and_rerank,
        clean_q,
        req.session_id,
        req.top_k,
        req.enable_reranking,
    )

    # 3. Format prompt context
    context_str = format_retrieved_context(chunks)
    prompt = build_grounded_prompt(clean_q, context_str)

    # 4. LLM Generation via Groq API
    t_gen_start = time.perf_counter()
    model_name = req.model or config.groq_model

    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer_text = chat_completion.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq API error: {str(e)[:150]}",
        )
    generation_ms = round((time.perf_counter() - t_gen_start) * 1000.0, 2)
    total_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)

    # 5. Format citations
    citations = [
        Citation(
            source_id=idx,
            filename=c.get("filename", "unknown.pdf"),
            page_number=c.get("page_number", 1),
            similarity_score=c.get("similarity_score", 0.0),
            distance=c.get("distance", 0.0),
            chunk_id=c.get("chunk_id", ""),
            text=c.get("text", ""),
            is_table=c.get("is_table", False),
        )
        for idx, c in enumerate(chunks, 1)
    ]

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        timings=TimingsBreakdown(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        ),
        model=model_name,
        session_id=req.session_id,
        remaining_queries=remaining_budget,
    )


@router.post("/query/stream")
async def query_rag_stream_endpoint(
    req: QueryRequest,
    x_groq_api_key: Optional[str] = Header(None, alias="X-Groq-API-Key"),
):
    """Streams response tokens using Server-Sent Events (SSE).
    Emits token chunks and concludes with a metadata event containing citations and timings.
    """
    clean_q = req.query.strip()
    if not clean_q:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

    api_key, is_custom_key = _get_active_groq_key(x_groq_api_key)
    remaining_budget = None

    if not is_custom_key:
        if not req.session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'session_id'. Required for demo query rate tracking.",
            )
        allowed, remaining_budget = check_and_consume_quota(req.session_id)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Demo quota reached for this session ({config.rate_limit_demo_queries} queries max). "
                    "Please enter your own free Groq API key in the sidebar to continue querying."
                ),
            )

    t_total_start = time.perf_counter()
    loop = asyncio.get_running_loop()
    chunks, retrieval_ms, rerank_ms = await loop.run_in_executor(
        None,
        _retrieve_and_rerank,
        clean_q,
        req.session_id,
        req.top_k,
        req.enable_reranking,
    )

    context_str = format_retrieved_context(chunks)
    prompt = build_grounded_prompt(clean_q, context_str)
    model_name = req.model or config.groq_model

    client = Groq(api_key=api_key)

    async def event_generator() -> AsyncGenerator[str, None]:
        t_gen_start = time.perf_counter()
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.0,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)[:150]})}\n\n"
            return

        generation_ms = round((time.perf_counter() - t_gen_start) * 1000.0, 2)
        total_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)

        citations_data = [
            {
                "source_id": idx,
                "filename": c.get("filename", "unknown.pdf"),
                "page_number": c.get("page_number", 1),
                "similarity_score": c.get("similarity_score", 0.0),
                "distance": c.get("distance", 0.0),
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", ""),
                "is_table": c.get("is_table", False),
            }
            for idx, c in enumerate(chunks, 1)
        ]

        meta_payload = {
            "type": "done",
            "citations": citations_data,
            "timings": {
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
            "model": model_name,
            "remaining_queries": remaining_budget,
        }
        yield f"data: {json.dumps(meta_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
