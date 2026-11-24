"""
app/api/upload.py
-----------------
FastAPI router for PDF document upload, extraction, section-aware chunking,
FastEmbed ONNX vector generation, and session-isolated Qdrant upsert.
Memory-guarded with 10MB file caps and explicit garbage collection.
"""

import gc
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import config
from app.schemas import UploadResponse
from app.services.chunking import chunk_document
from app.services.embeddings import generate_embeddings
from app.services.ingestion import load_pdf
from app.services.vector_store import upsert_chunks_to_qdrant

router = APIRouter(prefix="", tags=["upload"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_index_document(
    file: UploadFile = File(..., description="PDF document file to index"),
    session_id: str = Form(..., description="Session UUID for tenant document isolation"),
):
    """Ingests a PDF, extracts pages, chunks text, generates embeddings, and upserts to Qdrant."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents are supported.",
        )

    clean_session_id = session_id.strip()
    if not clean_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'session_id'. Each upload must be bound to a session.",
        )

    max_bytes = config.max_upload_size_mb * 1024 * 1024

    # Save to temporary file while checking size limit to avoid memory bloat
    temp_dir = Path(tempfile.gettempdir()) / "docusense_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{clean_session_id}_{file.filename}"

    file_size = 0
    try:
        with open(temp_path, "wb") as f_out:
            while chunk := await file.read(1024 * 1024):  # 1 MB buffer
                file_size += len(chunk)
                if file_size > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {config.max_upload_size_mb} MB for free tier.",
                    )
                f_out.write(chunk)

        # 1. Ingest PDF & extract pages
        pages = load_pdf(str(temp_path))
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No readable text extracted from PDF. The document may be empty or image-only.",
            )

        page_count = len(pages)

        # 2. Section-aware text chunking
        chunks = chunk_document(pages)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text chunks generated from the uploaded document.",
            )

        # Free raw page structures from memory immediately
        del pages
        gc.collect()

        # 3. Dense vector embeddings (Cloudflare Workers AI 384D GPU)
        chunk_texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(chunk_texts)

        # 4. Batched upsert to Qdrant Cloud with session isolation
        upserted_count = upsert_chunks_to_qdrant(
            chunks=chunks,
            embeddings=embeddings,
            session_id=clean_session_id,
            batch_size=16,
        )

        # 5. Free all chunk and vector lists and force GC
        del chunk_texts, chunks, embeddings
        gc.collect()

        return UploadResponse(
            filename=file.filename,
            extracted_pages=page_count,
            indexed_chunks=upserted_count,
            session_id=clean_session_id,
            status="indexed",
            message=f"Indexed {upserted_count} chunks from '{file.filename}' into Qdrant.",
        )

    finally:
        # Clean up temporary upload file on disk
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
