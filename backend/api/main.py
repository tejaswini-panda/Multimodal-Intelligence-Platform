"""
Backend API
Coursera Multimodal Intelligence Platform

Pipeline:
    Upload / Sample Assets -> Preprocessing -> Gemini Embeddings -> ChromaDB Retrieval
    -> Relevance Filter -> Gemini Grounded Synthesis -> Database Logging
    -> Human Review -> Dashboard Metrics

Endpoints:
    POST /api/ingest           (NEW - upload a file and index it)
    POST /api/query
    GET  /api/insights/{id}
    POST /api/review-feedback
    GET  /api/metrics
    GET  /

Run locally:
    uvicorn backend.api.main:app --reload

Swagger:
    http://127.0.0.1:8000/docs
"""

# ===================================================================
# IMPORTS
# ===================================================================

import sys
import os
import io
import csv
import json
import traceback

# ===================================================================
# PROJECT ROOT
# ===================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===================================================================
# FASTAPI / PYDANTIC
# ===================================================================

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

# ===================================================================
# DATABASE
# ===================================================================

from sqlalchemy.orm import Session

from backend.database.db import init_db, get_db

from backend.database.models import QueryLog, ReviewAction

# ===================================================================
# AI PIPELINE
# ===================================================================

from ai.preprocessing.chunker import preprocess_all_assets

from ai.retrieval.retriever import (
    retrieve,
    index_segments,
    index_uploaded_chunks,
    collection_count,
)

from ai.synthesis.synthesizer import synthesize_insight

# ===================================================================
# FASTAPI APPLICATION
# ===================================================================

app = FastAPI(
    title="Coursera Multimodal Intelligence Platform",
    version="1.1.0",
    description=(
        "Backend API for unified retrieval, grounded Gemini synthesis, "
        "human review feedback, and operational metrics."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================
# STARTUP
# ===================================================================


@app.on_event("startup")
def on_startup():
    """Initialize database tables when the application starts."""

    try:
        init_db()
        print("Database initialized successfully.")

    except Exception as exc:
        print("Database initialization failed:")
        print(str(exc))
        traceback.print_exc()
        raise


# ===================================================================
# REQUEST SCHEMAS
# ===================================================================


class QueryRequest(BaseModel):
    """Request body for the unified query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question about learner friction.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of evidence pieces to retrieve.",
    )


class ReviewFeedbackRequest(BaseModel):
    """Request body for human review feedback."""

    query_log_id: str = Field(..., min_length=1)

    decision: str = Field(
        ...,
        description="Review decision: approved, rejected, or needs_revision.",
    )

    reviewer_note: str = Field(
        default="",
        description="Optional reviewer comment.",
    )


# ===================================================================
# CONFIG + UPLOAD HELPERS (NEW)
# ===================================================================

# Chunks farther than this distance are treated as NOT relevant.
# Tune it on Render via an environment variable named MAX_DISTANCE.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.0"))

MAX_CHUNKS_PER_FILE = 200
ALLOWED_MODALITIES = {"video", "slide", "quiz", "discussion"}


def _extract_text(filename: str, data: bytes) -> str:
    """Convert an uploaded file into plain text."""

    name = filename.lower()

    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    text = data.decode("utf-8", errors="ignore")

    if name.endswith(".csv"):
        rows = csv.DictReader(io.StringIO(text))
        return "\n\n".join(
            " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            for row in rows
        )

    if name.endswith(".json"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return text

        items = obj if isinstance(obj, list) else [obj]
        parts = []

        for item in items:
            if isinstance(item, dict):
                parts.append(" | ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                parts.append(str(item))

        return "\n\n".join(parts)

    return text  # .txt and anything else


def _chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Split text into paragraph-based chunks of roughly max_chars."""

    paragraphs = [
        p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()
    ]

    chunks, buffer = [], ""

    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{para}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            while len(para) > max_chars:
                chunks.append(para[:max_chars])
                para = para[max_chars:]
            buffer = para

    if buffer:
        chunks.append(buffer)

    return chunks


# ===================================================================
# HELPER: ENSURE CHROMADB HAS DATA
# ===================================================================


def ensure_index_ready() -> int:
    """
    Make sure ChromaDB contains searchable evidence.

    Render uses a separate runtime environment from the local machine,
    so the local ChromaDB data may not exist after deployment.

    If the collection is empty, index the sample assets automatically.

    Returns:
        Number of searchable segments.
    """

    try:
        current_count = collection_count()

        print(f"Current ChromaDB collection count: {current_count}")

        if current_count > 0:
            return current_count

        print("ChromaDB collection is empty.")
        print("Starting automatic indexing of sample assets...")

        segments = preprocess_all_assets()

        if not segments:
            raise RuntimeError(
                "No sample segments were generated during preprocessing."
            )

        print(f"Preprocessed {len(segments)} segments.")

        index_segments(segments, clear_existing=False)

        final_count = collection_count()

        print(
            "ChromaDB indexing completed. "
            f"Collection now contains {final_count} segments."
        )

        if final_count == 0:
            raise RuntimeError(
                "Indexing completed but ChromaDB is still empty."
            )

        return final_count

    except Exception as exc:
        print("ChromaDB initialization/indexing failed:")
        print(str(exc))
        traceback.print_exc()

        raise RuntimeError(
            f"Vector database initialization failed: {exc}"
        ) from exc


# ===================================================================
# POST /api/ingest (NEW)
# ===================================================================


@app.post("/api/ingest")
def ingest_file(
    file: UploadFile = File(...),
    modality: str = Form("video"),
    source_title: str = Form(""),
):
    """Upload a file, chunk it, embed it, and store it in ChromaDB."""

    modality = modality.strip().lower()

    if modality not in ALLOWED_MODALITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid modality. Use one of: {sorted(ALLOWED_MODALITIES)}",
        )

    data = file.file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = _extract_text(file.filename or "upload.txt", data)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read file: {exc}",
        ) from exc

    chunks_text = _chunk_text(text)[:MAX_CHUNKS_PER_FILE]

    if not chunks_text:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in the file.",
        )

    title = source_title.strip() or file.filename

    chunks = [
        {
            "text": t,
            "source_id": file.filename,
            "modality": modality,
            "source_title": title,
        }
        for t in chunks_text
    ]

    try:
        indexed = index_uploaded_chunks(chunks)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {exc}",
        ) from exc

    return {
        "status": "ingested",
        "filename": file.filename,
        "modality": modality,
        "chunks_indexed": indexed,
        "collection_total": collection_count(),
    }


# ===================================================================
# POST /api/query
# ===================================================================


@app.post("/api/query")
def run_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    """
    Main unified query endpoint.

    Workflow:
        1. Validate query
        2. Ensure ChromaDB is ready
        3. Retrieve evidence
        4. Drop irrelevant evidence (distance filter)
        5. Generate grounded Gemini insight
        6. Store query and result in database
        7. Return evidence and insight
    """

    # 1. VALIDATE QUERY

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 2. ENSURE CHROMADB IS READY

    try:
        available_count = ensure_index_ready()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Vector database error: {exc}",
        )

    # 3. SAFE TOP-K

    actual_top_k = min(request.top_k, available_count)

    if actual_top_k <= 0:
        raise HTTPException(
            status_code=500,
            detail="No searchable evidence is available.",
        )

    # 4. RETRIEVAL + RELEVANCE FILTER

    try:
        print(f"Running retrieval for query: '{query}' with top_k={actual_top_k}")

        evidence = retrieve(query, top_k=actual_top_k)

        if evidence is None:
            evidence = []

        print(f"Retrieved {len(evidence)} evidence pieces.")

        # Debug: use these values in Render logs to tune MAX_DISTANCE
        print("Evidence distances:", [round(e["distance"], 3) for e in evidence])

        # Keep only relevant evidence
        evidence = [e for e in evidence if e["distance"] <= MAX_DISTANCE]

        # Nothing relevant -> do NOT call Gemini
        if not evidence:
            return {
                "status": "success",
                "query_log_id": None,
                "query": query,
                "evidence_count": 0,
                "evidence": [],
                "insight": {
                    "insight": None,
                    "evidence_used": [],
                    "confidence": "low",
                    "confidence_reason": (
                        "No relevant evidence found for this question in the "
                        "uploaded content. Upload relevant material in the "
                        "Upload Data tab and try again."
                    ),
                    "recommendation": None,
                    "error": "insufficient_evidence",
                },
            }

    except Exception as exc:
        print("Retrieval failed:")
        print(str(exc))
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    # 5. GEMINI SYNTHESIS

    try:
        print("Starting Gemini synthesis...")

        insight = synthesize_insight(query, evidence)

        print("Gemini synthesis completed.")

    except Exception as exc:
        print("Gemini synthesis failed:")
        print(str(exc))
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Gemini synthesis failed: {exc}",
        ) from exc

    # 6. DATABASE LOGGING

    try:
        confidence = (
            insight.get("confidence", "unknown")
            if isinstance(insight, dict)
            else "unknown"
        )

        log_entry = QueryLog(
            query_text=query,
            evidence_json=json.dumps(evidence, default=str),
            insight_json=json.dumps(insight, default=str),
            confidence=confidence,
        )

        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

    except Exception as exc:
        db.rollback()

        print("Database logging failed:")
        print(str(exc))
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Database logging failed: {exc}",
        ) from exc

    # 7. RESPONSE

    return {
        "status": "success",
        "query_log_id": log_entry.id,
        "query": query,
        "evidence_count": len(evidence),
        "evidence": evidence,
        "insight": insight,
    }


# ===================================================================
# GET /api/insights/{query_log_id}
# ===================================================================


@app.get("/api/insights/{query_log_id}")
def get_insight(
    query_log_id: str,
    db: Session = Depends(get_db),
):
    """Fetch a previously generated insight, evidence, and review history."""

    log_entry = (
        db.query(QueryLog)
        .filter(QueryLog.id == query_log_id)
        .first()
    )

    if not log_entry:
        raise HTTPException(status_code=404, detail="Insight not found.")

    try:
        evidence = json.loads(log_entry.evidence_json or "[]")
        insight = json.loads(log_entry.insight_json or "{}")

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Stored insight data is corrupted: {exc}",
        )

    reviews = []

    for review in log_entry.reviews:
        reviews.append(
            {
                "decision": review.decision,
                "note": review.reviewer_note,
                "created_at": (
                    review.created_at.isoformat()
                    if review.created_at
                    else None
                ),
            }
        )

    return {
        "query_log_id": log_entry.id,
        "query_text": log_entry.query_text,
        "evidence": evidence,
        "insight": insight,
        "created_at": (
            log_entry.created_at.isoformat()
            if log_entry.created_at
            else None
        ),
        "reviews": reviews,
    }


# ===================================================================
# POST /api/review-feedback
# ===================================================================


@app.post("/api/review-feedback")
def submit_review(
    request: ReviewFeedbackRequest,
    db: Session = Depends(get_db),
):
    """
    Record human review feedback.

    Allowed decisions: approved, rejected, needs_revision
    """

    log_entry = (
        db.query(QueryLog)
        .filter(QueryLog.id == request.query_log_id)
        .first()
    )

    if not log_entry:
        raise HTTPException(status_code=404, detail="Query log not found.")

    decision = request.decision.strip().lower()

    allowed_decisions = {"approved", "rejected", "needs_revision"}

    if decision not in allowed_decisions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid decision value. "
                "Use: approved, rejected, or needs_revision."
            ),
        )

    try:
        review = ReviewAction(
            query_log_id=request.query_log_id,
            decision=decision,
            reviewer_note=request.reviewer_note.strip(),
        )

        db.add(review)
        db.commit()

    except Exception as exc:
        db.rollback()

        print("Review logging failed:")
        print(str(exc))
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Could not save review: {exc}",
        ) from exc

    return {
        "status": "recorded",
        "query_log_id": request.query_log_id,
        "decision": decision,
    }


# ===================================================================
# GET /api/metrics
# ===================================================================


@app.get("/api/metrics")
def get_metrics(db: Session = Depends(get_db)):
    """Return basic metrics for the dashboard."""

    try:
        total_queries = db.query(QueryLog).count()

        total_reviews = db.query(ReviewAction).count()

        confidence_breakdown = {}

        for level in ("high", "medium", "low"):
            confidence_breakdown[level] = (
                db.query(QueryLog)
                .filter(QueryLog.confidence == level)
                .count()
            )

        decision_breakdown = {}

        for decision in ("approved", "rejected", "needs_revision"):
            decision_breakdown[decision] = (
                db.query(ReviewAction)
                .filter(ReviewAction.decision == decision)
                .count()
            )

        return {
            "total_queries": total_queries,
            "total_reviews": total_reviews,
            "confidence_breakdown": confidence_breakdown,
            "decision_breakdown": decision_breakdown,
        }

    except Exception as exc:
        print("Metrics query failed:")
        print(str(exc))
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Could not load metrics: {exc}",
        ) from exc


# ===================================================================
# GET /
# ===================================================================


@app.get("/")
def root():
    """Health-check endpoint."""

    return {
        "status": "ok",
        "message": "Coursera Multimodal Intelligence Platform API is running.",
    }