from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.document_loader import SUPPORTED_EXTENSIONS
from app.embeddings import HashingEmbeddingModel
from app.jobs import JobRegistry, ingest_document
from app.llm import generate_answer
from app.models import (
    DocumentInfo,
    JobInfo,
    QueryMetrics,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UploadResponse,
)
from app.rate_limit import InMemoryRateLimiter
from app.vector_store import LocalVectorStore

settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0")
embedding_model = HashingEmbeddingModel(dimensions=settings.embedding_dimensions)
vector_store = LocalVectorStore(settings.vector_dir)
jobs = JobRegistry()
rate_limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)


def get_rate_limiter() -> InMemoryRateLimiter:
    return rate_limiter


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/documents",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(get_rate_limiter())],
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_settings: Settings = Depends(get_settings),
) -> UploadResponse:
    original_name = file.filename or "uploaded_document"
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported types: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()
    max_bytes = current_settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is larger than {current_settings.max_upload_mb} MB",
        )

    document_id = uuid4().hex
    stored_path = current_settings.upload_dir / f"{document_id}{extension}"
    stored_path.write_bytes(content)

    job = jobs.create(document_id=document_id, filename=original_name)
    uploaded_at = datetime.now(timezone.utc)
    background_tasks.add_task(
        ingest_document,
        job.job_id,
        document_id,
        stored_path,
        original_name,
        file.content_type or "application/octet-stream",
        uploaded_at,
        jobs,
        embedding_model,
        vector_store,
        current_settings.chunk_size,
        current_settings.chunk_overlap,
    )

    return UploadResponse(
        document_id=document_id,
        filename=original_name,
        job_id=job.job_id,
        status=job.status,
    )


@app.get("/jobs/{job_id}", response_model=JobInfo, dependencies=[Depends(get_rate_limiter())])
def get_job(job_id: str) -> JobInfo:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@app.get("/documents", response_model=list[DocumentInfo], dependencies=[Depends(get_rate_limiter())])
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**item) for item in vector_store.list_documents()]


@app.delete("/documents/{document_id}", dependencies=[Depends(get_rate_limiter())])
def delete_document(
    document_id: str,
    current_settings: Settings = Depends(get_settings),
) -> dict[str, int | str]:
    deleted_chunks = vector_store.delete_document(document_id)
    for path in current_settings.upload_dir.glob(f"{document_id}.*"):
        path.unlink(missing_ok=True)
    if deleted_chunks == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return {"document_id": document_id, "deleted_chunks": deleted_chunks}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(get_rate_limiter())])
def query_documents(
    request: QueryRequest,
    current_settings: Settings = Depends(get_settings),
) -> QueryResponse:
    started = perf_counter()
    query_embedding = embedding_model.embed([request.question])[0]
    retrieved = vector_store.search(query_embedding, request.top_k)
    filtered = [
        item
        for item in retrieved
        if item["similarity"] >= request.min_similarity
    ]
    answer, provider = generate_answer(request.question, filtered, current_settings)
    latency_ms = (perf_counter() - started) * 1000
    best_similarity = filtered[0]["similarity"] if filtered else None

    return QueryResponse(
        answer=answer,
        answer_provider=provider,
        sources=[
            SourceChunk(
                document_id=item["document_id"],
                filename=item["filename"],
                chunk_id=item["chunk_id"],
                text=item["text"],
                similarity=item["similarity"],
            )
            for item in filtered
        ],
        metrics=QueryMetrics(
            latency_ms=round(latency_ms, 2),
            best_similarity=best_similarity,
            retrieved_chunks=len(filtered),
        ),
    )

