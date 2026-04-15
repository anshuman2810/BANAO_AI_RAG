from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.chunking import chunk_text
from app.document_loader import load_document_text
from app.embeddings import HashingEmbeddingModel
from app.models import JobInfo, JobStatus
from app.vector_store import LocalVectorStore


class JobRegistry:
    def __init__(self) -> None:
        self.jobs: dict[str, JobInfo] = {}

    def create(self, document_id: str, filename: str) -> JobInfo:
        now = datetime.now(timezone.utc)
        job = JobInfo(
            job_id=uuid4().hex,
            document_id=document_id,
            filename=filename,
            status=JobStatus.queued,
            created_at=now,
            updated_at=now,
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobInfo | None:
        return self.jobs.get(job_id)

    def update(
        self,
        job_id: str,
        status: JobStatus,
        error: str | None = None,
        chunks_indexed: int | None = None,
    ) -> None:
        job = self.jobs[job_id]
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        job.error = error
        if chunks_indexed is not None:
            job.chunks_indexed = chunks_indexed


async def ingest_document(
    job_id: str,
    document_id: str,
    path: Path,
    filename: str,
    content_type: str,
    uploaded_at: datetime,
    registry: JobRegistry,
    embedding_model: HashingEmbeddingModel,
    vector_store: LocalVectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    try:
        registry.update(job_id, JobStatus.running)
        text = load_document_text(path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        embeddings = embedding_model.embed(chunks)
        metadata = [
            {
                "document_id": document_id,
                "filename": filename,
                "content_type": content_type,
                "uploaded_at": uploaded_at.isoformat(),
                "chunk_id": f"{document_id}:{index}",
                "chunk_index": index,
                "text": chunk,
            }
            for index, chunk in enumerate(chunks)
        ]
        vector_store.add(embeddings, metadata)
        registry.update(job_id, JobStatus.completed, chunks_indexed=len(chunks))
    except Exception as exc:
        registry.update(job_id, JobStatus.failed, error=str(exc))

