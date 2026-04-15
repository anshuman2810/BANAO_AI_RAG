from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    job_id: str
    status: JobStatus


class JobInfo(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    chunks_indexed: int = 0


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    content_type: str
    uploaded_at: datetime
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    min_similarity: float = Field(default=0.05, ge=-1.0, le=1.0)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    text: str
    similarity: float


class QueryMetrics(BaseModel):
    latency_ms: float
    best_similarity: float | None
    retrieved_chunks: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    metrics: QueryMetrics
    answer_provider: str

