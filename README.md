# RAG-Based Question Answering System

A compact FastAPI backend for uploading documents, ingesting them in a background job, retrieving relevant chunks, and answering questions with a Retrieval-Augmented Generation workflow.

The project intentionally avoids full RAG frameworks so the chunking, retrieval, ingestion jobs, request validation, rate limiting, and metrics are visible in the code.

## Features

- Upload `.txt` and `.pdf` documents
- Background ingestion jobs
- Text chunking with overlap
- Embedding generation
- Local vector store persisted on disk
- Similarity search over stored chunks
- LLM answer generation with Ollama
- Extractive fallback when no LLM provider is available
- Pydantic request and response models
- Basic in-memory rate limiting
- Query latency and similarity score tracking

## Tech Stack

- FastAPI
- Pydantic
- NumPy
- scikit-learn hashing embeddings
- pypdf for PDF extraction
- Ollama local model for generated answers

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file to configure Ollama:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
```

If Ollama is not running, the API still works and returns an extractive answer using the retrieved chunks.

To use Ollama locally:

```bash
ollama pull llama3.2
ollama serve
```

## Run

```bash
uvicorn app.main:app --reload
```

Open:

- API docs: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

## API Usage

### Upload a document

```bash
curl -X POST "http://127.0.0.1:8000/documents" ^
  -F "file=@sample_docs/sample.txt"
```

### Check ingestion job status

```bash
curl "http://127.0.0.1:8000/jobs/{job_id}"
```

### Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is the document about?\",\"top_k\":3}"
```

The response includes `answer`, `sources`, `metrics`, and `answer_provider`.

## Architecture

The architecture diagram is available at:

- `docs/architecture.mmd`
- `docs/architecture.drawio`

## Project Notes

Detailed screening explanations are in `docs/EXPLANATION.md`.

## Repository Link

Add your GitHub link here after pushing the project:

```text
https://github.com/anshuman2810/BANAO_AI_RAG
```
