# RAG-Based Question Answering System

This project is a FastAPI backend for document-based question answering using a Retrieval-Augmented Generation workflow. It supports document upload, background ingestion, chunking, embedding, local vector search, and answer generation through a locally running Ollama model.

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

If Ollama is not running, the API returns an extractive answer using the retrieved chunks.

To use Ollama locally:

```bash
ollama pull llama3.2
ollama serve
```

## Run

```bash
uvicorn app.main:app --reload
```

Available endpoints:

- API docs: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>
- LLM warmup: <http://127.0.0.1:8000/llm/warmup>

## Latency Notes

Ollama response time depends heavily on the selected local model and machine resources. The backend keeps the model loaded for 15 minutes and limits generated output to reduce latency.

Recommended demo settings:

- Call `POST /llm/warmup` once before testing `/query`.
- Use `top_k` between `1` and `3` for shorter prompts.
- Use `OLLAMA_MODEL=llama3.2:1b` for faster answers on low-resource machines, or `OLLAMA_MODEL=llama3.2` for better answer quality.
- Reduce `OLLAMA_NUM_PREDICT` to `80` or `120` for shorter, faster answers.

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

## System Explanation

Detailed design notes are available in `docs/EXPLANATION.md`.

## Repository Link

https://github.com/anshuman2810/BANAO_AI_RAG
