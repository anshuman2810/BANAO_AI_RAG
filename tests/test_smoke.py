from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


def test_upload_and_query_txt_document() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        ollama_base_url="http://127.0.0.1:9"
    )
    client = TestClient(app)

    upload_response = client.post(
        "/documents",
        files={
            "file": (
                "rag.txt",
                "RAG stores document chunks as embeddings and retrieves them for questions.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 202
    job_id = upload_response.json()["job_id"]

    job_response = client.get(f"/jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "completed"

    query_response = client.post(
        "/query",
        json={"question": "What does RAG store?", "top_k": 2, "min_similarity": 0.0},
    )

    assert query_response.status_code == 200
    body = query_response.json()
    assert body["sources"]
    assert body["metrics"]["retrieved_chunks"] >= 1
    app.dependency_overrides.clear()
