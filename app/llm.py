import httpx

from app.config import Settings


def build_context(chunks: list[dict]) -> str:
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[Source {index}: {chunk['filename']} / {chunk['chunk_id']}]\n{chunk['text']}"
        )
    return "\n\n".join(lines)


def extractive_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I could not find relevant information in the uploaded documents."
    context = build_context(chunks)
    return (
        "Ollama is unavailable, so this is an extractive answer based on the "
        f"most relevant retrieved text for: {question}\n\n{context}"
    )


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    return (
        "Answer the user's question using only the provided document context. "
        "Keep the answer concise, with a maximum of three short sentences. "
        "If the context does not contain the answer, say that the uploaded documents "
        "do not provide enough information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def generate_ollama_answer(prompt: str, settings: Settings) -> str:
    response = httpx.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": 0.1,
                "num_predict": settings.ollama_num_predict,
                "num_ctx": 2048,
            },
        },
        timeout=settings.ollama_timeout_seconds,
    )
    response.raise_for_status()
    return str(response.json().get("response", "")).strip()


def warm_ollama_model(settings: Settings) -> str:
    response = httpx.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": "Reply with ready.",
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": 8,
                "num_ctx": 512,
            },
        },
        timeout=settings.ollama_timeout_seconds,
    )
    response.raise_for_status()
    return str(response.json().get("response", "")).strip()


def generate_answer(question: str, chunks: list[dict], settings: Settings) -> tuple[str, str]:
    if not chunks:
        return "I could not find relevant information in the uploaded documents.", "none"

    prompt = build_prompt(question, chunks)

    try:
        ollama_answer = generate_ollama_answer(prompt, settings)
        if ollama_answer:
            return ollama_answer, "ollama"
    except httpx.HTTPError:
        pass

    return extractive_answer(question, chunks), "local_extractive"
