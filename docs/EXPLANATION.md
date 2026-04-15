# RAG System Explanation

## Chunk Size Choice

The backend uses a default chunk size of 900 characters with 150 characters of overlap.

This size balances retrieval precision and context completeness:

- Smaller chunks are easier to match during similarity search but can lose surrounding meaning.
- Larger chunks preserve more context but often retrieve irrelevant neighboring text.
- 900 characters usually holds one or two compact paragraphs, which is enough context for common question answering while keeping the LLM prompt small.
- 150 characters of overlap reduces the chance that a key sentence split across two chunks disappears from retrieval.

The chunking logic also attempts to end chunks on sentence boundaries when possible, which makes retrieved context cleaner for answer generation.

## Retrieval Failure Case Observed

One failure case appears with broad or indirect questions.

Example:

```text
Question: What problem does this project solve?
```

If the document says "retrieval-augmented generation combines information retrieval with language generation" but never uses the words "problem" or "solve", the local hashing embedding can rank a less useful chunk above the best explanatory chunk. This happens because the lightweight local embedding model is based on lexical overlap and hashed word n-grams, so it is weaker at semantic paraphrases than neural embeddings.

Mitigation strategies:

- Chunk overlap keeps related details together.
- Multiple top chunks are returned instead of only one.
- Similarity scores are exposed so weak retrieval can be detected.
- The embedding layer can be replaced with Sentence Transformers or another semantic model later.

## Metric Tracked

The `/query` endpoint tracks query latency in milliseconds.

The response also returns the best similarity score and number of retrieved chunks:

```json
{
  "metrics": {
    "latency_ms": 12.34,
    "best_similarity": 0.42,
    "retrieved_chunks": 3
  }
}
```

Latency is useful because a RAG API has several steps: query embedding, vector search, context construction, and LLM generation. Tracking latency helps identify whether slow responses come from retrieval or from the LLM call.

Similarity score is useful because low scores usually mean the uploaded documents do not contain a strong answer or the query wording does not match the document wording.

## Framework Choice

The implementation avoids heavy RAG frameworks such as LangChain or LlamaIndex so the core system behavior remains transparent:

- chunking behavior
- embedding generation
- vector persistence
- background ingestion
- retrieval filtering
- answer generation
- API validation and rate limiting

