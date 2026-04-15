import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class HashingEmbeddingModel:
    """Small local embedding model for reproducible screening demos."""

    def __init__(self, dimensions: int = 384) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.vectorizer.n_features), dtype=np.float32)
        vectors = self.vectorizer.transform(texts)
        vectors = normalize(vectors, norm="l2", axis=1)
        return vectors.astype(np.float32).toarray()

