import json
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


class LocalVectorStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.directory / "embeddings.npy"
        self.metadata_path = self.directory / "metadata.json"
        self.lock = Lock()
        self.embeddings, self.metadata = self._load()

    def _load(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if self.embeddings_path.exists() and self.metadata_path.exists():
            embeddings = np.load(self.embeddings_path)
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return embeddings.astype(np.float32), metadata
        return np.empty((0, 0), dtype=np.float32), []

    def _persist(self) -> None:
        np.save(self.embeddings_path, self.embeddings)
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def add(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must have the same length")
        if len(embeddings) == 0:
            return

        with self.lock:
            if self.embeddings.size == 0:
                self.embeddings = embeddings.astype(np.float32)
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
            self.metadata.extend(metadata)
            self._persist()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        with self.lock:
            if self.embeddings.size == 0:
                return []
            scores = self.embeddings @ query_embedding.reshape(-1)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for index in top_indices:
                item = dict(self.metadata[int(index)])
                item["similarity"] = float(scores[int(index)])
                results.append(item)
            return results

    def delete_document(self, document_id: str) -> int:
        with self.lock:
            keep_indices = [
                index
                for index, item in enumerate(self.metadata)
                if item.get("document_id") != document_id
            ]
            deleted = len(self.metadata) - len(keep_indices)
            if deleted == 0:
                return 0
            self.metadata = [self.metadata[index] for index in keep_indices]
            if self.embeddings.size == 0 or not keep_indices:
                self.embeddings = np.empty((0, 0), dtype=np.float32)
            else:
                self.embeddings = self.embeddings[keep_indices]
            self._persist()
            return deleted

    def list_documents(self) -> list[dict[str, Any]]:
        documents: dict[str, dict[str, Any]] = {}
        with self.lock:
            for item in self.metadata:
                document_id = item["document_id"]
                if document_id not in documents:
                    documents[document_id] = {
                        "document_id": document_id,
                        "filename": item["filename"],
                        "content_type": item["content_type"],
                        "uploaded_at": item["uploaded_at"],
                        "chunks_indexed": 0,
                    }
                documents[document_id]["chunks_indexed"] += 1
        return list(documents.values())

