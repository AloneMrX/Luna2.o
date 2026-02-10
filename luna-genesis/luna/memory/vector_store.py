"""Vector memory store with sentence-transformers embeddings and FAISS persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import faiss
    import numpy as np


@dataclass
class VectorStoreConfig:
    index_path: Path
    docs_path: Path
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    """Persistent vector store for retrieval-augmented generation (RAG)."""

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.docs_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "VectorStore dependencies are missing. Install sentence-transformers, faiss-cpu, and numpy."
            ) from exc

        self._faiss = faiss
        self._np = np
        self._embedder = SentenceTransformer(self.config.embedding_model)
        self._dimension = int(self._embedder.get_sentence_embedding_dimension())

        self._documents = self._load_documents()
        self._index = self._load_index()
        self._reconcile_index_and_docs()

    def _load_documents(self) -> list[str]:
        if not self.config.docs_path.exists():
            self.config.docs_path.write_text("[]", encoding="utf-8")
            return []

        try:
            data = json.loads(self.config.docs_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(item) for item in data]
            return []
        except (json.JSONDecodeError, OSError):
            return []

    def _load_index(self) -> "faiss.Index":
        if self.config.index_path.exists():
            try:
                return self._faiss.read_index(str(self.config.index_path))
            except RuntimeError:
                # Corrupted/incompatible index; rebuild from documents.
                return self._faiss.IndexFlatL2(self._dimension)
        return self._faiss.IndexFlatL2(self._dimension)

    def _persist(self) -> None:
        self.config.docs_path.write_text(json.dumps(self._documents, indent=2), encoding="utf-8")
        self._faiss.write_index(self._index, str(self.config.index_path))

    def _embed(self, text: str) -> "np.ndarray":
        vector = self._embedder.encode([text], normalize_embeddings=True)
        return self._np.asarray(vector, dtype="float32")

    def _rebuild_index_from_documents(self) -> None:
        self._index = self._faiss.IndexFlatL2(self._dimension)
        if not self._documents:
            return

        embeddings = self._embedder.encode(self._documents, normalize_embeddings=True)
        matrix = self._np.asarray(embeddings, dtype="float32")
        self._index.add(matrix)

    def _reconcile_index_and_docs(self) -> None:
        index_count = int(self._index.ntotal)
        docs_count = len(self._documents)

        if index_count == docs_count:
            return

        self._rebuild_index_from_documents()
        self._persist()

    def add_document(self, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return

        embedding = self._embed(clean_text)
        self._index.add(embedding)
        self._documents.append(clean_text)
        self._persist()

    def search(self, query: str, k: int = 3) -> list[str]:
        clean_query = query.strip()
        if not clean_query or not self._documents:
            return []

        top_k = max(1, min(k, len(self._documents)))
        query_embedding = self._embed(clean_query)
        _, indexes = self._index.search(query_embedding, top_k)

        results: list[str] = []
        for idx in indexes[0]:
            if 0 <= idx < len(self._documents):
                results.append(self._documents[idx])
        return results
