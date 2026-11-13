from dataclasses import dataclass

import faiss
import numpy as np

from src.researchforge.retrieval.chunking import TextChunk


@dataclass
class RetrievedChunk:
    chunk: TextChunk
    score: float


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add(self, chunks: list[TextChunk], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        if len(chunks) == 0:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimensions do not match the index.")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding, dtype=np.float32
        ).reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension does not match.")

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(score)
                )
            )

        return results