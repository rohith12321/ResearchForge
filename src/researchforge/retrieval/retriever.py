from src.researchforge.retrieval.embeddings import EmbeddingModel
from src.researchforge.retrieval.vector_store import VectorStore
from src.researchforge.retrieval.chunking import TextChunk


class Retriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def add_documents(self, chunks: list[TextChunk]):
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        self.vector_store.add(chunks, embeddings)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.embed_query(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )