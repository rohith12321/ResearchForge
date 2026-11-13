from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def embed_query(self, query: str):
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )