from src.researchforge.retrieval.chunking import chunk_text
from src.researchforge.retrieval.retriever import Retriever


def test_retriever():
    texts = [
        "MAML is a meta-learning algorithm designed for few-shot learning.",
        "Transformers use self-attention mechanisms to process sequences.",
        "Model-agnostic meta-learning allows models to adapt quickly to new tasks.",
        "Convolutional neural networks are commonly used for image classification."
    ]

    chunks = []

    for i, text in enumerate(texts):
        chunks.extend(
            chunk_text(
                text=text,
                paper_id=f"paper-{i}",
                title=f"Paper {i}",
                chunk_size=1000,
                overlap=100
            )
        )

    retriever = Retriever()
    retriever.add_documents(chunks)

    results = retriever.retrieve(
        query="What is meta-learning and how does MAML work?",
        top_k=2
    )

    print("\nRetrieved results:")

    for result in results:
        print(f"\nScore: {result.score:.3f}")
        print(f"Text: {result.chunk.text}")

    assert len(results) == 2
    assert "MAML" in results[0].chunk.text or "meta-learning" in results[0].chunk.text.lower()

    print("\nRetriever test passed!")


if __name__ == "__main__":
    test_retriever()