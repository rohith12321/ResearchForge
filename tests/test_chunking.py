
from src.researchforge.retrieval.chunking import chunk_text


def test_chunking():

    text = "A" * 2500

    chunks = chunk_text(
        text=text,
        paper_id="test-paper",
        title="Test Paper",
        chunk_size=1000,
        overlap=200
    )

    assert len(chunks) == 3

    assert len(chunks[0].text) == 1000
    assert len(chunks[1].text) == 1000
    assert len(chunks[2].text) == 900

    assert chunks[0].chunk_id == 0
    assert chunks[1].chunk_id == 1

    assert chunks[0].paper_id == "test-paper"
    assert chunks[0].title == "Test Paper"

    assert chunks[0].end_index - chunks[1].start_index == 200

    print("Chunking successful!")
    print("Total chunks:", len(chunks))


if __name__ == "__main__":
    test_chunking()
