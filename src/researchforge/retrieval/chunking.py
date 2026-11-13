from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: int
    paper_id: str
    title: str
    text: str
    start_index: int
    end_index: int


def chunk_text(
    text: str,
    paper_id: str,
    title: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[TextChunk]:
    """
    Split document text into overlapping chunks.

    chunk_size: Maximum number of characters per chunk.
    overlap: Number of characters shared between consecutive chunks.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    title=title,
                    text=chunk,
                    start_index=start,
                    end_index=end
                )
            )

            chunk_id += 1

        if end == len(text):
            break

        start = end - overlap

    return chunks
