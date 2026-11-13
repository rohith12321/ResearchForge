
import os
import json
from pathlib import Path

from src.researchforge.retrieval.academic_search import search_all
from src.researchforge.retrieval.pdf_processor import process_paper
from src.researchforge.retrieval.chunking import chunk_text
from src.researchforge.retrieval.retriever import Retriever
from src.researchforge.agents.synthesizer import Synthesizer


# Load environment variables from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

env_path = PROJECT_ROOT / ".env"

if env_path.exists():
    with env_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def run_baseline(question):

    print("\nStarting Conventional RAG baseline...\n")

    # 1. Search directly using the original question
    print("Searching for papers...")

    search_results = search_all(
        query=question,
        max_results=5
    )

    # Keep only academic papers with downloadable PDFs
    papers = [
        result for result in search_results
        if result.source == "arXiv"
        and result.pdf_url
        and result.paper_id
    ]

    # Avoid duplicate papers
    unique_papers = {}
    for paper in papers:
        unique_papers[paper.paper_id] = paper

    papers = list(unique_papers.values())[:3]

    if not papers:
        raise RuntimeError("No downloadable academic papers found.")

    print(f"\nSelected {len(papers)} papers.")

    # 2. Download PDFs and extract text
    all_chunks = []

    for paper in papers:
        try:
            print(f"\nProcessing: {paper.title}")

            processed = process_paper(
                paper_id=paper.paper_id,
                title=paper.title,
                pdf_url=paper.pdf_url
            )

            # 3. Split paper text into overlapping chunks
            chunks = chunk_text(
                text=processed.text,
                paper_id=processed.paper_id,
                title=processed.title
            )

            all_chunks.extend(chunks)

            print(f"Created {len(chunks)} chunks.")

        except Exception as error:
            print(f"Could not process {paper.title}: {error}")

    if not all_chunks:
        raise RuntimeError("No paper chunks were created.")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # 4. Create the vector store and add all chunks
    retriever = Retriever()
    retriever.add_documents(all_chunks)

    # 5. Retrieve the top 5 chunks using the original question
    print("\nRetrieving relevant evidence...")

    retrieved = retriever.retrieve(
        query=question,
        top_k=5
    )

    if not retrieved:
        raise RuntimeError("No relevant chunks were retrieved.")

    print(f"Retrieved {len(retrieved)} chunks.")

    # 6. Generate the answer in one pass using Gemini
    print("\nGenerating conventional RAG answer...")

    synthesizer = Synthesizer()

    research_results = [
        {
            "subquestion": question,
            "evidence": retrieved
        }
    ]

    answer = synthesizer.synthesize(
        research_question=question,
        research_results=research_results
    )

    # 7. Save the answer and retrieved evidence
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "conventional_rag_report.txt"
    evidence_path = output_dir / "conventional_rag_evidence.json"

    report_path.write_text(answer, encoding="utf-8")

    evidence_data = []

    for item in retrieved:
        evidence_data.append({
            "paper_id": item.chunk.paper_id,
            "title": item.chunk.title,
            "score": item.score,
            "text": item.chunk.text
        })

    evidence_path.write_text(
        json.dumps(evidence_data, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("CONVENTIONAL RAG REPORT")
    print("=" * 60)
    print(answer)

    print(f"\nReport saved to: {report_path}")
    print(f"Evidence saved to: {evidence_path}")
    print(f"Gemini model used: {synthesizer.model}")

    return answer


if __name__ == "__main__":

    question = (
        "How do Retrieval-Augmented Generation systems improve "
        "the factual reliability of Large Language Models, "
        "and what are their limitations?"
    )

    run_baseline(question)
