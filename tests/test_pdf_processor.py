from src.researchforge.retrieval.academic_search import search_arxiv
from src.researchforge.retrieval.pdf_processor import process_paper


def test_pdf_processing():
    # Search for one paper
    papers = search_arxiv(
        query="retrieval augmented generation",
        max_results=1
    )

    assert len(papers) > 0, "No papers found on arXiv."

    paper = papers[0]

    # Download and process the paper
    processed = process_paper(
        paper_id=paper.paper_id,
        title=paper.title,
        pdf_url=paper.pdf_url
    )

    # Verify extracted information
    assert processed.title == paper.title
    assert processed.paper_id == paper.paper_id
    assert len(processed.text) > 0
    assert processed.pdf_path.endswith(".pdf")

    print("\nPDF processing successful!")
    print("Title:", processed.title)
    print("Extracted characters:", len(processed.text))
    print("PDF location:", processed.pdf_path)


if __name__ == "__main__":
    test_pdf_processing()
