from src.researchforge.retrieval.academic_search import search_arxiv

papers = search_arxiv(
    query="retrieval augmented generation",
    max_results=5,
    field="ti",
    exact_phrase=False,
    sort_by="relevance",
    sort_order="descending",
)

for i, paper in enumerate(papers, 1):
    print(f"\n{'=' * 70}")
    print(f"{i}. {paper.title}")
    print(f"Authors: {', '.join(paper.authors)}")
    print(f"Published: {paper.published}")
    print(f"Abstract: {paper.abstract[:300]}...")
    print(f"PDF: {paper.pdf_url}")