from src.researchforge.retrieval.academic_search import (
    Paper,
    SearchResult,
    build_arxiv_query,
    search_arxiv,
    search_semantic_scholar,
    search_wikipedia,
    search_all,
    print_results
)

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