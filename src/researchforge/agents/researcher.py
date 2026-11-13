from src.researchforge.retrieval.academic_search import search_all
from src.researchforge.retrieval.pdf_processor import process_paper
from src.researchforge.retrieval.chunking import chunk_text
from src.researchforge.retrieval.retriever import Retriever


class Researcher:
    def __init__(self, max_papers=3, top_k=5):
        self.max_papers = max_papers
        self.top_k = top_k
        self.retriever = Retriever()

    def research(self, subquestion: str):
        # 1. Search for relevant papers and information
        search_query = subquestion

        if "model-agnostic meta-learning" in subquestion.lower():
            search_query = "meta-learning"

        search_results = search_all(search_query)

        # Keep only academic papers with PDF links
        papers = [
            result for result in search_results
            if result.source.lower() == "arxiv" and result.pdf_url
        ][:self.max_papers]

        all_chunks = []
        processed_papers = []

        # 2. Download and process papers
        for paper in papers:
            try:
                processed = process_paper(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    pdf_url=paper.pdf_url
                )

                processed_papers.append(processed)

                # 3. Split paper text into chunks
                chunks = chunk_text(
                    text=processed.text,
                    paper_id=processed.paper_id,
                    title=processed.title
                )

                all_chunks.extend(chunks)

            except Exception as error:
                print(f"Could not process {paper.title}: {error}")

        # 4. Add all chunks to the retriever
        self.retriever.add_documents(all_chunks)

        # 5. Retrieve evidence relevant to the subquestion
        evidence = self.retriever.retrieve(
            query=subquestion,
            top_k=self.top_k
        )

        return {
            "subquestion": subquestion,
            "papers": processed_papers,
            "evidence": evidence
        }