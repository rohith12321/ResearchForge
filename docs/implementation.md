Implementation Notes
This document describes the current implementation, organized by the main code areas.
Project layout
- src/researchforge/agents/: individual agent implementations.
- src/researchforge/retrieval/: academic search, PDF processing, chunking, embeddings, vector indexing, and retrieval.
- src/researchforge/pipeline/: LangGraph state, nodes, and graph construction.
- src/researchforge/evaluation/: benchmark questions, metrics, and experiment execution.
- scripts/: executable entry points.
Retrieval modules
academic_search.py
Searches the arXiv Atom API and Wikipedia API and returns structured search results. Semantic Scholar support is present but currently disabled in the active search path because of rate-limit issues.
pdf_processor.py
Downloads available PDFs and extracts their text using PyMuPDF.
chunking.py
Splits extracted text into overlapping character-based chunks. Chunk metadata retains the source paper identifier and title so that evidence can be traced back to its source.
embeddings.py
Uses the Sentence Transformers model paraphrase-MiniLM-L6-v2 to generate 384-dimensional embeddings.
vector_store.py and retriever.py
Use FAISS IndexFlatIP to rank normalized embeddings by inner product. Since the vectors are normalized, this ranking corresponds to cosine similarity. The retriever returns evidence chunks together with source metadata.
Agent modules
research_manager.py
Creates approximately 5–8 subquestions from the original research question.
researcher.py
Runs the retrieval process for each subquestion. The current configuration processes subquestions sequentially, handles at most three papers per subquestion, and retrieves the top five chunks.
synthesizer.py
Uses the retrieved chunks to generate a structured research report. The prompt asks the model to use retrieved evidence and retain paper identifiers for citations.
fact_checker.py
Checks generated claims against their cited evidence. The current implementation combines LLM-based claim/evidence assessment with programmatic checks for valid paper IDs and exact quote-substring matches. The exact-match check is a known limitation and can reject text that differs only because of formatting or paraphrasing.
critic.py
Evaluates the draft for relevance, evidence coverage, grounding, and depth. It identifies missing aspects and follow-up questions.
Pipeline modules
- state.py defines the shared research state.
- nodes.py adapts agent and processing functions into workflow nodes.
- graph.py connects the nodes and defines conditional routing.
The workflow proceeds through planning, research, synthesis, fact checking, and critique. The Critic can route the workflow through a follow-up preparation step and back to the Researcher. Execution is bounded by a maximum of three iterations.
Running the code
From the repository root, activate the project virtual environment and install the dependencies:
pip install -r requirements.txt
pip install -e .
Run the search entry point:
python scripts/run_search.py
Run the research workflow:
python scripts/run_research.py
The exact arguments and configuration options are defined in the scripts. Configure the Gemini API credential using the mechanism expected by the project; keep credentials out of source control.
Implementation notes and known limitations
- Semantic Scholar is not currently part of the active search path.
- Subquestions are processed sequentially rather than concurrently.
- Vector storage is local and embeddings may be recomputed between runs.
- Exact-string quote validation is sensitive to PDF extraction and text formatting.
- The current implementation should not be described as proving factual correctness merely because a claim receives a particular automated verdict.