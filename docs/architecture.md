System Architecture
Overview
ResearchForge represents literature review as a state-driven workflow orchestrated by LangGraph. Each processing node reads from and updates a shared ResearchState. Conditional routing determines whether the system should conduct another research iteration or return the final report.
Workflow
START
  |
  v
Research Manager
  |
  v
Researcher
  |
  v
Synthesizer
  |
  v
Fact Checker
  |
  v
Critic
  |
  +---- Follow-up required ----> Prepare Follow-up ----+
  |                                                    |
  +---- No follow-up / iteration limit ----------------+
                         |
                         v
                    Finalize
                         |
                        END
The workflow is capped at three iterations to prevent unbounded agent execution. The Researcher currently processes subquestions sequentially.
Agents and responsibilities
Research Manager
Receives the user's research question and decomposes it into approximately 5–8 subquestions. It plans the investigation; paper retrieval is handled by the Researcher.
Researcher
Processes the subquestions, searches the supported sources, downloads available papers, extracts and chunks their text, and retrieves relevant passages. The current configuration processes at most three papers per subquestion and retrieves the top five chunks for a subquestion. Retrieved chunks retain their source paper identifiers and titles.
Synthesizer
Uses retrieved evidence to generate a structured report. Its prompt instructs it to ground the report in the retrieved passages and organize the output into sections such as key findings, agreements and disagreements, and limitations.
Fact Checker
Extracts claims and associated evidence quotes from the synthesis. It checks that referenced paper IDs exist and that the supplied quote occurs as an exact substring in the corresponding retrieved text. It assigns Supported, Partially Supported, or Unsupported verdicts.

Important limitation: the exact-substring requirement is sensitive to whitespace, PDF extraction artifacts, and paraphrasing. A failed exact match does not by itself prove that a claim is false.
Critic
Reviews the report for relevance, evidence coverage, evidence grounding, and depth. It identifies missing aspects and can propose follow-up questions for another research iteration.
Shared state
The workflow passes a typed dictionary between nodes. Its principal fields are:
Field	Purpose
research_question	Original user question.
subquestions	Questions produced by the Research Manager.
research_results	Retrieved papers and evidence chunks.
synthesis	Draft report produced by the Synthesizer.
fact_check_results	Claim-level verification results.
critique	Critic feedback, missing aspects, and follow-up questions.
iteration_count	Current workflow iteration.
max_iterations	Maximum allowed iterations.
final_report	Report returned to the user.
completion_status	Workflow completion state.
processed_paper_ids	Paper identifiers already processed.


Retrieval flow
1. Search arXiv and Wikipedia for relevant material.
2. Download available paper PDFs.
3. Extract text using PyMuPDF.
4. Split text into overlapping character-based chunks.
5. Generate 384-dimensional embeddings using paraphrase-MiniLM-L6-v2.
6. Normalize embeddings and index them with FAISS IndexFlatIP.
7. Embed each subquestion and retrieve its most relevant chunks.
Because vectors are normalized, inner-product ranking is equivalent to cosine-similarity ranking.
External services
- arXiv Atom API: academic paper metadata and PDF links.
- Wikipedia API: general background information.
- Gemini API: agent reasoning, planning, synthesis, and critique.
- Sentence Transformers: local text embedding generation.
- FAISS: local vector similarity search.
Semantic Scholar integration exists but is disabled in the active search path at present because of API rate-limit issues.