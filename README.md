# ResearchForge
### A Multi-Agent AI Research Assistant for Evidence-Grounded Scientific Literature Synthesis

ResearchForge is an AI-powered research assistant designed to automate scientific literature research. It aims to retrieve relevant academic papers, extract supporting evidence, synthesize findings across multiple sources, verify generated claims, identify disagreements, and suggest potential research gaps.

The initial research domain is **Artificial Intelligence and Machine Learning (AI/ML)**.

The project aims to go beyond conventional Retrieval-Augmented Generation (RAG) systems by introducing a structured multi-agent workflow with evidence verification and critical analysis.

---

# 1. Project Motivation

Conducting a literature review requires researchers to:

- Search across large collections of academic papers.
- Identify papers relevant to a research question.
- Read and extract important findings from multiple sources.
- Compare methodologies, assumptions and results.
- Identify disagreements and limitations in existing research.
- Establish what has already been investigated and what remains unexplored.
- Produce a well-structured review with reliable citations.

This process is time-consuming and becomes increasingly difficult as the volume of published research grows.

Large Language Models can assist with this process, but conventional LLM-based approaches have limitations:

- They may generate unsupported claims or hallucinate references.
- They may overlook important papers.
- They may fail to identify disagreements between studies.
- They may produce superficial summaries without adequately connecting findings across papers.
- They may suggest research gaps without sufficient supporting evidence.

ResearchForge aims to address these limitations through a controlled research workflow in which different components handle planning, retrieval, synthesis, verification and critical analysis.

# 2. Project Objectives

The main objectives are:

1. **Academic Paper Discovery:** Retrieve relevant scientific papers from academic sources such as arXiv and Semantic Scholar.

2. **Document Processing:** Download papers, extract their contents and divide them into meaningful passages.

3. **Evidence Retrieval:** Identify passages relevant to individual research questions and maintain references to their original papers.

4. **Research Planning:** Decompose complex research questions into smaller, manageable subquestions.

5. **Literature Synthesis:** Combine findings from multiple papers into a coherent research review.

6. **Citation Verification:** Verify that generated claims are actually supported by the cited evidence.

7. **Critical Analysis:** Identify contradictions, methodological differences, limitations and potential research gaps.

8. **Research Report Generation:** Produce structured reports with traceable references and clearly identified uncertainties.

9. **Experimental Evaluation:** Compare conventional single-pass RAG against the proposed multi-agent research workflow.

# 3. Proposed System Architecture

The intended workflow is:

```text
                  USER QUERY
                      |
                      v
              RESEARCH MANAGER
                      |
                      v
               RESEARCH PLANNING
                      |
                      v
              ACADEMIC RETRIEVAL
                      |
                      v
             PAPER DOWNLOADING
                      |
                      v
             PDF TEXT EXTRACTION
                      |
                      v
              DOCUMENT CHUNKING
                      |
                      v
             EVIDENCE RETRIEVAL
                      |
                      v
                SYNTHESIZER
                      |
                      v
               FACT CHECKER
                      |
                      v
                   CRITIC
                      |
                      v
              FINAL RESEARCH REPORT
```

The workflow will be implemented using **LangGraph**, allowing individual agents and processing components to communicate through a shared state.

## 3.1 Research Manager

The Research Manager receives the user's research question and determines how it should be investigated.

Responsibilities:
- Understand the research question.
- Break it into approximately 3–5 focused subquestions.
- Identify the information required to answer each subquestion.
- Coordinate the research workflow.
- Track the progress of individual research tasks.

## 3.2 Researcher

The Researcher investigates assigned subquestions.

Responsibilities:
- Search academic databases.
- Retrieve relevant papers.
- Identify useful passages from papers.
- Return structured evidence with source references.
- Report insufficient or missing evidence.

Multiple research tasks may eventually execute in parallel.

## 3.3 Synthesizer

The Synthesizer combines findings from the retrieved papers.

Responsibilities:
- Organize findings into common themes.
- Identify recurring conclusions.
- Compare findings across papers.
- Generate structured research claims.
- Maintain references to supporting evidence.

## 3.4 Fact Checker

The Fact Checker verifies the reliability of generated claims.

Responsibilities:
- Check whether cited passages support the associated claims.
- Identify unsupported or partially supported claims.
- Detect incorrect or irrelevant citations.
- Flag claims requiring qualification.
- Return verification results.

## 3.5 Critic

The Critic performs critical analysis of the literature.

Responsibilities:
- Identify contradictions between papers.
- Compare different methodologies and assumptions.
- Identify limitations in existing research.
- Highlight underexplored areas.
- Suggest potential research gaps.

Research-gap suggestions must be distinguished from established findings and supported by the available literature wherever possible.

## 3.6 Final Report Generator

The final component combines the research findings into a structured report.

Expected sections include:
- Research question and scope.
- Research methodology.
- Overview of relevant literature.
- Thematic synthesis of findings.
- Comparison of existing approaches.
- Contradictions and limitations.
- Potential research gaps.
- References and supporting evidence.

---

# 4. Research and Experimental Objective

A major part of ResearchForge is evaluating whether the multi-agent architecture actually improves research synthesis.

The central research question is:

**Does a controlled multi-agent research workflow with evidence verification and iterative refinement improve scientific literature synthesis over a single-pass RAG baseline, and what additional latency and cost does it introduce?**

We intend to compare the following configurations:

| Configuration | Description |
|---|---|
| A | Conventional single-pass RAG |
| B | RAG with improved retrieval and reranking |
| C | Multi-agent research synthesis |
| D | Multi-agent synthesis with citation verification and bounded refinement |

## 4.1 Evaluation Metrics

The proposed evaluation metrics include:

| Metric | Purpose |
|---|---|
| Recall@10 | Measures whether relevant papers or passages are retrieved. |
| Citation Validity | Measures whether cited references correspond to actual sources. |
| Citation Support Precision | Measures whether cited evidence genuinely supports generated claims. |
| Claim Support Rate | Measures the proportion of generated claims supported by evidence. |
| Coverage | Measures how comprehensively the system addresses the research question. |
| Contradiction Identification | Measures the ability to identify disagreements between studies. |
| Research Gap Quality | Evaluates the usefulness and evidence basis of suggested research gaps. |
| Latency | Measures end-to-end execution time. |
| API Cost | Measures the computational and LLM API expenditure per query. |

The same evaluation questions, source corpus and model configurations should be used across the experimental variants wherever possible.

Results should be recorded and compared rather than assuming that the multi-agent approach will necessarily perform better.

---

# 5. Technology Stack

The planned technology stack is:

| Technology | Purpose |
|---|---|
| Python | Primary development language |
| LangGraph | Multi-agent workflow orchestration |
| arXiv API | Academic paper discovery |
| Semantic Scholar API | Additional academic paper discovery |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| LLM API | Research planning, synthesis and verification |
| Streamlit | User interface |
| unittest / pytest | Automated testing |
| Git and GitHub | Version control and collaboration |

Dependencies will be introduced as the corresponding components are implemented.

---

# 6. Repository Structure

The project follows a modular Python `src` layout.

```text
ResearchForge/
│
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── src/
│   └── researchforge/
│       ├── __init__.py
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── research_manager.py
│       │   ├── researcher.py
│       │   ├── synthesizer.py
│       │   ├── fact_checker.py
│       │   └── critic.py
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── academic_search.py
│       │   ├── pdf_processor.py
│       │   ├── chunking.py
│       │   ├── embeddings.py
│       │   ├── vector_store.py
│       │   └── retriever.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── state.py
│       │   ├── graph.py
│       │   └── nodes.py
│       │
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── paper.py
│       │   ├── evidence.py
│       │   ├── claim.py
│       │   └── report.py
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py
│           ├── datasets.py
│           └── experiments.py
│
├── tests/
│   ├── test_academic_search.py
│   ├── test_pdf_processor.py
│   ├── test_chunking.py
│   ├── test_retriever.py
│   ├── test_agents.py
│   └── test_pipeline.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── docs/
│   ├── architecture.md
│   ├── implementation.md
│   └── evaluation.md
│
└── scripts/
    ├── run_search.py
    └── run_research.py
```

This represents the intended architecture. Files should be created incrementally when their corresponding functionality is implemented.

## 6.1 Directory Responsibilities

### `src/researchforge/agents/`

Contains the implementation of individual AI agents.

Each agent should have a clearly defined responsibility and communicate through structured inputs and outputs.

### `src/researchforge/retrieval/`

Contains all academic search and document retrieval functionality.

This includes paper discovery, PDF downloading, text extraction, chunking, embedding generation, vector storage and retrieval.

### `src/researchforge/pipeline/`

Contains LangGraph orchestration.

It manages the shared state, workflow nodes, execution order, conditional transitions and agent coordination.

It should coordinate existing components rather than duplicate their logic.

### `src/researchforge/schemas/`

Contains shared data models used across the application.

Examples include:
- Paper metadata.
- Evidence passages.
- Research claims.
- Verification results.
- Final research reports.

Stable identifiers should be maintained throughout the workflow so that generated claims can be traced back to their supporting passages and papers.

### `src/researchforge/evaluation/`

Contains the experimental evaluation framework, datasets, metrics and experiment execution logic.

### `tests/`

Contains automated unit and integration tests.

External API calls should be mocked in unit tests wherever practical.

### `data/raw/`

Stores original downloaded research papers.

### `data/processed/`

Stores extracted text, processed documents and generated chunks.

### `data/evaluation/`

Stores evaluation questions, reference evidence, experiment outputs and results.

Large datasets, PDFs and generated artifacts should generally not be committed to Git.

### `docs/`

Contains architecture documentation, implementation notes and experimental methodology.

### `scripts/`

Contains executable entry points for manually running individual components or complete workflows.

Core application logic should remain inside `src/researchforge/`.

---

# 7. Work Completed So Far

## 7.1 Repository and Environment Setup

Completed:
- Created the ResearchForge GitHub repository.
- Cloned the repository locally.
- Configured the local Git environment.
- Created and activated a Python virtual environment.
- Added `.gitignore` to exclude unnecessary files and generated artifacts.
- Established the initial Python `src` package structure.
- Created the initial retrieval module and test directory.

Development environment:
- Operating system: Windows.
- IDE: Visual Studio Code.
- Python: 3.13.5.
- Version control: Git and GitHub.

## 7.2 Academic Search Module

**Implemented file:** `src/researchforge/retrieval/academic_search.py`

The first functional component of ResearchForge is an academic search module using the arXiv API.

It currently supports:

1. Searching academic papers through the arXiv Atom API.
2. Constructing search queries programmatically.
3. Searching across all supported fields.
4. Searching specifically within paper titles.
5. Searching specifically within abstracts.
6. Exact-phrase searching.
7. Multi-term searching using AND conditions.
8. Configurable maximum result counts.
9. Configurable sorting by relevance, last updated date or submission date.
10. Configurable ascending and descending sorting.

The module parses arXiv XML responses and returns structured `Paper` dataclass objects.

The current `Paper` representation contains:

| Field | Description |
|---|---|
| `paper_id` | arXiv paper identifier |
| `title` | Paper title |
| `authors` | List of authors |
| `published` | Publication timestamp |
| `abstract` | Paper abstract |
| `pdf_url` | Direct PDF URL |

The module also validates search inputs and raises appropriate exceptions for invalid arguments and API or XML failures.

## 7.3 Unit Testing

**Implemented file:** `tests/test_academic_search.py`

The academic search module currently has 12 passing unit tests.

Test coverage includes:
- Correct parsing of paper metadata.
- Extraction of multiple authors.
- PDF link extraction.
- Handling empty search results.
- Empty-query validation.
- Invalid result-count validation.
- Malformed XML handling.
- General query construction.
- Exact-phrase query construction.
- Title-only search construction.
- Abstract-only search construction.
- Invalid search fields and sorting configurations.

External API calls are mocked during unit testing so that these tests do not depend on network connectivity.

## 7.4 Live API Testing

The search module has also been tested against the actual arXiv API.

A manual search for:

`retrieval augmented generation`

was executed using title-only searching.

The API successfully returned five papers with correctly populated:
- Titles.
- Author lists.
- Publication dates.
- Abstracts.
- PDF URLs.

This verifies that the implementation works beyond the mocked unit tests.

---

# 8. Current Development Status

The following table distinguishes implemented functionality from future work.

| Component | Status |
|---|---|
| GitHub repository setup | Completed |
| Python virtual environment | Completed |
| Initial project structure | Completed |
| Git ignore configuration | Completed |
| arXiv API integration | Implemented |
| XML parsing | Implemented |
| Structured Paper dataclass | Implemented |
| Search query construction | Implemented |
| Search field selection | Implemented |
| Sorting and result limits | Implemented |
| Academic search unit tests | 12 passing |
| Live arXiv API verification | Completed |
| PDF downloading | Not implemented |
| PDF text extraction | Not implemented |
| Document chunking | Not implemented |
| Embedding generation | Not implemented |
| Vector database / FAISS | Not implemented |
| Semantic retrieval | Not implemented |
| Single-pass RAG baseline | Not implemented |
| LangGraph workflow | Not implemented |
| Research agents | Not implemented |
| Citation verification | Not implemented |
| Critical analysis agent | Not implemented |
| Streamlit interface | Not implemented |
| Experimental evaluation | Not implemented |

The current implementation should be treated as the initial academic retrieval baseline, not as a completed research assistant.

---

# 9. Immediate Next Steps

Development should proceed incrementally. The immediate priority is to understand and strengthen the existing implementation before adding more features.

## Step 1: Understand the Existing Academic Search Module

Review:
- arXiv API request structure.
- URL parameter encoding.
- Atom XML format.
- XML parsing using ElementTree.
- Dataclasses and structured return values.
- Search query construction.
- Exception handling.
- Unit testing and mocking.

The goal is for the team to understand how the current implementation works before extending it.

## Step 2: Improve Academic Retrieval

After understanding the current implementation, consider:
- Supporting multiple search queries.
- Removing duplicate papers using stable identifiers.
- Combining results from multiple academic sources.
- Normalizing metadata across sources.
- Improving relevance and result ordering.

## Step 3: Implement PDF Processing

Implement `pdf_processor.py`.

Expected functionality:
- Download a paper using its PDF URL.
- Handle network failures and invalid responses.
- Save the original PDF.
- Extract text using PyMuPDF.
- Preserve page information.
- Handle empty or unreadable documents.

The output should be structured and suitable for downstream processing.

## Step 4: Implement Document Chunking

Implement `chunking.py`.

Expected functionality:
- Split extracted paper text into meaningful passages.
- Preserve page numbers and paper identifiers.
- Avoid losing source information during splitting.
- Support configurable chunk sizes and overlaps.

Every passage should retain a reference to its original paper.

## Step 5: Implement Embeddings and Retrieval

Implement:
- `embeddings.py`
- `vector_store.py`
- `retriever.py`

Expected functionality:
- Generate embeddings for document passages.
- Store passage embeddings.
- Retrieve relevant passages for research queries.
- Return passages with their associated paper metadata and source references.

FAISS and Sentence Transformers are the planned initial technologies.

## Step 6: Build the Single-Pass RAG Baseline

Before implementing multiple agents, establish a working RAG system.

Expected workflow:

```text
User Query
    |
    v
Academic Search
    |
    v
PDF Processing
    |
    v
Chunking
    |
    v
Embedding and Indexing
    |
    v
Passage Retrieval
    |
    v
LLM Generation
    |
    v
Answer with Citations
```

This baseline is essential because it provides the reference system against which the proposed multi-agent workflow will be evaluated.

## Step 7: Implement the Multi-Agent Workflow

Once the baseline works:
- Define shared schemas.
- Implement the Research Manager.
- Implement the Researcher.
- Implement the Synthesizer.
- Implement the Fact Checker.
- Implement the Critic.
- Integrate everything using LangGraph.
- Add bounded refinement and error handling.

## Step 8: Evaluation and User Interface

Finally:
- Construct a fixed evaluation dataset.
- Compare baseline RAG and multi-agent configurations.
- Calculate retrieval and evidence-support metrics.
- Measure latency and API costs.
- Inspect failures and unsupported claims.
- Build a Streamlit interface.
- Prepare the final report and demonstration.

---

# 10. Development Guidelines

All contributors should follow these principles:

1. **Incremental implementation:** Build and test individual components before integrating them.
2. **Modularity:** Keep retrieval, processing, agent logic, orchestration and evaluation separate.
3. **Structured data:** Use shared schemas rather than passing arbitrary dictionaries between components.
4. **Traceability:** Every evidence passage must retain its paper ID and location information.
5. **Evidence-grounded generation:** Generated claims should reference supporting evidence.
6. **Reproducibility:** Record configurations and outputs for experiments.
7. **Testing:** Introduce unit tests alongside new functionality.
8. **Controlled workflows:** Avoid unbounded agent loops and excessive API calls.
9. **Meaningful Git commits:** Commit working milestones rather than unrelated changes.
10. **Avoid premature complexity:** Do not implement planned components before they are needed.

---

# 11. How to Set Up the Current Project

Clone the repository:

```bash
git clone <GITHUB_REPOSITORY_URL>
cd ResearchForge
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

The current academic search implementation uses Python's standard library, so no third-party package installation is required for it.

Run the unit tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Expected result for the current academic search test suite:

```text
Ran 12 tests

OK
```

The manual search script can be used to test live arXiv connectivity:

```powershell
python run_search.py
```

Note: The manual script is currently a development/testing entry point. Its location may move into `scripts/` as the repository structure is finalized.

---

# 12. Project Milestones

| Milestone | Deliverable |
|---|---|
| M1 | Working academic search |
| M2 | PDF processing and extraction |
| M3 | Chunking, embeddings and retrieval |
| M4 | Single-pass RAG baseline |
| M5 | Multi-agent LangGraph workflow |
| M6 | Citation verification and critical analysis |
| M7 | Experimental evaluation |
| M8 | Streamlit interface, final report and demonstration |

Each milestone should have working code, tests and a clear acceptance criterion.

---

# 13. Project Philosophy

ResearchForge is intended to be a research-oriented engineering project, not merely a demonstration of calling multiple LLM agents.

The key contribution should be a systematic and experimentally evaluated research workflow that improves the reliability and depth of scientific literature synthesis.

The project should prioritize:

- Correct retrieval over simply retrieving more papers.
- Evidence-supported claims over fluent but unsupported summaries.
- Traceable citations over generated reference lists.
- Meaningful critical analysis over superficial paper summaries.
- Measurable improvements over assumed benefits of multi-agent architectures.

The final objective is to deliver a working research assistant and a credible experimental study of its strengths, limitations, computational cost and practical usefulness.