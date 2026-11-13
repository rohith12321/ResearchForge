ResearchForge
A Multi-Agent AI Research Assistant for Scientific Literature Synthesis

ResearchForge is a Python-based research assistant that helps investigate broad research questions by searching academic sources, retrieving relevant passages from papers, synthesizing findings, checking claims against source text, and iteratively reviewing the generated report.

The project currently focuses on AI/ML research and uses a LangGraph workflow with five specialized agents.
Features
- Research planning: breaks a broad question into focused subquestions.
- Academic search: searches arXiv and Wikipedia for relevant material.
- PDF processing: downloads available papers and extracts text using PyMuPDF.
- Semantic retrieval: chunks paper text, embeds it using Sentence Transformers, and retrieves relevant passages with FAISS.
- Literature synthesis: generates a structured report from retrieved evidence.
- Claim checking: checks source identifiers and evidence quotes associated with generated claims.
- Iterative review: uses a Critic agent to identify missing aspects and request follow-up research, with a limit of three iterations.
Workflow
Research Question
       |
       v
Research Manager
       |
       v
Researcher <-------------------+
       |                       |
       v                       |
Synthesizer                    |
       |                       |
       v                       |
Fact Checker                   |
       |                       |
       v                       |
Critic ---- Follow-up needed --+
       |
       v
Final Research Report
Tech Stack
Component	Technology
Language	Python
Agent orchestration	LangGraph
LLM	Gemini API
Academic search	arXiv Atom API, Wikipedia API
PDF text extraction	PyMuPDF
Embeddings	Sentence Transformers (paraphrase-MiniLM-L6-v2)
Vector search	FAISS (IndexFlatIP)


Semantic Scholar integration is present but currently disabled in the active search path due to API rate-limit issues.
Getting Started
Prerequisites
- Python 3.10 or newer
- Git
- A Gemini API key
- Internet connection
1. Clone the repository
git clone https://github.com/rohith12321/ResearchForge.git
cd ResearchForge
2. Create and activate a virtual environment
Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Linux / macOS
python -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
4. Configure the Gemini API
Set your Gemini API key using the environment variable expected by the application. For example, if the project configuration reads GEMINI_API_KEY:

Windows PowerShell
$env:GEMINI_API_KEY="YOUR_API_KEY"
Linux / macOS
export GEMINI_API_KEY="YOUR_API_KEY"
Do not commit API keys or other secrets to Git.
Running the Project
Run the complete research workflow from the repository root:
python scripts/run_research.py
Run the academic search component independently:
python scripts/run_search.py
The scripts are the command-line entry points. Check their source or --help output, if supported, for the current input arguments and options.
Running Tests
Run the test suite from the repository root:
python -m unittest discover -s tests -v
Project Structure
ResearchForge/
├── src/researchforge/
│   ├── agents/          # Research Manager, Researcher, Synthesizer, Fact Checker, Critic
│   ├── retrieval/       # Search, PDF processing, chunking, embeddings, FAISS retrieval
│   ├── pipeline/        # LangGraph state, nodes, and workflow graph
│   └── evaluation/      # Benchmark questions, metrics, and experiment runner
├── scripts/
│   ├── run_research.py  # Run the complete research workflow
│   └── run_search.py    # Run academic search independently
├── tests/               # Automated tests
├── docs/                # Detailed project documentation
├── requirements.txt
└── pyproject.toml
Documentation
Further details are available in the [`docs/`](docs/) directory:
- [Architecture](docs/architecture.md) — agents, workflow, shared state, and retrieval pipeline.
- [Implementation](docs/implementation.md) — implementation details of the main modules.
- [Evaluation](docs/evaluation.md) — benchmark setup, preliminary results, and limitations.
Known Limitations
- Evidence verification currently relies on exact quote matching, which can fail because of PDF extraction artifacts, whitespace differences, or paraphrasing.
- Subquestions are processed sequentially.
- Semantic Scholar is not enabled in the active search path.
- Current benchmark results are preliminary and should not be treated as proof of factual correctness or general superiority over conventional RAG.
Contributors
- Rohith Rajeev Nambiar
- Alok Kumar
- Aman Agrawal
- Harsh Verma