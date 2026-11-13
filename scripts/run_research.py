
import os
from pathlib import Path

from src.researchforge.pipeline.graph import build_research_graph


def _load_env_file(env_path: Path) -> None:
    """
    Load key=value pairs from a .env file into os.environ.

    - Lines starting with # are treated as comments.
    - Surrounding quotes on values are stripped.
    - Existing environment variables are NOT overwritten.
    """
    if not env_path.exists():
        return

    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# Load .env from the project root (file next to this script's parent)
_project_root = Path(__file__).resolve().parent.parent
_load_env_file(_project_root / ".env")



def run_research():
    research_question = input("\nEnter your research question: ").strip()

    if not research_question:
        print("Research question cannot be empty.")
        return

    # Build the LangGraph workflow
    research_graph = build_research_graph()

    # Initialize the shared state
    initial_state = {
        "research_question": research_question,
        "subquestions": [],
        "research_results": [],
        "synthesis": "",
        "fact_check_results": [],
        "critique": {},
        "iteration_count": 0,
        "max_iterations": 3,
        "final_report": "",
    }

    print("\nStarting ResearchForge...\n")

    # Execute the complete workflow
    result = research_graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)

    print(result["final_report"])

    print("\n" + "=" * 60)
    print("RESEARCH EVALUATION")
    print("=" * 60)

    print(f"Recommendation: {result['critique']['recommendation']}")
    print(f"Iterations: {result['iteration_count']}")
    print(f"Fact-check claims: {len(result['fact_check_results'])}")
    print(f"Completion status: {result['completion_status']}")

    claims = result["fact_check_results"]

    verdict_counts = {
        "supported": 0,
        "partially_supported": 0,
        "unsupported": 0
    }

    for claim in claims:
        verdict = claim.get("verdict")

        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    print(f"Fact-check claims: {len(claims)}")
    print(f"Verdict distribution: {verdict_counts}")

    for i, claim in enumerate(claims, 1):
        print(f"\nClaim {i}: {claim['claim']}")
        print(f"Verdict: {claim['verdict']}")
        print(f"Explanation: {claim['explanation']}")
        print(f"Evidence: {claim['evidence']}")

        print("\nResearchForge execution completed.")


if __name__ == "__main__":
    run_research()
