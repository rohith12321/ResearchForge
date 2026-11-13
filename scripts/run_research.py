
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




def run_research(research_question=None, max_iterations=3):
    """Execute ResearchForge and return the final research state."""

    if research_question is None:
        research_question = input("\nEnter your research question: ").strip()

    if not research_question:
        print("Research question cannot be empty.")
        return None

    research_graph = build_research_graph()

    initial_state = {
        "research_question": research_question,
        "subquestions": [],
        "research_results": [],
        "synthesis": "",
        "fact_check_results": [],
        "critique": {},
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "final_report": "",
        "completion_status": "",
        "processed_paper_ids": []
    }

    print("\nStarting ResearchForge...\n")

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

    verdict_counts = {
        "supported": 0,
        "partially_supported": 0,
        "unsupported": 0
    }

    for claim in result["fact_check_results"]:
        verdict = claim.get("verdict")

        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    print(f"Verdict distribution: {verdict_counts}")

    for i, claim in enumerate(result["fact_check_results"], 1):
        print(f"\nClaim {i}: {claim['claim']}")
        print(f"Verdict: {claim['verdict']}")
        print(f"Explanation: {claim['explanation']}")
        print(f"Evidence: {claim['evidence']}")

    print("\nResearchForge execution completed.")

    return result


if __name__ == "__main__":
    run_research()
