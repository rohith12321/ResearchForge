
from src.researchforge.pipeline.graph import build_research_graph


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

    print("Recommendation:", result["critique"]["recommendation"])
    print("Iterations:", result["iteration_count"])
    print("Fact-check claims:", len(result["fact_check_results"]))

    print("\nResearchForge execution completed.")


if __name__ == "__main__":
    run_research()
