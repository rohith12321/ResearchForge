
from src.researchforge.agents.research_manager import ResearchManager
from src.researchforge.agents.researcher import Researcher
from src.researchforge.agents.synthesizer import Synthesizer
from src.researchforge.agents.fact_checker import FactChecker
from src.researchforge.agents.critic import Critic

from src.researchforge.pipeline.state import ResearchState


def research_manager_node(state: ResearchState):
    """Decompose the original research question."""

    manager = ResearchManager()

    subquestions = manager.decompose_query(
        state["research_question"]
    )

    print(f"\nResearch Manager generated {len(subquestions)} subquestions.")

    return {
        "subquestions": subquestions
    }


def researcher_node(state: ResearchState):
    """Retrieve evidence while avoiding previously processed papers."""

    researcher = Researcher(
        seen_paper_ids=state.get("processed_paper_ids", [])
    )

    new_results = []

    for question in state["subquestions"]:
        print(f"\nResearching: {question}")

        result = researcher.research(question)
        new_results.append(result)

    all_results = state.get("research_results", []) + new_results

    processed_paper_ids = list(researcher.seen_paper_ids)

    print(f"\nUnique papers processed so far: {len(processed_paper_ids)}")

    return {
        "research_results": all_results,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "processed_paper_ids": processed_paper_ids
    }


def synthesizer_node(state: ResearchState):
    """Generate a report using all retrieved evidence."""

    synthesizer = Synthesizer()

    synthesis = synthesizer.synthesize(
        research_question=state["research_question"],
        research_results=state["research_results"]
    )

    return {
        "synthesis": synthesis
    }


def fact_checker_node(state: ResearchState):
    """Check the generated report against retrieved evidence."""

    fact_checker = FactChecker()

    results = fact_checker.check(
        synthesis=state["synthesis"],
        research_results=state["research_results"]
    )

    return {
        "fact_check_results": results
    }


def critic_node(state: ResearchState):
    """Evaluate the report, including its factual reliability."""

    critic = Critic()

    critique = critic.evaluate(
        research_question=state["research_question"],
        synthesis=state["synthesis"],
        research_results=state["research_results"],
        fact_check_results=state["fact_check_results"]
    )

    return {
        "critique": critique
    }


def prepare_follow_up_research(state: ResearchState):
    """Prepare the Critic's follow-up questions for another research pass."""

    questions = state["critique"].get("follow_up_questions", [])

    print("\nPreparing follow-up research:")
    for question in questions:
        print(f"- {question}")

    return {
        "subquestions": questions
    }


def finalize_report(state: ResearchState):
    """Store the final report and its completion status."""

    status = (
        "accepted"
        if state["critique"]["recommendation"] == "accept"
        else "max_iterations_reached"
    )

    return {
        "final_report": state["synthesis"],
        "completion_status": status
    }