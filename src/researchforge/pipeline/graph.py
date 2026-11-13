
from langgraph.graph import StateGraph, START, END

from src.researchforge.pipeline.state import ResearchState
from src.researchforge.pipeline.nodes import (
    research_manager_node,
    researcher_node,
    synthesizer_node,
    fact_checker_node,
    critic_node,
    prepare_follow_up_research,
    finalize_report,
)


def should_continue(state: ResearchState):
    """Decide whether to accept the report or conduct more research."""

    critique = state["critique"]

    if (
        critique["recommendation"] == "research_more"
        and state["iteration_count"] < state["max_iterations"]
    ):
        return "research_more"

    return "accept"


def build_research_graph():
    """Construct and compile the multi-agent research workflow."""

    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("research_manager", research_manager_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("critic", critic_node)
    graph.add_node("prepare_follow_up", prepare_follow_up_research)
    graph.add_node("finalize", finalize_report)

    # Initial workflow
    graph.add_edge(START, "research_manager")
    graph.add_edge("research_manager", "researcher")
    graph.add_edge("researcher", "synthesizer")
    graph.add_edge("synthesizer", "fact_checker")
    graph.add_edge("fact_checker", "critic")

    # Conditional routing after Critic evaluation
    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "research_more": "prepare_follow_up",
            "accept": "finalize",
        },
    )

    # Research loop
    graph.add_edge("prepare_follow_up", "researcher")

    # Finish
    graph.add_edge("finalize", END)

    return graph.compile()
