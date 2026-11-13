from src.researchforge.agents.researcher import Researcher
from src.researchforge.agents.synthesizer import Synthesizer
from src.researchforge.agents.fact_checker import FactChecker


def test_researcher():
    researcher = Researcher(max_papers=3, top_k=5)

    result = researcher.research(
        "What are the applications of model-agnostic meta-learning?"
    )

    print("\nResearch completed!")
    print("Subquestion:", result["subquestion"])
    print("Papers processed:", len(result["papers"]))
    print("Evidence chunks retrieved:", len(result["evidence"]))

    for item in result["evidence"]:
        print("\nScore:", round(item.score, 3))
        print("Paper:", item.chunk.title)
        print("Evidence:", item.chunk.text[:500])

    assert len(result["papers"]) > 0
    assert len(result["evidence"]) > 0
    # Test Synthesizer
    synthesizer = Synthesizer()

    synthesis = synthesizer.synthesize(
        research_question="What are the applications of model-agnostic meta-learning?",
        research_results=[result]
    )

    print("\n========== SYNTHESIS ==========\n")
    print(synthesis)

    assert len(synthesis) > 0

    print("\nSynthesizer test passed!")

    # Test Fact Checker
    fact_checker = FactChecker()

    verified_claims = fact_checker.check(
        synthesis=synthesis,
        research_results=[result]
    )

    print("\n========== FACT CHECK RESULTS ==========\n")

    for claim in verified_claims:
        print("Claim:", claim["claim"])
        print("Verdict:", claim["verdict"])
        print("Explanation:", claim["explanation"])
        print("Evidence:", claim["evidence"])
        print("-" * 60)

    assert len(verified_claims) > 0

    print("\nFact Checker test passed!")


if __name__ == "__main__":
    test_researcher()