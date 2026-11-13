from src.researchforge.agents.researcher import Researcher


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


if __name__ == "__main__":
    test_researcher()