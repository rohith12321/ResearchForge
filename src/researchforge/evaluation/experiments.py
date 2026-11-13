
import json
from pathlib import Path
from datetime import datetime
from statistics import mean

from src.researchforge.evaluation.datasets import get_benchmark_dataset
from src.researchforge.evaluation.metrics import evaluate_research_state
from scripts.run_research import run_research


def run_benchmark(
    max_iterations: int = 3,
    output_path: str = "data/benchmark_results.json"
) -> dict:
    """Run ResearchForge on the complete benchmark dataset."""

    dataset = get_benchmark_dataset()

    results = []

    print("\n" + "=" * 60)
    print("RESEARCHFORGE BENCHMARK EVALUATION")
    print("=" * 60)

    for index, item in enumerate(dataset, 1):
        question = item["question"]

        print(f"\n\nBenchmark {index}/{len(dataset)}")
        print(f"Category: {item['category']}")
        print(f"Question: {question}")

        try:
            state = run_research(
                research_question=question,
                max_iterations=max_iterations
            )

            if state is None:
                raise RuntimeError("Research pipeline returned no result.")

            evaluation = evaluate_research_state(
                state=state,
                expected_aspects=item["expected_aspects"]
            )

            results.append({
                "category": item["category"],
                "question": question,
                "status": "success",
                "evaluation": evaluation
            })

            print("\nBenchmark evaluation completed.")

        except Exception as error:
            print(f"\nBenchmark failed: {error}")

            results.append({
                "category": item["category"],
                "question": question,
                "status": "failed",
                "error": str(error)
            })

    successful_results = [
        result["evaluation"]
        for result in results
        if result["status"] == "success"
    ]

    # Aggregate metrics across successful experiments.
    if successful_results:
        aggregate = {
            "successful_runs": len(successful_results),
            "failed_runs": len(results) - len(successful_results),

            "average_aspect_coverage": round(
                mean(
                    r["aspect_coverage"]["coverage"]
                    for r in successful_results
                ), 3
            ),

            "average_citation_validity": round(
                mean(
                    r["citation_validity"]["validity"]
                    for r in successful_results
                ), 3
            ),

            "average_critic_score": round(
                mean(
                    r["critic"]["average_score"]
                    for r in successful_results
                ), 2
            ),

            "total_fact_checked_claims": sum(
                r["fact_check"]["total_claims"]
                for r in successful_results
            ),

            "verdict_counts": {
                verdict: sum(
                    r["fact_check"]["verdict_counts"][verdict]
                    for r in successful_results
                )
                for verdict in [
                    "supported",
                    "partially_supported",
                    "unsupported"
                ]
            }
        }

    else:
        aggregate = {
            "successful_runs": 0,
            "failed_runs": len(results)
        }

    benchmark_report = {
        "experiment": "ResearchForge Benchmark Evaluation",
        "timestamp": datetime.now().isoformat(),
        "max_iterations": max_iterations,
        "total_questions": len(dataset),
        "aggregate_metrics": aggregate,
        "individual_results": results
    }

    # Save results.
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(benchmark_report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(json.dumps(aggregate, indent=4))

    print(f"\nResults saved to: {output_file.resolve()}")

    return benchmark_report


if __name__ == "__main__":
    run_benchmark()
