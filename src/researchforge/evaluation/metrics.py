
import re
from typing import Any


def calculate_aspect_coverage(
    report: str,
    expected_aspects: list[str]
) -> dict[str, Any]:
    """
    Estimate topical coverage by checking whether expected aspects
    are mentioned in the generated report.

    This is a lexical baseline, not semantic evaluation.
    """
    if not expected_aspects:
        return {
            "coverage": 0.0,
            "covered_aspects": [],
            "missing_aspects": []
        }

    report_lower = report.lower()

    covered = []
    missing = []

    for aspect in expected_aspects:
        keywords = [
            word.lower()
            for word in re.findall(r"\b[a-zA-Z0-9]+\b", aspect)
            if len(word) > 2
        ]

        if not keywords:
            missing.append(aspect)
            continue

        matches = sum(
            1 for keyword in keywords
            if keyword in report_lower
        )

        # Consider an aspect covered if at least half its keywords appear.
        if matches / len(keywords) >= 0.5:
            covered.append(aspect)
        else:
            missing.append(aspect)

    coverage = len(covered) / len(expected_aspects)

    return {
        "coverage": round(coverage, 3),
        "covered_aspects": covered,
        "missing_aspects": missing
    }



def calculate_citation_validity(
    report: str,
    research_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Check whether arXiv paper IDs cited in the report
    exist in the retrieved research results.
    """

    cited_ids = set(
        re.findall(
            r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?",
            report
        )
    )

    available_ids = set()

    for result in research_results:

        # ProcessedPaper objects
        for paper in result.get("papers", []):
            paper_id = (
                paper.get("paper_id")
                if isinstance(paper, dict)
                else getattr(paper, "paper_id", None)
            )

            if paper_id:
                available_ids.add(str(paper_id))

        # Retrieved evidence objects
        for evidence in result.get("evidence", []):
            chunk = getattr(evidence, "chunk", None)

            if chunk is not None:
                paper_id = getattr(chunk, "paper_id", None)

                if paper_id:
                    available_ids.add(str(paper_id))

    valid_ids = cited_ids.intersection(available_ids)
    invalid_ids = cited_ids - available_ids

    validity = (
        len(valid_ids) / len(cited_ids)
        if cited_ids else 0.0
    )

    return {
        "validity": round(validity, 3),
        "cited_ids": sorted(cited_ids),
        "valid_ids": sorted(valid_ids),
        "invalid_ids": sorted(invalid_ids),
        "total_citations": len(cited_ids)
    }




def calculate_fact_check_statistics(
    fact_check_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Calculate the distribution of fact-check verdicts."""

    verdicts = {
        "supported": 0,
        "partially_supported": 0,
        "unsupported": 0
    }

    claims = []

    for result in fact_check_results:
        # Support both direct claim lists and wrapped results.
        if isinstance(result, dict) and "claims" in result:
            claims.extend(result["claims"])
        elif isinstance(result, dict) and "verdict" in result:
            claims.append(result)

    for claim in claims:
        verdict = claim.get("verdict")

        if verdict in verdicts:
            verdicts[verdict] += 1

    total = sum(verdicts.values())

    rates = {
        key: round(value / total, 3) if total else 0.0
        for key, value in verdicts.items()
    }

    return {
        "total_claims": total,
        "verdict_counts": verdicts,
        "verdict_rates": rates
    }



def calculate_critic_metrics(
    critique: dict[str, Any]
) -> dict[str, Any]:
    """Extract quality scores assigned by the Critic agent."""

    score_fields = [
        "relevance",
        "evidence_coverage",
        "evidence_grounding",
        "depth",
        "overall"
    ]

    scores = {}

    for field in score_fields:
        value = critique.get(field)

        if isinstance(value, (int, float)):
            scores[field] = value

    if scores:
        average_score = round(
            sum(scores.values()) / len(scores), 2
        )
    else:
        average_score = 0.0

    return {
        "scores": scores,
        "average_score": average_score,
        "recommendation": critique.get("recommendation"),
        "missing_aspects": critique.get("missing_aspects", []),
        "strengths": critique.get("strengths", []),
        "weaknesses": critique.get("weaknesses", [])
    }


def evaluate_research_state(
    state: dict[str, Any],
    expected_aspects: list[str]
) -> dict[str, Any]:
    """
    Evaluate a completed ResearchForge state using all available metrics.
    """

    report = state.get("final_report") or state.get("synthesis", "")

    aspect_coverage = calculate_aspect_coverage(
        report,
        expected_aspects
    )

    citation_validity = calculate_citation_validity(
        report,
        state.get("research_results", [])
    )

    fact_check = calculate_fact_check_statistics(
        state.get("fact_check_results", [])
    )

    critic = calculate_critic_metrics(
        state.get("critique", {})
    )

    return {
        "research_question": state.get("research_question", ""),
        "completion_status": state.get("completion_status", "unknown"),
        "iterations": state.get("iteration_count", 0),
        "aspect_coverage": aspect_coverage,
        "citation_validity": citation_validity,
        "fact_check": fact_check,
        "critic": critic
    }
